# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""SDXL export parity (slow, opt-in): the exported Core AI components against the torch modules.

    CAIPACK_SDXL_SOURCE=<diffusers SDXL folder> CAIPACK_PARITY_WORK=<work dir> \\
        uv run pytest -m slow tests/test_sdxl_parity.py -s

Exports the four components at --precision 4bit and at fp16, runs each .aimodel with
coreai.runtime (GPU specialization, like the app) and compares it with the torch module in float32
on identical inputs: both text encoders on three prompts, the UNet on a real mid-denoise latent, the
float32 VAE decoder on real final latents, and an end-to-end render driven by the exported
components (DPM-Solver++ 2M, linspace spacing, NumPy noise, zero unconditional embeddings -- what
the app's Swift pipeline does) against diffusers with the same latents and scheduler.

Every run writes ``parity-results.json`` and the PNGs into the work dir. The thresholds below are
the first run's measurements times 1.25 (measured values in the comments); they are measured,
never chosen.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

SEED = 42
STEPS = 25
GUIDANCE = 5.0
UNET_STEP = 10
TAGS = ", masterpiece, high score, great score, absurdres"
PROMPTS = {
    "girl": "1girl, solo, long silver hair, blue eyes, school uniform, standing in a cherry blossom park, "
            "looking at viewer, smile, full body" + TAGS,
    "scenery": "no humans, scenery, mountain lake at sunrise, pine forest, mist, reflections on water" + TAGS,
    "cafe": "1boy, 1girl, sitting at a cafe table, holding coffee cups, hands, window, city street, "
            "upper body" + TAGS,
}
# Measured 2026-10-08 (Animagine XL 4.0, M4 Pro GPU) -> threshold. Errors x1.25; cosines: the
# (1 - cos) gap x1.25; PSNR: the MSE x1.25 (-0.97 dB).
THRESHOLDS: dict[str, dict[str, float] | None] = {
    "fp16": {
        "te1_max_abs": 0.36777496,  # measured 0.29421997
        "te1_min_cosine": 0.99998345,  # measured 0.99998676
        "te2_max_abs": 0.18145561,  # measured 0.14516449
        "te2_pooled_max_abs": 0.0022608042,  # measured 0.0018086433
        "unet_max_abs": 0.0029978156,  # measured 0.0023982525
        "unet_cosine": 0.99999992,  # measured 0.99999994
        "vae_max_abs": 1.847744e-05,  # measured 1.4781952e-05
        "e2e_min_psnr_db": 32.685624,  # measured 33.654724
    },
    "4bit": {
        "te1_max_abs": 2.4945068,  # measured 1.9956055
        "te1_min_cosine": 0.83634394,  # measured 0.86907515
        "te2_max_abs": 7.2080794,  # measured 5.7664635
        "te2_pooled_max_abs": 0.23160331,  # measured 0.18528265
        "unet_max_abs": 0.2812586,  # measured 0.22500688
        "unet_cosine": 0.9995432,  # measured 0.99963456
        "vae_max_abs": 1.847744e-05,  # measured 1.4781952e-05
        "e2e_min_psnr_db": 15.035514,  # measured 16.004614
    },
}


def _env() -> tuple[Path, Path]:
    source, work = os.environ.get("CAIPACK_SDXL_SOURCE"), os.environ.get("CAIPACK_PARITY_WORK")
    if not source or not work:
        pytest.skip("set CAIPACK_SDXL_SOURCE (a diffusers SDXL folder) and CAIPACK_PARITY_WORK")
    return Path(source), Path(work)


def _export(source: Path, work: Path, precision: str) -> Path:
    from coreai_diffusion_converter import cli

    pack_id = f"parity-{precision}"
    bundle = work / precision / "export" / pack_id
    if (bundle / "metadata.json").is_file():
        return bundle
    rc = cli.main(["convert", str(source), "--target", "macos", "--name", f"parity {precision}",
                   "--precision", precision, "--allow-missing-license", "--overwrite",
                   "--output-dir", str(work / "packs"), "--work-dir", str(work / precision), "--keep-work"])
    assert rc == 0
    return bundle


async def _load(bundle: Path, asset: str):
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions

    model = await AIModel.load(bundle / f"{asset}.aimodel",
                               SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu()))
    return model.load_function("main")


def _np(out) -> np.ndarray:
    return np.asarray(out.numpy(), dtype=np.float32)


async def _call(fn, **inputs) -> dict[str, np.ndarray]:
    from coreai.runtime import NDArray

    out = await fn({k: NDArray(np.ascontiguousarray(v)) for k, v in inputs.items()})
    return {k: _np(v) for k, v in out.items()}


def _metrics(got: np.ndarray, want: np.ndarray, tokens: bool = False) -> dict[str, float]:
    got, want = got.astype(np.float64), want.astype(np.float64)
    diff = np.abs(got - want)
    out = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean())}
    if tokens:
        g, w = got.reshape(-1, got.shape[-1]), want.reshape(-1, want.shape[-1])
        cos = (g * w).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1) + 1e-12)
        out["min_cosine"] = float(cos.min())
        out["mean_cosine"] = float(cos.mean())
    else:
        g, w = got.ravel(), want.ravel()
        out["cosine"] = float((g * w).sum() / (np.linalg.norm(g) * np.linalg.norm(w) + 1e-12))
    return out


def _psnr(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(peak * peak / mse)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """[1, 3, H, W] in [-1, 1] -> [H, W, 3] uint8."""
    x = np.clip((image[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255)
    return x.round().astype(np.uint8)


def _ids(tokenizer, prompt: str) -> np.ndarray:
    return tokenizer(prompt, padding="max_length", max_length=77, truncation=True,
                     return_tensors="np").input_ids.astype(np.int32)


def _reference(pipe, prompt: str, device: str):
    """diffusers at seed 42 with the Swift pipeline's choices; returns (step-10 inputs, final latents, image)."""
    import torch

    from diffusers import DPMSolverMultistepScheduler

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="linspace")
    noise = np.random.RandomState(SEED).standard_normal((1, 4, 128, 128)).astype(np.float32)
    captured: dict = {}

    def capture(p, i, t, kw):
        if i == UNET_STEP - 1:
            captured["latents"] = kw["latents"].detach().float().cpu().numpy()
            captured["t_next"] = int(p.scheduler.timesteps[UNET_STEP])
        return kw

    with torch.no_grad():
        out = pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUIDANCE, height=1024, width=1024,
                   latents=torch.from_numpy(noise).to(device, pipe.unet.dtype), output_type="latent",
                   callback_on_step_end=capture, callback_on_step_end_tensor_inputs=["latents"])
        final = out.images.float()
        image = pipe.vae.decode(final / pipe.vae.config.scaling_factor).sample.float().cpu().numpy()
    return captured, final.cpu().numpy(), image, noise


async def _render(fns: dict, cond: np.ndarray, pooled: np.ndarray, noise: np.ndarray, scheduler_config,
                  scaling: float) -> np.ndarray:
    """The app's loop, in Python, over the exported components."""
    import torch

    from diffusers import DPMSolverMultistepScheduler

    sched = DPMSolverMultistepScheduler.from_config(scheduler_config, timestep_spacing="linspace")
    sched.set_timesteps(STEPS)
    context = np.concatenate([np.zeros_like(cond), cond], axis=0).astype(np.float16)
    pooled2 = np.concatenate([np.zeros_like(pooled), pooled], axis=0).astype(np.float16)
    tids = np.array([[1024, 1024, 0, 0, 1024, 1024]] * 2, dtype=np.float16)
    latents = torch.from_numpy(noise.copy())
    for t in sched.timesteps:
        sample = np.concatenate([latents.numpy()] * 2, axis=0).astype(np.float16)
        out = await _call(fns["Unet"], sample=sample, timestep=np.array([float(t)] * 2, dtype=np.float16),
                          encoder_hidden_states=context, text_embeds=pooled2, time_ids=tids)
        pred = out["noise_pred"]
        guided = pred[:1] + GUIDANCE * (pred[1:] - pred[:1])
        latents = sched.step(torch.from_numpy(guided), t, latents).prev_sample
    image = await _call(fns["VAEDecoder"], z=(latents.numpy() / scaling).astype(np.float32))
    return image["image"]


async def _run_precision(source: Path, work: Path, precision: str, pipe32, refs: dict) -> dict:
    import torch

    from coreai_diffusion_converter import _sdxl_export as X

    bundle = _export(source, work, precision)
    fns = {a: await _load(bundle, a) for a in ("TextEncoder", "TextEncoder2", "Unet", "VAEDecoder")}
    results: dict = {"text_encoder": {}, "text_encoder_2": {}, "end_to_end": {}}
    te1 = X.SDXLTextEncoderWrapper(pipe32.text_encoder.cpu().float()).eval()
    te2 = X.SDXLTextEncoder2Wrapper(pipe32.text_encoder_2.cpu().float()).eval()
    conds = {}
    for name, prompt in PROMPTS.items():
        ids1, ids2 = _ids(pipe32.tokenizer, prompt), _ids(pipe32.tokenizer_2, prompt)
        with torch.no_grad():
            want1 = te1(torch.from_numpy(ids1).long()).numpy()
            want2, wantp = (x.numpy() for x in te2(torch.from_numpy(ids2).long()))
        got1 = (await _call(fns["TextEncoder"], input_ids=ids1))["hidden_embeds"]
        out2 = await _call(fns["TextEncoder2"], input_ids=ids2)
        results["text_encoder"][name] = _metrics(got1, want1, tokens=True)
        results["text_encoder_2"][name] = {"hidden": _metrics(out2["hidden_embeds"], want2, tokens=True),
                                           "pooled": _metrics(out2["pooled_outputs"], wantp)}
        conds[name] = (np.concatenate([got1, out2["hidden_embeds"]], axis=-1), out2["pooled_outputs"])
    # UNet on the real step-10 latent of the girl prompt, with the torch conditioning.
    ref = refs["girl"]
    lat = ref["captured"]["latents"]
    cond, pooled = ref["cond"], ref["pooled"]
    t = ref["captured"]["t_next"]
    sample = np.concatenate([lat, lat], axis=0)
    context = np.concatenate([np.zeros_like(cond), cond], axis=0)
    pooled2 = np.concatenate([np.zeros_like(pooled), pooled], axis=0)
    tids = np.array([[1024, 1024, 0, 0, 1024, 1024]] * 2, dtype=np.float32)
    with torch.no_grad():
        dev = pipe32.unet.device
        want = pipe32.unet(torch.from_numpy(sample).to(dev), torch.tensor([float(t)] * 2, device=dev),
                           torch.from_numpy(context).to(dev),
                           added_cond_kwargs={"text_embeds": torch.from_numpy(pooled2).to(dev),
                                              "time_ids": torch.from_numpy(tids).to(dev)}).sample.float().cpu().numpy()
    got = (await _call(fns["Unet"], sample=sample.astype(np.float16), timestep=np.array([float(t)] * 2, np.float16),
                       encoder_hidden_states=context.astype(np.float16), text_embeds=pooled2.astype(np.float16),
                       time_ids=tids.astype(np.float16)))["noise_pred"]
    results["unet"] = _metrics(got, want)
    # VAE decoder on the final latents of the same reference run.
    scaling = pipe32.vae.config.scaling_factor
    z = (ref["final"] / scaling).astype(np.float32)
    got_img = (await _call(fns["VAEDecoder"], z=z))["image"]
    results["vae_decoder"] = {**_metrics(np.clip(got_img, -1, 1), np.clip(ref["image"], -1, 1)),
                              "psnr_db": _psnr(np.clip(got_img, -1, 1), np.clip(ref["image"], -1, 1), 2.0)}
    # End to end.
    from PIL import Image

    for name in PROMPTS:
        cond_x, pooled_x = conds[name]
        img = await _render(fns, cond_x, pooled_x, refs[name]["noise"], pipe32.scheduler.config, scaling)
        a, b = _to_uint8(img), _to_uint8(refs[name]["image"])
        Image.fromarray(a).save(work / f"e2e-{precision}-{name}.png")
        results["end_to_end"][name] = {"mean_abs_pixel": float(np.abs(a.astype(float) - b.astype(float)).mean()),
                                       "psnr_db": _psnr(a, b, 255.0), "pixel_std": float(a.std())}
    return results


def test_sdxl_components_match_torch():
    import torch

    from diffusers import StableDiffusionXLPipeline
    from PIL import Image

    source, work = _env()
    work.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe32 = StableDiffusionXLPipeline.from_pretrained(str(source), torch_dtype=torch.float32).to(device)
    pipe32.set_progress_bar_config(disable=True)
    refs = {}
    for name, prompt in PROMPTS.items():
        captured, final, image, noise = _reference(pipe32, prompt, device)
        with torch.no_grad():
            cond, _, pooled, _ = pipe32.encode_prompt(prompt, device=device, num_images_per_prompt=1,
                                                      do_classifier_free_guidance=False)
        refs[name] = {"captured": captured, "final": final, "image": image, "noise": noise,
                      "cond": cond.float().cpu().numpy(), "pooled": pooled.float().cpu().numpy()}
        Image.fromarray(_to_uint8(image)).save(work / f"reference-{name}.png")
    results = {}
    for precision in ("fp16", "4bit"):
        results[precision] = asyncio.run(_run_precision(source, work, precision, pipe32, refs))
    (work / "parity-results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    for precision, limits in THRESHOLDS.items():
        if limits is None:
            continue
        r = results[precision]
        assert max(m["max_abs"] for m in r["text_encoder"].values()) <= limits["te1_max_abs"]
        assert min(m["min_cosine"] for m in r["text_encoder"].values()) >= limits["te1_min_cosine"]
        assert max(m["hidden"]["max_abs"] for m in r["text_encoder_2"].values()) <= limits["te2_max_abs"]
        assert max(m["pooled"]["max_abs"] for m in r["text_encoder_2"].values()) <= limits["te2_pooled_max_abs"]
        assert r["unet"]["max_abs"] <= limits["unet_max_abs"] and r["unet"]["cosine"] >= limits["unet_cosine"]
        assert r["vae_decoder"]["max_abs"] <= limits["vae_max_abs"]
        assert min(m["psnr_db"] for m in r["end_to_end"].values()) >= limits["e2e_min_psnr_db"]
