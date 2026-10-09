# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""SDXL export: Apple's coreai_models building blocks driven by the converter, because
export_diffusion has no SDXL registry and applies one dtype to every component.

Imported only by the export worker. Every Apple symbol is reached as an attribute of
``coreai_models.diffusion.pipeline`` (``P``) -- the worker patches some of them (``snapshot_download``,
``build_aimodel_metadata``) and the contract test pins them there. No Apple source is copied.

Components and their I/O (the app's SDXL pipeline binds them by these names and this order):

* ``TextEncoder`` (CLIP-L): ``input_ids`` [1, 77] -> ``hidden_embeds`` [1, 77, 768], the
  penultimate hidden state without the final layer norm (what diffusers' encode_prompt uses).
* ``TextEncoder2`` (OpenCLIP bigG): ``input_ids`` -> ``hidden_embeds`` [1, 77, 1280] (penultimate)
  and ``pooled_outputs`` [1, 1280] (the projected pooled ``text_embeds``).
* ``Unet``: ``sample`` [2, 4, 128, 128], ``timestep`` [2], ``encoder_hidden_states`` [2, 77, 2048],
  ``text_embeds`` [2, 1280], ``time_ids`` [2, 6] -> ``noise_pred``. Batch 2 = classifier-free
  guidance (unconditional first).
* ``VAEDecoder``: ``z`` [1, 4, 128, 128] float32 -> ``image``. The VAE decoder keeps float32 weights
  and compute: the stock SDXL VAE overflows in float16.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from .errors import ExportError

LOG = logging.getLogger("caipack.worker.sdxl")

UNET_INPUTS = ("sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids")
TE1_OUTPUTS = ("hidden_embeds",)
TE2_OUTPUTS = ("hidden_embeds", "pooled_outputs")
ASSETS = {"text_encoder": "TextEncoder", "text_encoder_2": "TextEncoder2", "unet": "Unet",
          "vae_decoder": "VAEDecoder"}
QUANTIZABLE = ("text_encoder", "text_encoder_2", "unet")
TOKEN_COUNT = 77
EDGE = 1024


class SDXLTextEncoderWrapper(torch.nn.Module):
    """CLIP-L: the penultimate hidden state, no final layer norm (diffusers encode_prompt)."""

    def __init__(self, text_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.model = text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids, output_hidden_states=True)
        return out.hidden_states[-2]


class SDXLTextEncoder2Wrapper(torch.nn.Module):
    """OpenCLIP bigG: (penultimate hidden state [1,77,1280], projected pooled text_embeds [1,1280])."""

    def __init__(self, text_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.model = text_encoder

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(input_ids, output_hidden_states=True)
        return out.hidden_states[-2], out.text_embeds


class SDXLUNetWrapper(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module, patch_upsample: bool = True) -> None:
        super().__init__()
        self.model = unet
        if patch_upsample:
            import coreai_models.diffusion.components as C

            C._patch_nearest_upsample(self.model)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor,
                text_embeds: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        return self.model(sample, timestep, encoder_hidden_states,
                          added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids}).sample


def time_ids(edge: int = EDGE) -> list[float]:
    """[orig_h, orig_w, crop_top, crop_left, target_h, target_w]."""
    return [float(edge), float(edge), 0.0, 0.0, float(edge), float(edge)]


def dummy_inputs(pipe: Any) -> dict[str, tuple[torch.Tensor, ...]]:
    unet = pipe.unet
    dtype = next(unet.parameters()).dtype
    size = int(unet.config.sample_size)
    channels = int(unet.config.in_channels)
    context = int(unet.config.cross_attention_dim)
    pooled = int(pipe.text_encoder_2.config.projection_dim)
    ids = torch.zeros(1, TOKEN_COUNT, dtype=torch.long)
    return {
        "text_encoder": (ids,),
        "text_encoder_2": (ids,),
        "unet": (torch.randn(2, channels, size, size, dtype=dtype),
                 torch.tensor([999.0, 999.0], dtype=dtype),
                 torch.randn(2, TOKEN_COUNT, context, dtype=dtype),
                 torch.randn(2, pooled, dtype=dtype),
                 torch.tensor([time_ids(size * 8)] * 2, dtype=dtype)),
        "vae_decoder": (torch.randn(1, int(pipe.vae.config.latent_channels), size, size, dtype=torch.float32),),
    }


def load_pipeline(tree: str, *, variant: str | None, sample_size: int | None, loras: Any,
                  tuning: dict) -> tuple[Any, dict]:
    """The SDXL pipeline ready to trace: LoRAs merged, tuning applied, VAE in float32."""
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    from .lora import apply_loras
    from .tuning import apply_tuning

    info: dict = {}
    pipe = StableDiffusionXLPipeline.from_pretrained(tree, torch_dtype=torch.float16, variant=variant)
    # USER DECISION: VAE decoder in float32. Load it from its own float32 weights rather than
    # upcasting the float16 copy.
    pipe.vae = AutoencoderKL.from_pretrained(tree, subfolder="vae", torch_dtype=torch.float32, variant=variant)
    if sample_size:
        LOG.info("sample_size %s -> %d (image edge %d)", pipe.unet.config.sample_size, sample_size, sample_size * 8)
        pipe.unet.register_to_config(sample_size=sample_size)
    if loras:
        info["loras"] = apply_loras(pipe, loras, "sdxl")
    info.update(apply_tuning(pipe, vae=tuning.get("vae"), clip_skip=1,
                             prediction_type=tuning.get("prediction_type"), config_tree=tree))
    pipe.vae = pipe.vae.float()
    return pipe, info


def wrappers(pipe: Any) -> dict[str, tuple[torch.nn.Module, tuple[str, ...], tuple[str, ...]]]:
    import coreai_models.diffusion.components as C

    return {
        "text_encoder": (SDXLTextEncoderWrapper(pipe.text_encoder), ("input_ids",), TE1_OUTPUTS),
        "text_encoder_2": (SDXLTextEncoder2Wrapper(pipe.text_encoder_2), ("input_ids",), TE2_OUTPUTS),
        "unet": (SDXLUNetWrapper(pipe.unet), UNET_INPUTS, ("noise_pred",)),
        "vae_decoder": (C.VAEDecoderWrapper(pipe.vae), ("z",), ("image",)),
    }


def check_tokenizers(out: Path) -> None:
    """_save_tokenizer only logs a warning on failure; the app cannot run without them."""
    for sub in ("tokenizer", "tokenizer_2"):
        for f in ("vocab.json", "merges.txt"):
            if not (out / sub / f).is_file():
                raise ExportError("the tokenizer was not exported")


def rewrite_metadata(out: Path, pipe: Any) -> dict:
    path = out / "metadata.json"
    md = json.loads(path.read_text(encoding="utf-8"))
    diffusion = md.setdefault("diffusion", {})
    diffusion["type"] = "stable-diffusion-xl"
    diffusion["force_zeros_for_empty_prompt"] = bool(pipe.config.get("force_zeros_for_empty_prompt", True))
    path.write_text(json.dumps(md, indent=2), encoding="utf-8")
    return md


async def export_sdxl(plan: dict, result: dict) -> None:
    import coreai_models.diffusion.pipeline as P

    pack_id = plan["pack_id"]
    out = Path(plan["out_root"]) / pack_id
    out.mkdir(parents=True, exist_ok=True)
    pipe, info = load_pipeline(plan["tree"], variant=plan.get("variant"), sample_size=plan.get("sample_size"),
                               loras=plan.get("loras") or (), tuning=plan.get("tuning") or {})
    result.update(info)
    quant = P._resolve_compression(plan["compression"])
    dummies = dummy_inputs(pipe)
    built = wrappers(pipe)
    results: dict[str, str] = {}
    for name in plan["components"]:
        wrapper, input_names, output_names = built[name]
        LOG.info("exporting %s -> %s.aimodel", name, ASSETS[name])
        program = P.export_stateless(wrapper, dummies[name], input_names, output_names)
        if quant is not None and name in QUANTIZABLE:
            LOG.info("quantizing %s", name)
            program = await P.apply_mlir_quantization(program, quant)
        asset = out / f"{ASSETS[name]}.aimodel"
        if asset.exists():
            import shutil

            shutil.rmtree(asset)
        # Through the module attribute, which the worker patched with the licence wrapper.
        program.save_asset(asset, P.build_aimodel_metadata(pack_id, component=ASSETS[name]))
        del program
        results[name] = str(asset)
    P._save_tokenizer(pack_id, out, pipe, overwrite=True)
    check_tokenizers(out)
    P._write_metadata_json(pipe, pack_id, "sd", out, plan["compression"], results)
    md = rewrite_metadata(out, pipe)
    result["prediction_type"] = (md.get("diffusion") or {}).get("prediction_type") or "epsilon"
    result["vae_precision"] = "float32"
