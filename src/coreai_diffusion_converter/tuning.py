# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""--vae, --clip-skip and --prediction-type for SD 1.x / 2.x, applied to the loaded pipeline inside
the export worker, before tracing. The exported bundle simply contains the result."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .errors import UsageError

LOG = logging.getLogger(__name__)
PICKLE_SUFFIXES = (".ckpt", ".pt", ".pth", ".bin")


def load_vae(ref: str, dtype: Any, config_tree: str | Path | None = None) -> Any:
    """An AutoencoderKL from a .safetensors file, a folder or a Hub id (``vae/`` subfolder if any).

    A single file takes its configuration from ``config_tree/vae`` (the model being converted):
    the worker runs offline, so diffusers cannot fetch a default configuration from the Hub, and
    check_vae requires the same layout anyway."""
    from diffusers import AutoencoderKL

    p = Path(ref).expanduser()
    if p.suffix.lower() in PICKLE_SUFFIXES:
        raise UsageError(f"--vae: {p.suffix} files are pickle and are not loaded; use .safetensors")
    if p.is_file():
        if p.suffix.lower() != ".safetensors":
            raise UsageError("--vae must be a .safetensors file, a folder or a Hub id")
        if config_tree is not None and (Path(config_tree) / "vae" / "config.json").is_file():
            return AutoencoderKL.from_single_file(str(p), config=str(config_tree), subfolder="vae",
                                                  torch_dtype=dtype)
        return AutoencoderKL.from_single_file(str(p), torch_dtype=dtype)
    if p.is_dir():
        sub = "vae" if (p / "vae" / "config.json").is_file() else None
        folder = p / sub if sub else p
        weights = [w.name for w in folder.glob("*.safetensors")]
        variant = "fp16" if weights and all(w.endswith(".fp16.safetensors") for w in weights) else None
        return AutoencoderKL.from_pretrained(str(p), subfolder=sub, torch_dtype=dtype, variant=variant,
                                             use_safetensors=True)
    try:
        return AutoencoderKL.from_pretrained(ref, subfolder="vae", torch_dtype=dtype, use_safetensors=True)
    except OSError:
        return AutoencoderKL.from_pretrained(ref, torch_dtype=dtype, use_safetensors=True)


def check_vae(vae_config: Any, pipe_vae_config: Any) -> None:
    if getattr(vae_config, "latent_channels", None) != 4:
        raise UsageError("--vae: the VAE must have 4 latent channels (an SD 1.x / 2.x VAE)")
    if list(getattr(vae_config, "block_out_channels", [])) != list(pipe_vae_config.block_out_channels):
        raise UsageError("--vae: block_out_channels differ from the model's own VAE")


def clip_encoder(text_encoder: Any) -> Any:
    """The CLIP encoder module: ``text_model.encoder`` before transformers 5, ``encoder`` after."""
    inner = getattr(text_encoder, "text_model", text_encoder)
    return inner.encoder


def apply_clip_skip(text_encoder: Any, clip_skip: int) -> int:
    """Keep the first L - (N - 1) CLIP encoder layers. The exporter's text-encoder wrapper returns
    last_hidden_state, which CLIP computes after final_layer_norm on the last REMAINING layer, so the
    traced encoder equals diffusers' encode_prompt(clip_skip=N-1). Returns the kept layer count."""
    if not 1 <= clip_skip <= 4:
        raise UsageError("--clip-skip must be between 1 and 4")
    encoder = clip_encoder(text_encoder)
    layers = encoder.layers
    keep = len(layers) - (clip_skip - 1)
    if keep < 1:
        raise UsageError("--clip-skip removes every text-encoder layer")
    if clip_skip > 1:
        import torch

        encoder.layers = torch.nn.ModuleList(list(layers)[:keep])
        text_encoder.config.num_hidden_layers = keep
        LOG.info("clip skip %d: text encoder keeps %d of %d layers", clip_skip, keep, len(layers))
    return keep


def apply_prediction_type(scheduler: Any, prediction_type: str) -> None:
    if prediction_type not in ("epsilon", "v_prediction"):
        raise UsageError("--prediction-type must be epsilon or v_prediction")
    scheduler.register_to_config(prediction_type=prediction_type)


def effective_prediction_type(scheduler: Any) -> str:
    """What the exporter will write to metadata.json (its own fallback is epsilon)."""
    return getattr(scheduler.config, "prediction_type", None) or "epsilon"


def apply_tuning(pipe: Any, *, vae: str | None, clip_skip: int, prediction_type: str | None,
                 config_tree: str | Path | None = None) -> dict:
    """Apply the tuning to an SD 1.x / 2.x pipeline. Returns the effective values."""
    if vae:
        dtype = next(pipe.vae.parameters()).dtype
        new_vae = load_vae(vae, dtype, config_tree)
        check_vae(new_vae.config, pipe.vae.config)
        pipe.vae = new_vae
        LOG.info("VAE replaced")
    apply_clip_skip(pipe.text_encoder, clip_skip)
    if prediction_type:
        apply_prediction_type(pipe.scheduler, prediction_type)
    effective = effective_prediction_type(pipe.scheduler)
    LOG.info("prediction type: %s (%s)", effective, "set" if prediction_type else "inferred")
    return {"prediction_type": effective}
