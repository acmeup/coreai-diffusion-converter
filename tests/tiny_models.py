# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tiny pipelines built in memory for the LoRA and SDXL tests: no Hub access, no tokenizers."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

TE_CFG = dict(num_hidden_layers=2, hidden_size=32, intermediate_size=37, num_attention_heads=4, vocab_size=1000,
              projection_dim=32, bos_token_id=0, eos_token_id=2, pad_token_id=1)
RANK = 4


def clip(with_projection: bool = False, layers: int = 2):
    import torch
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    torch.manual_seed(1 if with_projection else 0)
    cls = CLIPTextModelWithProjection if with_projection else CLIPTextModel
    return cls(CLIPTextConfig(**{**TE_CFG, "num_hidden_layers": layers})).eval()


def unet(cross_attention_dim: int, sdxl: bool):
    import torch
    from diffusers import UNet2DConditionModel

    torch.manual_seed(2)
    kw = dict(block_out_channels=(32, 64), layers_per_block=1, sample_size=8, in_channels=4, out_channels=4,
              down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
              up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), attention_head_dim=(2, 4),
              cross_attention_dim=cross_attention_dim, norm_num_groups=8)
    if sdxl:
        kw.update(use_linear_projection=True, addition_embed_type="text_time", addition_time_embed_dim=8,
                  transformer_layers_per_block=(1, 1), projection_class_embeddings_input_dim=32 + 6 * 8)
    return UNet2DConditionModel(**kw).eval()


def vae():
    from diffusers import AutoencoderKL

    return AutoencoderKL(block_out_channels=(32,), latent_channels=4, norm_num_groups=8,
                         down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",)).eval()


def sd1_pipe():
    from diffusers import PNDMScheduler, StableDiffusionPipeline

    return StableDiffusionPipeline(vae=vae(), text_encoder=clip(), tokenizer=None, unet=unet(32, False),
                                   scheduler=PNDMScheduler(), safety_checker=None, feature_extractor=None,
                                   requires_safety_checker=False)


def sdxl_pipe():
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

    return StableDiffusionXLPipeline(vae=vae(), text_encoder=clip(), text_encoder_2=clip(True), tokenizer=None,
                                     tokenizer_2=None, unet=unet(64, True), scheduler=EulerDiscreteScheduler())


def kohya(prefix: str, din: int, dout: int, alpha: float = 2.0, seed: int = 0) -> dict:
    import torch

    g = torch.Generator().manual_seed(seed)
    return {f"{prefix}.lora_down.weight": torch.randn(RANK, din, generator=g),
            f"{prefix}.lora_up.weight": torch.randn(dout, RANK, generator=g),
            f"{prefix}.alpha": torch.tensor(alpha)}


def expected_delta(state: dict, prefix: str, scale: float = 1.0):
    down, up = state[f"{prefix}.lora_down.weight"], state[f"{prefix}.lora_up.weight"]
    alpha = float(state[f"{prefix}.alpha"])
    return scale * alpha / down.shape[0] * (up @ down)


def save(state: dict, path) -> str:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in state.items()}, str(path))
    return str(path)
