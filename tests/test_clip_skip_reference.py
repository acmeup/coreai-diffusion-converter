# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Clip skip by truncation equals diffusers' clip_skip (slow, opt-in).

Source: CAIPACK_CLIP_SKIP_SOURCE (a local SD 1.x diffusers folder), else the public
hf-internal-testing/tiny-stable-diffusion-torch test model.

The reference is diffusers' StableDiffusionPipeline.encode_prompt(clip_skip=N-1). With
transformers 5 that method raises AttributeError (it addresses ``text_encoder.text_model``, which
transformers 5 flattened); the test then applies the same two lines diffusers uses
(``hidden_states[-(clip_skip + 1)]`` followed by ``final_layer_norm``) to the full encoder.
"""

import copy
import os

import pytest

pytestmark = pytest.mark.slow

PROMPT = "a lighthouse at dusk, oil painting"


def _weights_variant(src):
    from pathlib import Path

    p = Path(src) / "text_encoder"
    if p.is_dir():
        names = [f.name for f in p.glob("*.safetensors")]
        if names and all(n.endswith(".fp16.safetensors") for n in names):
            return "fp16"
    return None


@pytest.mark.parametrize("n", [2, 3])
def test_truncated_encoder_matches_diffusers_clip_skip(n):
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    from coreai_diffusion_converter.tuning import apply_clip_skip

    src = os.environ.get("CAIPACK_CLIP_SKIP_SOURCE") or "hf-internal-testing/tiny-stable-diffusion-torch"
    variant = _weights_variant(src)
    tok = CLIPTokenizer.from_pretrained(src, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(src, subfolder="text_encoder", torch_dtype=torch.float32,
                                       variant=variant).eval()
    ids = tok(PROMPT, padding="max_length", max_length=tok.model_max_length, truncation=True,
              return_tensors="pt").input_ids

    with torch.no_grad():
        try:
            from diffusers import StableDiffusionPipeline

            pipe = StableDiffusionPipeline.from_pretrained(src, torch_dtype=torch.float32, variant=variant,
                                                           safety_checker=None, feature_extractor=None)
            ref, _ = pipe.encode_prompt(PROMPT, "cpu", 1, False, clip_skip=n - 1)
            te = pipe.text_encoder
        except AttributeError:
            out = te(ids, output_hidden_states=True)
            ref = te.final_layer_norm(out.hidden_states[-(n - 1 + 1)])
        truncated = copy.deepcopy(te)
        apply_clip_skip(truncated, n)
        got = truncated(ids).last_hidden_state
    assert got.shape == ref.shape
    assert (got - ref).abs().max().item() <= 1e-5
