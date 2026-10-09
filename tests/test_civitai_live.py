# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real Civitai API requests (opt-in: CAIPACK_LIVE=1 uv run pytest -m live). Metadata only; no
weight is downloaded. Pins the test subjects the fixtures were trimmed from."""

import os

import httpx
import pytest

from coreai_diffusion_converter import civitai

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client():
    if os.environ.get("CAIPACK_LIVE") != "1":
        pytest.skip("set CAIPACK_LIVE=1 to make real Civitai requests")
    c = civitai.CivitaiClient(os.environ.get(civitai.TOKEN_ENV))
    yield c
    c.close()


def test_animagine_checkpoint(client):
    v = client.resolve(civitai.CivitaiRef(1188071, 1408658), "model")
    assert v.model_type == "Checkpoint" and v.base_model == "SDXL 1.0" and v.family == "sdxl"
    assert v.file.name == "animagineXL40_v4Opt.safetensors" and v.file.fp == "fp16"
    assert v.file.sha256 == "6327eca98bfb6538dd7a4edce22484a1bbc57a8cff6b11d075d40da1afb847ac"


def test_perfect_eyes_lora(client):
    v = client.resolve(civitai.CivitaiRef(118427, 128461), "lora")
    assert v.model_type == "LORA" and v.file.name == "PerfectEyesXL.safetensors"
    assert "perfecteyes" in v.trained_words and len(v.file.sha256) == 64


def test_enums_still_list_every_mapped_base_model():
    if os.environ.get("CAIPACK_LIVE") != "1":
        pytest.skip("set CAIPACK_LIVE=1 to make real Civitai requests")
    data = httpx.get(f"{civitai.API}/enums", timeout=30).json()
    known = set(data["BaseModel"])
    mapped = set(civitai.BASE_MODEL_FAMILY) | set(civitai.DISTILLED_BASE_MODELS) | set(civitai.NEVER_BASE_MODELS)
    assert mapped <= known, sorted(mapped - known)
