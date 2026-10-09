# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate zip64_tiny.caipack: a few-KB, fully valid SD 1.x pack whose every member is
written with ZIP64 headers by the converter's own streaming writer.

    uv run python tests/fixtures/packs/make_zip64_tiny.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from coreai_diffusion_converter import pack as packmod  # noqa: E402

CREATED_AT = "2026-10-09T12:00:00Z"


def make_bundle(root: Path) -> None:
    def w(rel: str, text: str) -> None:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    w("LICENSE", "Sample licence text for the zip64_tiny fixture.\n")
    w("CHANGES.md", "Fixture pack. Not a real model.\n")
    w("metadata.json", json.dumps({
        "metadata_version": "0.2", "kind": "diffusion", "name": "zip64-tiny",
        "assets": {"text_encoder": "TextEncoder.aimodel", "unet": "Unet.aimodel",
                   "vae_decoder": "VAEDecoder.aimodel"},
        "diffusion": {"type": "stable-diffusion", "prediction_type": "epsilon", "image_size": 512},
        "source": {"model_definition": "torch", "hf_model_id": "zip64-tiny"},
        "compression": None, "compilation": {"date": CREATED_AT, "targets": []},
    }, indent=2) + "\n")
    for asset in ("TextEncoder", "Unet", "VAEDecoder"):
        w(f"{asset}.aimodel/main.mlirb", f"{asset} weights placeholder\n" * 4)
        w(f"{asset}.aimodel/main.hash", "0" * 64 + "\n")
        w(f"{asset}.aimodel/metadata.json", json.dumps({"component": asset}) + "\n")
    w("tokenizer/merges.txt", "#version: 0.2\na b\n")
    w("tokenizer/special_tokens_map.json", "{}\n")
    w("tokenizer/tokenizer_config.json", "{}\n")
    w("tokenizer/vocab.json", "{\"a\": 0, \"b\": 1}\n")


FIXTURE_CONVERTER_VERSION = "0.1.0"

HEADER = {
    "id": "zip64-tiny", "name": "ZIP64 Tiny", "description": "Test fixture, not a model.",
    "family": "sd1", "pipeline": "stable_diffusion", "target": "ios",
    "supported_sizes": [512], "default_size": 512, "default_steps": 25, "max_steps": 50,
    "guidance_scale": 7.5, "scheduler": "dpmpp", "precision": "fp16",
    "compute_precision": "float16", "lazy_model_loading": True, "excluded_architectures": ["h13"],
    "source": {"kind": "folder", "ref": "zip64-tiny", "revision": None},
    "conversion": {"clip_skip": 1, "vae": None, "prediction_type": "epsilon"},
    "license": {"name": "Sample Licence", "file": "LICENSE", "notice_file": None},
    "attribution": "",
}


def build(out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        make_bundle(bundle)
        pack = packmod.build_pack_json(HEADER, bundle, created_at=CREATED_AT)
        # Pinned to the converter version that first wrote the fixture, so a version bump does not
        # change the bytes every importer's copy of this file is compared against.
        pack["converter"]["version"] = FIXTURE_CONVERTER_VERSION
        saved = packmod.ZIP64_THRESHOLD
        packmod.ZIP64_THRESHOLD = 0  # every member gets ZIP64 headers
        try:
            packmod.write_pack(bundle, pack, out)
        finally:
            packmod.ZIP64_THRESHOLD = saved


if __name__ == "__main__":
    build(HERE / "zip64_tiny.caipack")
