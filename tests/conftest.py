# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIGS = FIXTURES / "configs"
PACKS = FIXTURES / "packs"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def family_tree(tmp_path):
    """Copy a family's config fixture into a temp tree; returns a factory(family) -> Path."""

    counter = iter(range(1000))

    def make(family: str, **edits) -> Path:
        n = next(counter)
        dst = tmp_path / (f"tree-{family}" if n == 0 else f"tree-{family}-{n}")
        shutil.copytree(CONFIGS / family, dst)
        for rel, changes in edits.items():
            p = dst / rel.replace("__json", ".json").replace("__", "/")
            data = json.loads(p.read_text())
            data.update(changes)
            p.write_text(json.dumps(data))
        return dst

    return make


def write_weights(tree: Path, components, fp16: bool = True, plain: bool = False) -> None:
    for c in components:
        (tree / c).mkdir(parents=True, exist_ok=True)
        if fp16:
            (tree / c / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
        if plain:
            (tree / c / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
