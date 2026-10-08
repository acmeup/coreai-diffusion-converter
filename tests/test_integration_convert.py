# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real conversion (slow, opt-in): CAIPACK_INTEGRATION_SOURCE=<SD 1.x diffusers folder>."""

import json
import os
import zipfile
from pathlib import Path

import pytest

from coreai_diffusion_converter import cli, validate

pytestmark = pytest.mark.slow


def test_convert_sd1_folder_for_ios(tmp_path):
    source = os.environ.get("CAIPACK_INTEGRATION_SOURCE")
    if not source:
        pytest.skip("set CAIPACK_INTEGRATION_SOURCE to a local SD 1.x diffusers folder")
    out = tmp_path / "out"
    work = tmp_path / "work"
    rc = cli.main(["convert", source, "--target", "ios", "--name", "Integration Pack",
                   "--allow-missing-license", "--output-dir", str(out), "--work-dir", str(work), "--keep-work"])
    assert rc == 0
    pack = out / "Integration Pack.ios.caipack"
    p = validate.validate_pack(pack, full=True)
    assert p["id"] == "integration-pack"
    assert (work / "export" / "integration-pack").is_dir()  # bundle dir is named by the pack id
    with zipfile.ZipFile(pack) as zf:
        md = json.loads(zf.read("metadata.json"))
        for name in zf.namelist():
            if name.endswith((".json", ".txt", ".md")):
                assert str(Path.home()) not in zf.read(name).decode("utf-8", "replace"), name
    assert md["name"] == "integration-pack"
    assert md["source"]["hf_model_id"] == Path(source).name
