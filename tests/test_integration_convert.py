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


def _texts(pack):
    with zipfile.ZipFile(pack) as zf:
        return {n: zf.read(n).decode("utf-8", "replace") for n in zf.namelist()
                if n.endswith((".json", ".txt", ".md")) or n in ("NOTICE", "LICENSE")}


def _raw(pack):
    """pack.json as written (validate_pack returns the decoded header, without `conversion`)."""
    with zipfile.ZipFile(pack) as zf:
        return json.loads(zf.read("pack.json"))


def _env(*names):
    values = [os.environ.get(n) for n in names]
    if not all(values):
        pytest.skip("set " + ", ".join(names))
    return values


def test_convert_sdxl_folder_for_macos(tmp_path):
    (source,) = _env("CAIPACK_INTEGRATION_SDXL_SOURCE")
    out = tmp_path / "out"
    rc = cli.main(["convert", source, "--target", "macos", "--name", "SDXL Integration", "--allow-missing-license",
                   "--output-dir", str(out), "--work-dir", str(tmp_path / "work")])
    assert rc == 0
    pack = out / "SDXL Integration.macos.caipack"
    p = validate.validate_pack(pack, full=True)
    assert p["family"] == "sdxl" and p["supported_sizes"] == [1024]
    assert sorted(a for a in p["assets"] if a.endswith(".aimodel")) == [
        "TextEncoder.aimodel", "TextEncoder2.aimodel", "Unet.aimodel", "VAEDecoder.aimodel"]
    texts = _texts(pack)
    md = json.loads(texts["metadata.json"])
    assert md["diffusion"]["type"] == "stable-diffusion-xl" and md["diffusion"]["image_size"] == 1024
    assert _raw(pack)["conversion"]["vae_precision"] == "float32"
    sizes = {f["path"].split("/")[0]: 0 for f in p["files"]}
    for f in p["files"]:
        sizes[f["path"].split("/")[0]] += f["size"]
    # An fp32 VAE decoder (49.5 M parameters) is about 4 bytes per weight: ~198 MB, not ~99 MB.
    assert sizes["VAEDecoder.aimodel"] > 150e6
    assert all(str(Path.home()) not in t for t in texts.values())


def test_convert_sdxl_with_a_local_kohya_lora(tmp_path):
    source, lora = _env("CAIPACK_INTEGRATION_SDXL_SOURCE", "CAIPACK_INTEGRATION_SDXL_LORA")
    out = tmp_path / "out"
    rc = cli.main(["convert", source, "--target", "macos", "--name", "SDXL Lora", "--allow-missing-license",
                   "--lora", f"{lora}:0.8", "--output-dir", str(out), "--work-dir", str(tmp_path / "work")])
    assert rc == 0
    validate.validate_pack(out / "SDXL Lora.macos.caipack", full=True)
    lo = _raw(out / "SDXL Lora.macos.caipack")["conversion"]["loras"][0]
    assert lo["name"] == Path(lora).name and lo["scale"] == 0.8 and lo["source"]["kind"] == "file"


def test_sd1_single_file_text_encoder_receives_the_lora(tmp_path):
    """The custom single-file TE path (transformers 5 flat CLIP) gets the LoRA: the exported text
    encoder differs from a no-LoRA export."""
    single, lora = _env("CAIPACK_INTEGRATION_SD1_SINGLE_FILE", "CAIPACK_INTEGRATION_SD1_LORA")
    hashes = {}
    for label, extra in (("plain", []), ("lora", ["--lora", lora])):
        out = tmp_path / label
        rc = cli.main(["convert", single, "--target", "ios", "--name", f"SD1 {label}", "--allow-missing-license",
                       *extra, "--output-dir", str(out), "--work-dir", str(tmp_path / f"w-{label}")])
        assert rc == 0
        p = validate.validate_pack(out / f"SD1 {label}.ios.caipack", full=True)
        hashes[label] = {f["path"]: f["sha256"] for f in p["files"]}
    te = "TextEncoder.aimodel/main.mlirb"
    assert te in hashes["plain"] and hashes["plain"][te] != hashes["lora"][te]


def test_flux2_transformer_lora(tmp_path):
    source, lora = _env("CAIPACK_INTEGRATION_FLUX2_SOURCE", "CAIPACK_INTEGRATION_FLUX2_LORA")
    out = tmp_path / "out"
    rc = cli.main(["convert", source, "--target", "macos", "--name", "Klein Lora", "--allow-missing-license",
                   "--lora", lora, "--output-dir", str(out), "--work-dir", str(tmp_path / "work")])
    assert rc == 0
    validate.validate_pack(out / "Klein Lora.macos.caipack", full=True)
