# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import zipfile

import pytest

from coreai_diffusion_converter import pack as packmod
from coreai_diffusion_converter import validate
from conftest import PACKS


def make_bundle(root, family="sd1"):
    def w(rel, data):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())

    w("LICENSE", "terms\n")
    w("CHANGES.md", "changes\n")
    w("metadata.json", json.dumps({"diffusion": {"type": "stable-diffusion", "image_size": 512,
                                                 "prediction_type": "epsilon"}}))
    for a in ("TextEncoder", "Unet", "VAEDecoder"):
        w(f"{a}.aimodel/main.mlirb", bytes(range(256)) * 8)
        w(f"{a}.aimodel/main.hash", "h")
        w(f"{a}.aimodel/metadata.json", "{}")
    for t in ("merges.txt", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"):
        w(f"tokenizer/{t}", "{}")
    return root


HEADER = {
    "id": "unit-pack", "name": "Unit Pack", "description": "", "family": "sd1", "pipeline": "stable_diffusion",
    "target": "ios", "supported_sizes": [512], "default_size": 512, "default_steps": 25, "max_steps": 50,
    "guidance_scale": 7.5, "scheduler": "dpmpp", "precision": "fp16", "compute_precision": "float16",
    "lazy_model_loading": True, "excluded_architectures": ["h13"],
    "source": {"kind": "folder", "ref": "unit", "revision": None},
    "conversion": {"clip_skip": 1, "vae": None, "prediction_type": "epsilon"},
    "license": {"name": "Test", "file": "LICENSE", "notice_file": None}, "attribution": "",
}


def build(tmp_path, name="a.caipack"):
    bundle = make_bundle(tmp_path / "bundle")
    p = packmod.build_pack_json(HEADER, bundle, created_at="2026-10-09T12:00:00Z")
    out = tmp_path / name
    packmod.write_pack(bundle, p, out)
    return bundle, p, out


def test_fields_assets_hashes(tmp_path):
    bundle, p, _ = build(tmp_path)
    assert p["format"] == "caipack" and p["format_version"] == 1
    assert p["assets"] == ["TextEncoder.aimodel", "Unet.aimodel", "VAEDecoder.aimodel", "metadata.json",
                           "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
                           "tokenizer/tokenizer_config.json", "tokenizer/vocab.json"]
    assert p["converter"]["exporter"] == "apple/coreai-models@7359dbcf6c3b"
    by_path = {f["path"]: f for f in p["files"]}
    data = (bundle / "Unet.aimodel/main.mlirb").read_bytes()
    assert by_path["Unet.aimodel/main.mlirb"] == {"path": "Unet.aimodel/main.mlirb", "size": len(data),
                                                  "sha256": hashlib.sha256(data).hexdigest()}
    assert [f["path"] for f in p["files"]] == sorted(by_path)


def test_archive_layout(tmp_path):
    _, _, out = build(tmp_path)
    with zipfile.ZipFile(out) as zf:
        infos = zf.infolist()
    names = [i.filename for i in infos]
    assert names[0] == "pack.json" and names[1:] == sorted(names[1:])
    for i in infos:
        assert i.compress_type == zipfile.ZIP_STORED and i.create_system == 3
        assert i.date_time == (1980, 1, 1, 0, 0, 0) and (i.external_attr >> 16) == 0o100644
        assert not i.is_dir() and i.comment == b""
    validate.validate_pack(out)


def test_two_builds_are_byte_identical(tmp_path):
    _, _, a = build(tmp_path / "x")
    _, _, b = build(tmp_path / "y")
    assert a.read_bytes() == b.read_bytes()


def test_injected_clock(tmp_path):
    bundle = make_bundle(tmp_path / "b")
    p = packmod.build_pack_json(HEADER, bundle, clock=lambda: "2030-01-01T00:00:00Z")
    assert p["created_at"] == "2030-01-01T00:00:00Z"


def test_streaming_zip64_writer_validates(tmp_path, monkeypatch):
    monkeypatch.setattr(packmod, "ZIP64_THRESHOLD", 1024)
    _, p, out = build(tmp_path)
    raw = out.read_bytes()
    assert b"\x01\x00" in raw  # ZIP64 extra header id present for the larger members
    validate.validate_pack(out)


def test_no_writestr_for_payload(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("writestr must not be used")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", boom)
    build(tmp_path)


def test_zip64_tiny_fixture_validates():
    p = validate.validate_pack(PACKS / "zip64_tiny.caipack", full=True)
    assert p["id"] == "zip64-tiny"
    with zipfile.ZipFile(PACKS / "zip64_tiny.caipack") as zf:
        assert all(i.create_system == 3 and i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())


def test_zip64_tiny_fixture_is_reproducible(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_zip64_tiny", PACKS / "make_zip64_tiny.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "z.caipack"
    mod.build(out)
    assert out.read_bytes() == (PACKS / "zip64_tiny.caipack").read_bytes()


def test_symlink_in_bundle_refused(tmp_path):
    bundle = make_bundle(tmp_path / "b")
    (bundle / "link").symlink_to(bundle / "LICENSE")
    with pytest.raises(ValueError):
        packmod.bundle_files(bundle)
