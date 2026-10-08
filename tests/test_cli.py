# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from coreai_diffusion_converter import cli, exporter
from conftest import PACKS, write_weights


def never_export(*a, **k):
    raise AssertionError("the exporter must not run")


@pytest.fixture
def sd1_folder(family_tree):
    tree = family_tree("sd1")
    write_weights(tree, ("text_encoder", "unet", "vae"))
    (tree / "LICENSE").write_text("terms")
    return tree


def test_dry_run_never_exports(sd1_folder, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(exporter, "run_export", never_export)
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--dry-run",
                     "--output-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "family:          sd1" in out and "tree-sd1.ios.caipack" in out
    assert not (tmp_path / ".caipack-work").exists()


def test_bad_name_fails_before_export(sd1_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "run_export", never_export)
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--name", "x" * 81,
                     "--output-dir", str(tmp_path)]) == 2
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--steps", "0",
                     "--output-dir", str(tmp_path)]) == 2
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--steps", "51",
                     "--output-dir", str(tmp_path)]) == 2
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--guidance", "31",
                     "--output-dir", str(tmp_path)]) == 2


def test_usage_errors(sd1_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "run_export", never_export)
    with pytest.raises(SystemExit) as err:
        cli.main(["convert", str(sd1_folder)])
    assert err.value.code == 2
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--size", "768",
                     "--output-dir", str(tmp_path)]) == 2
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--clip-skip", "5",
                     "--output-dir", str(tmp_path)]) == 2


def test_missing_licence_exit_2(family_tree, tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "run_export", never_export)
    tree = family_tree("sd1")
    assert cli.main(["convert", str(tree), "--target", "ios", "--output-dir", str(tmp_path)]) == 2
    assert cli.main(["convert", str(tree), "--target", "ios", "--output-dir", str(tmp_path),
                     "--allow-missing-license", "--dry-run"]) == 0
    # A dry run reports the missing licence as a warning instead of stopping.
    assert cli.main(["convert", str(tree), "--target", "ios", "--output-dir", str(tmp_path), "--dry-run"]) == 0


def test_unsupported_exit_3(family_tree, tmp_path):
    tree = family_tree("sd1", model_index__json={"_class_name": "StableDiffusionXLPipeline"})
    assert cli.main(["convert", str(tree), "--target", "ios", "--output-dir", str(tmp_path)]) == 3
    ckpt = tmp_path / "m.ckpt"
    ckpt.write_bytes(b"x")
    assert cli.main(["convert", str(ckpt), "--target", "ios", "--output-dir", str(tmp_path)]) == 3


def test_overwrite_guard(sd1_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "run_export", never_export)
    (tmp_path / "My Pack.ios.caipack").write_bytes(b"old")
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--name", "My Pack",
                     "--output-dir", str(tmp_path)]) == 2
    assert (tmp_path / "My Pack.ios.caipack").read_bytes() == b"old"


def test_full_convert_with_a_fake_export(sd1_folder, tmp_path, monkeypatch):
    """The whole convert flow with the export replaced by a tiny bundle; checks naming, privacy,
    pack.json and self-validation."""

    def fake_export(tree, plan, *, pack_id, work_dir, licence_name, verbose=False):
        bundle = work_dir / "export" / pack_id
        for a in ("TextEncoder", "Unet", "VAEDecoder"):
            (bundle / f"{a}.aimodel").mkdir(parents=True)
            (bundle / f"{a}.aimodel" / "main.mlirb").write_bytes(b"w")
            (bundle / f"{a}.aimodel" / "main.hash").write_text("h")
            (bundle / f"{a}.aimodel" / "metadata.json").write_text("{}")
        (bundle / "tokenizer").mkdir()
        (bundle / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({"name_or_path": str(tree)}))
        (bundle / "metadata.json").write_text(json.dumps({
            "name": pack_id, "diffusion": {"type": "stable-diffusion", "image_size": 512, "prediction_type": "epsilon"},
            "source": {"hf_model_id": pack_id}, "compilation": {"date": "2026-10-08T08:00:00-04:00"}}))
        return bundle, {"prediction_type": "epsilon"}

    monkeypatch.setattr(exporter, "run_export", fake_export)
    out_dir = tmp_path / "out"
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--name", "Unit Model",
                     "--output-dir", str(out_dir)]) == 0
    pack = out_dir / "Unit Model.ios.caipack"
    assert pack.is_file() and not (out_dir / ".caipack-work").exists()
    assert cli.main(["validate", str(pack)]) == 0
    assert cli.main(["inspect", str(pack)]) == 0
    import zipfile

    with zipfile.ZipFile(pack) as zf:
        p = json.loads(zf.read("pack.json"))
        md = json.loads(zf.read("metadata.json"))
        tok = json.loads(zf.read("tokenizer/tokenizer_config.json"))
    assert p["id"] == "unit-model" and p["source"] == {"kind": "folder", "ref": sd1_folder.name, "revision": None}
    assert md["source"]["hf_model_id"] == sd1_folder.name and md["compilation"]["date"].endswith("Z")
    assert "name_or_path" not in tok


def test_validate_exit_codes(tmp_path):
    assert cli.main(["validate", str(PACKS / "zip64_tiny.caipack")]) == 0
    bad = tmp_path / "bad.caipack"
    bad.write_text("x")
    assert cli.main(["validate", str(bad)]) == 5
