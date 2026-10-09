# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from coreai_diffusion_converter import privacy
from coreai_diffusion_converter.errors import ExportError


def bundle(tmp_path):
    b = tmp_path / "bundle"
    (b / "Unet.aimodel").mkdir(parents=True)
    (b / "Unet.aimodel" / "main.mlirb").write_bytes(b"weights")
    (b / "Unet.aimodel" / "main.hash").write_text("h")
    (b / "tokenizer").mkdir()
    (b / "metadata.json").write_text(json.dumps({
        "name": "pack-id", "diffusion": {"type": "stable-diffusion", "prediction_type": "epsilon"},
        "source": {"hf_model_id": "pack-id"}, "compilation": {"date": "2026-10-08T08:11:20.901357-04:00"}}))
    return b


def test_metadata_normalised(tmp_path):
    b = bundle(tmp_path)
    md = privacy.normalise_metadata(b, "org/model")
    assert md["compilation"]["date"] == "2026-10-08T12:11:20.901357Z"
    assert md["source"]["hf_model_id"] == "org/model" and md["name"] == "pack-id"


def test_absolute_name_or_path_removed(tmp_path):
    b = bundle(tmp_path)
    cfg = b / "tokenizer" / "tokenizer_config.json"
    cfg.write_text(json.dumps({"name_or_path": str(tmp_path / "snap"), "model_max_length": 77}))
    (b / "tokenizer" / "config.json").write_text(json.dumps({"_name_or_path": "org/model"}))
    assert privacy.strip_name_or_path(b) == ["tokenizer/tokenizer_config.json"]
    assert json.loads(cfg.read_text()) == {"model_max_length": 77}
    assert json.loads((b / "tokenizer" / "config.json").read_text()) == {"_name_or_path": "org/model"}


def test_mlirb_inside_packages_survives_cleanup(tmp_path):
    b = bundle(tmp_path)
    (b / ".DS_Store").write_bytes(b"x")
    (b / "Unet.aimodel" / ".DS_Store").write_bytes(b"x")
    assert privacy.remove_ds_store(b) == 2
    assert (b / "Unet.aimodel" / "main.mlirb").read_bytes() == b"weights"


def test_other_hidden_files_are_reported_not_deleted(tmp_path):
    b = bundle(tmp_path)
    (b / ".secret").write_text("x")
    privacy.remove_ds_store(b)
    assert privacy.hidden_files(b) == [".secret"]


def test_planted_work_dir_path_fails(tmp_path):
    b = bundle(tmp_path)
    work = tmp_path / "work"
    (b / "tokenizer" / "vocab.json").write_text(json.dumps({"x": str(work / "tree")}))
    with pytest.raises(ExportError, match="tokenizer/vocab.json") as err:
        privacy.scan_text(b, [work])
    assert err.value.exit_code == 4
    privacy.scan_text(b, [tmp_path / "elsewhere"])


def test_binary_scan_warns(tmp_path):
    b = bundle(tmp_path)
    home = tmp_path / "home"
    (b / "Unet.aimodel" / "main.mlirb").write_bytes(b"\0" * 100 + str(home).encode() + b"\0")
    assert privacy.scan_binary(b, home) == ["Unet.aimodel/main.mlirb"]


def test_prediction_type_mismatch_fails():
    privacy.check_prediction_type({"diffusion": {"prediction_type": "epsilon"}}, "epsilon")
    with pytest.raises(ExportError):
        privacy.check_prediction_type({"diffusion": {"prediction_type": "epsilon"}}, "v_prediction")


def test_utc():
    assert privacy.to_utc("2026-10-07T21:30:58.556550-04:00") == "2026-10-08T01:30:58.556550Z"


def test_token_in_a_text_file_fails(tmp_path):
    b = bundle(tmp_path)
    (b / "CHANGES.md").write_text("token tok-SECRET-1")
    with pytest.raises(ExportError, match="Civitai API token") as err:
        privacy.scan_text(b, [], secrets=["tok-SECRET-1"])
    assert "tok-SECRET-1" not in str(err.value)
    (b / "CHANGES.md").write_text("clean")
    (b / "NOTICE").write_text("tok-SECRET-1")  # files without a suffix are scanned too
    with pytest.raises(ExportError):
        privacy.scan_text(b, [], secrets=["tok-SECRET-1"])
