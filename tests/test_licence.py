# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest

from coreai_diffusion_converter import licence
from coreai_diffusion_converter.errors import UsageError


@pytest.mark.parametrize("name", ["LICENSE", "license.md", "LICENSE.txt", "LICENSE-MODEL", "license.txt"])
def test_discovery_variants(tmp_path, name):
    (tmp_path / name).write_text("terms")
    assert licence.discover(tmp_path).name == name


def test_discovery_none(tmp_path):
    (tmp_path / "README.md").write_text("x")
    assert licence.discover(tmp_path) is None
    assert licence.discover(None) is None


def test_card_mapping(tmp_path):
    assert licence.licence_display_name(None, ("creativeml-openrail-m", None)) == "CreativeML OpenRAIL-M"
    assert licence.licence_display_name(None, ("openrail++", None)) == "CreativeML Open RAIL++-M"
    assert licence.licence_display_name(None, ("apache-2.0", None)) == "Apache License 2.0"
    assert licence.licence_display_name(None, ("mit", None)) == "MIT License"
    assert licence.licence_display_name(None, ("other", "stabilityai-ai-community")) == licence.STABILITY_NAME
    assert licence.licence_display_name(None, ("bigscience-openrail-m", None)) == "bigscience-openrail-m"
    assert licence.licence_display_name(None, (None, None)) == "See LICENSE"
    assert licence.licence_display_name("Custom", ("mit", None)) == "Custom"


def test_card_front_matter(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("---\nlicense: other\nlicense_name: stabilityai-ai-community\ntags:\n- x\n---\n# Model\n")
    assert licence.read_card_license(readme) == ("other", "stabilityai-ai-community")
    readme.write_text("# no front matter")
    assert licence.read_card_license(readme) == (None, None)


def test_missing_licence_is_exit_2(tmp_path):
    with pytest.raises(UsageError, match="licence file is required") as err:
        licence.resolve(licence_file=None, licence_dir=tmp_path, name_override=None, card=(None, None),
                        allow_missing=False)
    assert err.value.exit_code == 2
    info = licence.resolve(licence_file=None, licence_dir=tmp_path, name_override=None, card=(None, None),
                           allow_missing=True)
    bundle = tmp_path / "b"
    bundle.mkdir()
    licence.write_files(info, bundle)
    assert info.file is None and info.notice_file is None and not (bundle / "LICENSE").exists()


def test_copied_as_license(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "license.txt").write_text("terms")
    (src / "NOTICE.md").write_text("notice text")
    info = licence.resolve(licence_file=None, licence_dir=src, name_override=None, card=("mit", None),
                           allow_missing=False)
    bundle = tmp_path / "b"
    bundle.mkdir()
    licence.write_files(info, bundle)
    assert (bundle / "LICENSE").read_text() == "terms"
    assert (bundle / "NOTICE").read_text() == "notice text"
    assert (info.file, info.notice_file, info.attribution) == ("LICENSE", "NOTICE", "")


def test_stability_notice(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "LICENSE.md").write_text("STABILITY AI COMMUNITY LICENSE AGREEMENT")
    info = licence.resolve(licence_file=None, licence_dir=src, name_override=None,
                           card=("other", "stabilityai-ai-community"), allow_missing=False)
    bundle = tmp_path / "b"
    bundle.mkdir()
    licence.write_files(info, bundle)
    notice = (bundle / "NOTICE").read_text()
    assert ("This Stability AI Model is licensed under the Stability AI Community License, "
            "Copyright © Stability AI Ltd. All Rights Reserved") in notice
    assert "Powered by Stability AI" in notice
    assert info.notice_file == "NOTICE" and info.attribution == "Powered by Stability AI"
    by_flag = licence.resolve(licence_file=src / "LICENSE.md", licence_dir=None,
                              name_override="Stability AI Community License", card=(None, None),
                              allow_missing=False)
    assert by_flag.stability


def test_changes_md_has_no_path_and_records_tuning(tmp_path):
    vae_dir = tmp_path / "my-vae"
    vae_dir.mkdir()
    info = licence.ChangesInfo(exporter_commit="7359dbcf6c3b", components=["text_encoder", "unet", "vae_decoder"],
                               compression="none", compute_precision="float16", size=512, vae=str(vae_dir),
                               clip_skip=2, prediction_type="v_prediction", prediction_type_overridden=True,
                               family="sd1")
    licence.write_changes(info, tmp_path)
    text = (tmp_path / "CHANGES.md").read_text()
    assert str(tmp_path) not in text and "my-vae" in text
    assert "Clip skip 2" in text and "v_prediction (set explicitly)" in text
    assert "apple/coreai-models 7359dbcf6c3b" in text and "safety checker" in text


def test_civitai_permission_sentences():
    from coreai_diffusion_converter.civitai import CivitaiPermissions

    strict = CivitaiPermissions(False, (), False, False, creator="someone")
    assert strict.summary_lines() == [
        "Credit the creator (someone) when you share the model or its outputs.",
        "No commercial use is allowed.",
        "Sharing merges or other derivatives is not allowed.",
        "Derivatives must keep these same permissions."]
    open_ = CivitaiPermissions(True, ("Image", "RentCivit", "Rent", "Sell", "SellMerge", "Future"), True, True)
    assert open_.summary_lines() == ["Commercial use allowed: selling images you generate, running it on Civitai's "
                                     "generator, running it on other generation services, selling the model, "
                                     "selling merges, Future."]
    assert open_.to_json()["allow_commercial_use"][-1] == "Future"


def test_extra_notice_forces_notice_file(tmp_path):
    info = licence.resolve(licence_file=None, licence_dir=tmp_path, name_override=None, card=(None, None),
                           allow_missing=True)
    info.extra_notice = ["Civitai permissions for X v1 (https://civitai.com/models/1?modelVersionId=2), by c:\n- a\n"]
    bundle = tmp_path / "b"
    bundle.mkdir()
    licence.write_files(info, bundle)
    assert info.notice_file == "NOTICE" and "Civitai permissions for X v1" in (bundle / "NOTICE").read_text()


def test_default_licence_names_from_civitai_base_models():
    from coreai_diffusion_converter.civitai import DEFAULT_LICENCE_NAMES

    assert DEFAULT_LICENCE_NAMES["SD 1.5"] == "CreativeML OpenRAIL-M"
    assert DEFAULT_LICENCE_NAMES["SDXL 1.0"] == "CreativeML Open RAIL++-M"
    assert DEFAULT_LICENCE_NAMES["SD 3.5 Medium"] == licence.STABILITY_NAME
    for ambiguous in ("Illustrious", "NoobAI", "Pony"):
        assert ambiguous not in DEFAULT_LICENCE_NAMES
    assert licence.licence_display_name(None, (None, None)) == "See LICENSE"


def test_changes_lists_loras_notes_and_vae_precision(tmp_path):
    info = licence.ChangesInfo(
        exporter_commit="7359dbcf6c3b", components=["text_encoder", "text_encoder_2", "unet", "vae_decoder"],
        compression="4bit", compute_precision="float16", size=1024, vae=None, clip_skip=1,
        prediction_type="epsilon", prediction_type_overridden=False, family="sdxl", vae_precision="float32",
        loras=({"name": "PerfectEyesXL.safetensors", "sha256": "ab" * 32, "scale": 0.8,
                "source": {"kind": "civitai", "ref": "118427@128461", "revision": None},
                "trained_words": ["green eyes", "perfecteyes"]},),
        notes=(licence.derivatives_warning("Pixel Art XL v1.1"),))
    text = licence.changes_text(info)
    assert "- LoRA merged: PerfectEyesXL.safetensors (sha256 abababababab), scale 0.8, from civitai 118427@128461" in text
    assert "  trigger words: green eyes, perfecteyes" in text
    assert "VAE decoder: float32 weights and compute" in text
    assert "Pixel Art XL v1.1: the creator does not allow derivatives" in text
    assert licence.LORA_TERMS_LINE in text
