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
