# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
import random
import re

from coreai_diffusion_converter.naming import default_name, output_file_name, pack_id

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


def test_simple_names():
    assert pack_id("DreamShaper_8") == "dreamshaper-8"
    assert pack_id("SD15 Single File") == "sd15-single-file"
    assert pack_id("FLUX.2 Klein 4B") == "flux-2-klein-4b"


def test_unicode_and_empty_fallback():
    assert ID_RE.match(pack_id("  --Ünïcode!! "))
    assert pack_id("!!!") == "model"
    assert pack_id("") == "model"
    assert pack_id("ééé") == "model"


def test_long_name_cut_without_trailing_dash():
    name = "a" * 47 + " b" + "c" * 31
    assert len(name) == 80
    pid = pack_id(name)
    assert len(pid) <= 48 and not pid.endswith("-") and ID_RE.match(pid)


def test_property_every_output_matches_the_id_pattern():
    rng = random.Random(1234)
    alphabet = "abcXYZ019 -_.!/üé中\t"
    for _ in range(5000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        assert ID_RE.match(pack_id(s)), s


def test_output_file_name():
    assert output_file_name("DreamShaper 8", "ios") == "DreamShaper 8.ios.caipack"
    assert output_file_name("a/b:c", "macos") == "a-b-c.macos.caipack"


def test_default_name():
    assert default_name("org/Model-X") == "Model-X"
    assert default_name("./some/dir/") == "dir"
    assert default_name("ckpt/v1-5-pruned.safetensors") == "v1-5-pruned"
    assert len(default_name("x" * 200)) == 80


def test_default_name_of_the_current_folder(tmp_path, monkeypatch):
    (tmp_path / "my-model").mkdir()
    monkeypatch.chdir(tmp_path / "my-model")
    assert default_name(".") == "my-model"
