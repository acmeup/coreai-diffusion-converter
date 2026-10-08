# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
import zipfile

import pytest

from coreai_diffusion_converter import validate
from conftest import PACKS, REPO

EXPECTED = json.loads((PACKS / "expected.json").read_text())


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_shared_case(case):
    data = json.loads((PACKS / "cases" / case).read_text(encoding="utf-8"))
    assert validate.validate_case(data) == EXPECTED[case]


def test_every_case_file_is_listed():
    assert sorted(p.name for p in (PACKS / "cases").glob("*.json")) == sorted(EXPECTED)


def test_every_code_reachable_without_an_archive_is_covered():
    archive_only = {"checksum_mismatch"}
    assert set(validate.CODES) - archive_only <= {c for c in EXPECTED.values() if c}


@pytest.mark.parametrize("case", sorted(c for c, v in EXPECTED.items() if v is None))
def test_valid_cases_pass_the_schema(case):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((REPO / "schema" / "pack.schema.json").read_text())
    data = json.loads((PACKS / "cases" / case).read_text(encoding="utf-8"))
    jsonschema.validate(data["pack"], schema)


def test_overflow_counts_do_not_hang():
    data = json.loads((PACKS / "cases" / "valid_sd1_512_ios.json").read_text())
    pack = data["pack"]
    pack["files"] = [{"path": f"f{i}", "size": 64 << 30, "sha256": "0" * 64} for i in range(20_000)]
    pack["files"].append({"path": "metadata.json", "size": 1, "sha256": "0" * 64})
    assert validate.validate_case({"pack": pack, "metadata": None}) == "files_invalid"


# --- archive-level rules ----------------------------------------------------------------------


def copy_pack(tmp_path):
    dst = tmp_path / "p.caipack"
    shutil.copyfile(PACKS / "zip64_tiny.caipack", dst)
    return dst


def rewrite(src, dst, mutate):
    """Rewrite an archive member by member through ``mutate(name, data) -> list[(ZipInfo, bytes)]``."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", allowZip64=True) as zout:
        for info in zin.infolist():
            for zi, data in mutate(info, zin.read(info)):
                zout.writestr(zi, data)


def _stored(name):
    zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    zi.create_system = 3
    zi.external_attr = 0o100644 << 16
    return zi


def test_tampered_byte_is_checksum_mismatch(tmp_path):
    out = tmp_path / "t.caipack"

    def flip(info, data):
        if info.filename == "LICENSE":
            data = b"X" + data[1:]
        return [(info, data)]

    rewrite(PACKS / "zip64_tiny.caipack", out, flip)
    with pytest.raises(validate.PackError) as err:
        validate.validate_pack(out)
    assert err.value.code == "checksum_mismatch"
    validate.validate_pack(out, full=False)  # sizes still match without re-hashing


@pytest.mark.parametrize("kind,code", [("extra", "entry_not_listed"), ("missing", "entry_missing"),
                                       ("deflated", "entry_compressed"), ("symlink", "entry_not_regular_file"),
                                       ("duplicate", "duplicate_entry")])
def test_archive_entry_rules(tmp_path, kind, code):
    out = tmp_path / "t.caipack"

    def mutate(info, data):
        if info.filename != "LICENSE":
            return [(info, data)]
        if kind == "extra":
            return [(info, data), (_stored("extra.bin"), b"x")]
        if kind == "missing":
            return []
        if kind == "deflated":
            zi = _stored("LICENSE")
            zi.compress_type = zipfile.ZIP_DEFLATED
            return [(zi, data)]
        if kind == "symlink":
            zi = _stored("LICENSE")
            zi.external_attr = 0o120777 << 16
            return [(zi, data)]
        return [(info, data), (_stored("LICENSE"), data)]

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rewrite(PACKS / "zip64_tiny.caipack", out, mutate)
    with pytest.raises(validate.PackError) as err:
        validate.validate_pack(out)
    assert err.value.code == code


def test_not_a_zip(tmp_path):
    p = tmp_path / "x.caipack"
    p.write_text("hello")
    with pytest.raises(validate.PackError) as err:
        validate.validate_pack(p)
    assert err.value.code == "not_a_pack"


def test_no_pack_json(tmp_path):
    p = tmp_path / "x.caipack"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("a.txt", "x")
    with pytest.raises(validate.PackError) as err:
        validate.validate_pack(p)
    assert err.value.code == "not_a_pack"


def test_header_check_runs_without_files():
    data = json.loads((PACKS / "cases" / "valid_sd1_512_ios.json").read_text())
    header = dict(data["pack"], default_steps=0)
    with pytest.raises(validate.PackError) as err:
        validate.check_header(validate.decode(header))
    assert err.value.code == "steps_invalid"
