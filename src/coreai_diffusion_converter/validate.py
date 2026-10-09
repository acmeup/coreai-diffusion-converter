# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""The .caipack validation rules, with stable error codes.

These rules are the contract between the converter and every app that imports a pack. Any
importer must apply the same rules and report the same codes; the shared fixtures under
``tests/fixtures/packs/`` pin the behaviour. ``docs/FORMAT.md`` describes them in prose.

Order: the decode step, then rules 1-8, then rule 11 (only when ``metadata.json`` is supplied),
then rule 9 (only when an archive entry list is supplied), then rule 10 (full validation only).
The first failing rule is reported.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT = "caipack"
SUPPORTED_FORMAT_VERSION = 1
PACK_JSON = "pack.json"

MAX_PACK_JSON_BYTES = 4 << 20
MAX_FILE_BYTES = 64 << 30
MAX_TOTAL_BYTES = 64 << 30
MAX_FILES = 20_000
MAX_NAME_CHARS = 80
MAX_DESCRIPTION_CHARS = 500
MAX_STEPS = 100
MAX_GUIDANCE = 30.0

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FAMILY_PIPELINE = {"sd1": "stable_diffusion", "sd2": "stable_diffusion", "sdxl": "sdxl", "sd3": "sd3",
                   "flux2": "flux2"}
FAMILY_SIZES = {"sd1": (512,), "sd2": (512, 768), "sdxl": (1024,), "sd3": (512, 1024), "flux2": (512, 1024)}
FLUX2_SIZE_ASSETS = {
    512: ("Transformer_512.aimodel", "VAEDecoder_half.aimodel"),
    1024: ("Transformer.aimodel", "VAEDecoder.aimodel"),
}
METADATA_TYPE = {"stable_diffusion": "stable-diffusion", "sdxl": "stable-diffusion-xl", "sd3": "stable-diffusion-3",
                 "flux2": "flux2"}
SCHEDULERS = ("dpmpp", "pndm", "flow_match_euler")
PRECISIONS = ("fp16", "4bit")
TARGETS = ("ios", "macos")
SD_PREDICTION_TYPES = ("epsilon", "v_prediction")

CODES = (
    "not_a_pack", "unreadable_pack_json", "format_version_invalid", "format_version_unsupported",
    "invalid_id", "invalid_name", "family_unsupported", "pipeline_mismatch", "target_invalid",
    "sizes_invalid", "steps_invalid", "guidance_invalid", "scheduler_unknown", "precision_invalid",
    "unsafe_path", "duplicate_path", "files_invalid", "metadata_missing", "asset_missing",
    "license_file_missing", "duplicate_entry", "entry_not_listed", "entry_missing",
    "entry_compressed", "entry_not_regular_file", "size_mismatch", "checksum_mismatch",
    "metadata_mismatch",
)


class PackError(Exception):
    """A validation failure with a stable ``code`` (one of ``CODES``) and a detail string."""

    def __init__(self, code: str, detail: str = "") -> None:
        assert code in CODES, code
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Entry:
    """One central-directory entry of a pack archive."""

    path: str
    size: int | None  # None when the size does not fit a signed 64-bit integer
    is_regular_file: bool
    is_compressed: bool


# ---------------------------------------------------------------------------
# Decode step: types only. Missing fields take defaults that then fail a rule; a present field of
# the wrong JSON type is unreadable (an importer's typed decoder fails the same way).
# ---------------------------------------------------------------------------

_STR, _INT, _NUM, _BOOL, _LIST_STR, _LIST_INT = "str", "int", "num", "bool", "list[str]", "list[int]"

_TOP_FIELDS: dict[str, tuple[str, Any]] = {
    "format": (_STR, ""),
    "format_version": (_INT, 0),
    "id": (_STR, ""),
    "name": (_STR, ""),
    "description": (_STR, ""),
    "family": (_STR, ""),
    "pipeline": (_STR, ""),
    "target": (_STR, ""),
    "supported_sizes": (_LIST_INT, []),
    "default_size": (_INT, 0),
    "default_steps": (_INT, 0),
    "max_steps": (_INT, 0),
    "guidance_scale": (_NUM, -1.0),
    "scheduler": (_STR, ""),
    "precision": (_STR, ""),
    "compute_precision": (_STR, ""),
    "lazy_model_loading": (_BOOL, True),
    "excluded_architectures": (_LIST_STR, []),
    "assets": (_LIST_STR, []),
    "attribution": (_STR, ""),
    "created_at": (_STR, ""),
}


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _type_ok(kind: str, v: Any) -> bool:
    if kind == _STR:
        return isinstance(v, str)
    if kind == _INT:
        return _is_int(v)
    if kind == _NUM:
        return _is_int(v) or isinstance(v, float)
    if kind == _BOOL:
        return isinstance(v, bool)
    if kind == _LIST_STR:
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    if kind == _LIST_INT:
        return isinstance(v, list) and all(_is_int(x) for x in v)
    raise AssertionError(kind)


def _field(obj: dict, key: str, kind: str, default: Any) -> Any:
    v = obj.get(key)
    if v is None:
        return default
    if not _type_ok(kind, v):
        raise PackError("unreadable_pack_json", f"{key} has the wrong type")
    return v


def decode(raw: Any) -> dict:
    """Normalise a parsed pack.json into a dict with every known field present."""
    if not isinstance(raw, dict):
        raise PackError("unreadable_pack_json", "pack.json is not an object")
    out = {k: _field(raw, k, kind, default) for k, (kind, default) in _TOP_FIELDS.items()}

    def sub(key: str) -> dict:
        v = raw.get(key)
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise PackError("unreadable_pack_json", f"{key} is not an object")
        return v

    src = sub("source")
    out["source"] = {"kind": _field(src, "kind", _STR, ""), "ref": _field(src, "ref", _STR, ""),
                     "revision": _field(src, "revision", _STR, None)}
    lic = sub("license")
    out["license"] = {"name": _field(lic, "name", _STR, ""), "file": _field(lic, "file", _STR, None),
                      "notice_file": _field(lic, "notice_file", _STR, None)}
    conv = sub("converter")
    out["converter"] = {"name": _field(conv, "name", _STR, ""),
                        "version": _field(conv, "version", _STR, ""),
                        "exporter": _field(conv, "exporter", _STR, "")}
    files = raw.get("files")
    if files is None:
        files = []
    if not isinstance(files, list):
        raise PackError("unreadable_pack_json", "files is not a list")
    decoded_files = []
    for f in files:
        if not isinstance(f, dict):
            raise PackError("unreadable_pack_json", "files[] element is not an object")
        decoded_files.append({"path": _field(f, "path", _STR, ""), "size": _field(f, "size", _INT, -1),
                              "sha256": _field(f, "sha256", _STR, "")})
    out["files"] = decoded_files
    return out


# ---------------------------------------------------------------------------
# Rules 1-5 (the header; also checked by the converter before any export)
# ---------------------------------------------------------------------------


def check_header(pack: dict) -> None:
    """Rules 1-5. ``pack`` is a decoded pack (see ``decode``)."""
    # Rule 1
    if pack["format"] != FORMAT:
        raise PackError("not_a_pack", "format is not caipack")
    if pack["format_version"] < 1:
        raise PackError("format_version_invalid", str(pack["format_version"]))
    if pack["format_version"] > SUPPORTED_FORMAT_VERSION:
        raise PackError("format_version_unsupported", str(pack["format_version"]))
    # Rule 2
    if not ID_RE.match(pack["id"]):
        raise PackError("invalid_id", pack["id"])
    if not 1 <= len(pack["name"]) <= MAX_NAME_CHARS:
        raise PackError("invalid_name", "name must be 1-80 characters")
    if len(pack["description"]) > MAX_DESCRIPTION_CHARS:
        # No separate code: a description overrun is reported as invalid_name.
        raise PackError("invalid_name", "description must be at most 500 characters")
    # Rule 3
    family = pack["family"]
    if family not in FAMILY_PIPELINE:
        raise PackError("family_unsupported", family)
    if pack["pipeline"] != FAMILY_PIPELINE[family]:
        raise PackError("pipeline_mismatch", f"{family} needs {FAMILY_PIPELINE[family]}")
    if pack["target"] not in TARGETS:
        raise PackError("target_invalid", pack["target"])
    # Rule 4
    size = pack["default_size"]
    if pack["supported_sizes"] != [size] or size not in FAMILY_SIZES[family]:
        raise PackError("sizes_invalid", f"{family} supports one traced size from {FAMILY_SIZES[family]}")
    if family == "flux2":
        needed = FLUX2_SIZE_ASSETS[size]
        if not all(a in pack["assets"] for a in needed):
            raise PackError("sizes_invalid", f"flux2 {size} needs assets {needed}")
    # Rule 5
    if not 1 <= pack["default_steps"] <= pack["max_steps"] <= MAX_STEPS:
        raise PackError("steps_invalid", "need 1 <= default_steps <= max_steps <= 100")
    if not 0 <= pack["guidance_scale"] <= MAX_GUIDANCE:
        raise PackError("guidance_invalid", str(pack["guidance_scale"]))
    if pack["scheduler"] not in SCHEDULERS:
        raise PackError("scheduler_unknown", pack["scheduler"])
    if pack["precision"] not in PRECISIONS:
        raise PackError("precision_invalid", pack["precision"])


def is_safe_path(path: str) -> bool:
    """Rule 6 for one path: relative, '/'-separated, ASCII [A-Za-z0-9._-] components, no
    component starting with '.', no empty component."""
    if not path:
        return False
    for comp in path.split("/"):
        if not comp or comp.startswith(".") or not COMPONENT_RE.match(comp):
            return False
    return True


def check_contents(pack: dict, metadata: Any = None, *, check_metadata: bool = False) -> None:
    """Rules 1-8, then rule 11 when ``check_metadata``."""
    check_header(pack)
    files = pack["files"]
    # Rule 6
    seen: set[str] = set()
    for f in files:
        p = f["path"]
        if not is_safe_path(p) or p.lower() == PACK_JSON:
            raise PackError("unsafe_path", p)
        folded = p.casefold()
        if folded in seen:
            raise PackError("duplicate_path", p)
        seen.add(folded)
    for a in pack["assets"]:
        if not is_safe_path(a) or a.lower() == PACK_JSON:
            raise PackError("unsafe_path", a)
    # Rule 7
    if len(files) > MAX_FILES:
        raise PackError("files_invalid", f"more than {MAX_FILES} files")
    for f in files:
        if not 0 <= f["size"] <= MAX_FILE_BYTES:
            raise PackError("files_invalid", f"size of {f['path']}")
        if not SHA256_RE.match(f["sha256"]):
            raise PackError("files_invalid", f"sha256 of {f['path']}")
    total = sum(f["size"] for f in files)  # every element is <= 64 GiB here
    if total > MAX_TOTAL_BYTES:
        raise PackError("files_invalid", "total size over 64 GiB")
    # Rule 8
    listed = {f["path"] for f in files}
    if "metadata.json" not in listed:
        raise PackError("metadata_missing")
    for a in pack["assets"]:
        if a not in listed and not any(p.startswith(a + "/") for p in listed):
            raise PackError("asset_missing", a)
    lic = pack["license"]
    if lic["file"] is not None and (lic["file"] != "LICENSE" or "LICENSE" not in listed):
        raise PackError("license_file_missing", str(lic["file"]))
    if lic["notice_file"] is not None and (lic["notice_file"] != "NOTICE" or "NOTICE" not in listed):
        raise PackError("license_file_missing", str(lic["notice_file"]))
    if check_metadata:
        check_metadata_json(pack, metadata)


def check_metadata_json(pack: dict, metadata: Any) -> None:
    """Rule 11."""
    diffusion = metadata.get("diffusion") if isinstance(metadata, dict) else None
    if not isinstance(diffusion, dict):
        raise PackError("metadata_mismatch", "metadata.json has no diffusion object")
    expected_type = METADATA_TYPE[pack["pipeline"]]
    if diffusion.get("type") != expected_type:
        raise PackError("metadata_mismatch", f"diffusion.type is not {expected_type}")
    if pack["family"] in ("sd1", "sd2", "sdxl", "sd3"):
        image_size = diffusion.get("image_size")
        if not _is_int(image_size) or image_size != pack["default_size"]:
            raise PackError("metadata_mismatch", "diffusion.image_size differs from default_size")
    if pack["family"] in ("sd1", "sd2", "sdxl"):
        if diffusion.get("prediction_type") not in SD_PREDICTION_TYPES:
            raise PackError("metadata_mismatch", "diffusion.prediction_type")


def check_entries(pack: dict, entries: list[Entry]) -> None:
    """Rule 9: the archive's entry LIST against ``files``."""
    seen_exact: set[str] = set()
    seen_folded: set[str] = set()
    for e in entries:
        if e.path in seen_exact or e.path.casefold() in seen_folded:
            raise PackError("duplicate_entry", e.path)
        seen_exact.add(e.path)
        seen_folded.add(e.path.casefold())
    expected = {f["path"]: f["size"] for f in pack["files"]}
    names = [e.path for e in entries]
    for n in names:
        if n != PACK_JSON and n not in expected:
            raise PackError("entry_not_listed", n)
    present = set(names)
    for p in sorted([*expected, PACK_JSON]):
        if p not in present:
            raise PackError("entry_missing", p)
    for e in entries:
        if e.is_compressed:
            raise PackError("entry_compressed", e.path)
    for e in entries:
        if not e.is_regular_file:
            raise PackError("entry_not_regular_file", e.path)
    for e in entries:
        if e.size is None:
            raise PackError("size_mismatch", e.path)
        if e.path != PACK_JSON and e.size != expected[e.path]:
            raise PackError("size_mismatch", e.path)


def validate_case(case: dict) -> str | None:
    """Run a shared fixture case; return None when valid, else the error code."""
    try:
        pack = decode(case.get("pack"))
        check_contents(pack, case.get("metadata"), check_metadata=case.get("metadata") is not None)
        if "entries" in case:
            entries = [Entry(e["path"], e["size"], e["is_regular_file"], e["is_compressed"])
                       for e in case["entries"]]
            check_entries(pack, entries)
    except PackError as err:
        return err.code
    return None


# ---------------------------------------------------------------------------
# Archives
# ---------------------------------------------------------------------------

_S_IFMT, _S_IFREG = 0o170000, 0o100000
_INT64_MAX = (1 << 63) - 1


def archive_entries(zf: zipfile.ZipFile) -> list[Entry]:
    out = []
    for zi in zf.infolist():
        mode = (zi.external_attr >> 16) & 0xFFFF
        if zi.is_dir():
            regular = False
        elif zi.create_system == 3 and mode:
            regular = (mode & _S_IFMT) == _S_IFREG
        else:
            regular = True
        size = zi.file_size if zi.file_size <= _INT64_MAX else None
        out.append(Entry(zi.filename, size, regular, zi.compress_type != zipfile.ZIP_STORED))
    return out


def _read_member(zf: zipfile.ZipFile, name: str, cap: int) -> bytes:
    zi = zf.getinfo(name)
    if zi.file_size > cap:
        raise PackError("files_invalid", f"{name} is larger than {cap} bytes")
    with zf.open(zi) as fh:
        data = fh.read(cap + 1)
    if len(data) > cap:
        raise PackError("files_invalid", f"{name} is larger than {cap} bytes")
    return data


def validate_pack(path: Path | str, *, full: bool = True) -> dict:
    """Validate an archive: rules 1-9 and 11, plus rule 10 (re-hash) when ``full``.

    Returns the decoded pack.json. Raises PackError.
    """
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as err:
        raise PackError("not_a_pack", str(err)) from err
    with zf:
        names = zf.namelist()
        if PACK_JSON not in names:
            raise PackError("not_a_pack", "no pack.json")
        try:
            raw = json.loads(_read_member(zf, PACK_JSON, MAX_PACK_JSON_BYTES))
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise PackError("unreadable_pack_json", str(err)) from err
        pack = decode(raw)
        check_contents(pack)
        # Rule 11 runs whenever the archive holds metadata.json; a missing member is left to
        # rule 9 (entry_missing).
        if "metadata.json" in names:
            try:
                metadata = json.loads(_read_member(zf, "metadata.json", MAX_PACK_JSON_BYTES))
            except (UnicodeDecodeError, json.JSONDecodeError):
                metadata = None
            check_metadata_json(pack, metadata)
        check_entries(pack, archive_entries(zf))
        if full:
            digests = {f["path"]: f["sha256"] for f in pack["files"]}
            for name, want in digests.items():
                h = hashlib.sha256()
                with zf.open(name) as fh:
                    for chunk in iter(lambda: fh.read(8 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != want:
                    raise PackError("checksum_mismatch", name)
    return pack

