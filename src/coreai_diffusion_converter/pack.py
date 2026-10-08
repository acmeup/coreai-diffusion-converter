# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build pack.json and write the deterministic, STORED, ZIP64-capable .caipack archive."""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__
from .validate import FORMAT, PACK_JSON, SUPPORTED_FORMAT_VERSION

CONVERTER_NAME = "coreai-diffusion-converter"
EXPORTER_REPO = "apple/coreai-models"
EXPORTER_COMMIT = "7359dbcf6c3babb4fbfadfd015ffcc1cb6d87420"
EXPORTER_REF = f"{EXPORTER_REPO}@{EXPORTER_COMMIT[:12]}"

FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
UNIX_REGULAR_644 = 0o100644 << 16
CHUNK = 8 << 20
# Entries at or over this size are written with ZIP64 headers. A module attribute so a test can
# lower it and exercise the ZIP64 path with tiny files.
ZIP64_THRESHOLD = 2**31 - 1


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def bundle_files(bundle: Path) -> list[str]:
    """Every regular file under ``bundle`` as a sorted, '/'-separated relative path (pack.json
    excluded). Directories are walked into, including ``.aimodel`` packages."""
    out = []
    for p in bundle.rglob("*"):
        if p.is_symlink():
            raise ValueError(f"symlink in bundle: {p.relative_to(bundle).as_posix()}")
        if p.is_file():
            rel = p.relative_to(bundle).as_posix()
            if rel != PACK_JSON:
                out.append(rel)
    return sorted(out)


def derive_assets(bundle: Path) -> list[str]:
    """The readiness list an importer checks: every top-level ``*.aimodel`` package, metadata.json,
    every file under ``tokenizer*/`` and every top-level ``*.npy`` (FLUX.2 batch-norm stats)."""
    assets = []
    for p in bundle.iterdir():
        if p.is_dir() and p.name.endswith(".aimodel"):
            assets.append(p.name)
        elif p.is_file() and (p.name == "metadata.json" or p.suffix == ".npy"):
            assets.append(p.name)
        elif p.is_dir() and p.name.startswith("tokenizer"):
            assets += [f.relative_to(bundle).as_posix() for f in p.rglob("*") if f.is_file()]
    return sorted(assets)


def hash_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def file_records(bundle: Path) -> list[dict[str, Any]]:
    records = []
    for rel in bundle_files(bundle):
        size, digest = hash_file(bundle / rel)
        records.append({"path": rel, "size": size, "sha256": digest})
    return records


def build_pack_json(header: dict[str, Any], bundle: Path, *, created_at: str | None = None,
                    clock: Callable[[], str] = utc_now) -> dict[str, Any]:
    """``header`` holds the descriptive fields (id ... attribution). The format, converter,
    assets, files and timestamp are filled here, in the documented field order."""
    pack: dict[str, Any] = {"format": FORMAT, "format_version": SUPPORTED_FORMAT_VERSION}
    order = ["id", "name", "description", "family", "pipeline", "target", "supported_sizes",
             "default_size", "default_steps", "max_steps", "guidance_scale", "scheduler",
             "precision", "compute_precision", "lazy_model_loading", "excluded_architectures"]
    for key in order:
        pack[key] = header[key]
    pack["assets"] = derive_assets(bundle)
    pack["source"] = header["source"]
    pack["conversion"] = header.get("conversion", {})
    pack["license"] = header["license"]
    pack["attribution"] = header.get("attribution", "")
    pack["converter"] = {"name": CONVERTER_NAME, "version": __version__, "exporter": EXPORTER_REF}
    pack["created_at"] = created_at if created_at is not None else clock()
    pack["files"] = file_records(bundle)
    return pack


def encode_pack_json(pack: dict[str, Any]) -> bytes:
    return (json.dumps(pack, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _zipinfo(name: str) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name, date_time=FIXED_DATE_TIME)
    zi.create_system = 3
    zi.external_attr = UNIX_REGULAR_644
    zi.compress_type = zipfile.ZIP_STORED
    return zi


def _write_stream(zf: zipfile.ZipFile, name: str, src: io.BufferedIOBase, size: int) -> None:
    zi = _zipinfo(name)
    zi.file_size = size
    with zf.open(zi, "w", force_zip64=size >= ZIP64_THRESHOLD) as dst:
        shutil.copyfileobj(src, dst, CHUNK)


def write_pack(bundle: Path, pack: dict[str, Any], out_path: Path) -> None:
    """Write the archive: pack.json first, then every listed file in path order, each STORED and
    streamed (never loaded whole)."""
    data = encode_pack_json(pack)
    tmp = out_path.with_name(out_path.name + ".partial")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            _write_stream(zf, PACK_JSON, io.BytesIO(data), len(data))
            for rec in sorted(pack["files"], key=lambda r: r["path"]):
                with (bundle / rec["path"]).open("rb") as src:
                    _write_stream(zf, rec["path"], src, rec["size"])
        tmp.replace(out_path)
    finally:
        if tmp.exists():
            tmp.unlink()
