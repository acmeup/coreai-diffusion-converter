# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Post-export clean-up and the privacy scan: no machine path may leave this computer in a pack."""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

from .errors import ExportError

LOG = logging.getLogger(__name__)
TEXT_SUFFIXES = (".json", ".txt", ".jinja", ".md")
NAME_OR_PATH_KEYS = ("name_or_path", "_name_or_path")
_OVERLAP = 4096


def to_utc(stamp: str) -> str:
    """ISO-8601 with any offset -> UTC with a trailing Z (seconds precision kept as given)."""
    dt = _dt.datetime.fromisoformat(stamp)
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def normalise_metadata(bundle: Path, source_ref: str) -> dict:
    """metadata.json: source.hf_model_id = the source ref; compilation.date in UTC."""
    path = bundle / "metadata.json"
    md = json.loads(path.read_text(encoding="utf-8"))
    md.setdefault("source", {})["hf_model_id"] = source_ref
    comp = md.get("compilation") or {}
    if isinstance(comp.get("date"), str):
        comp["date"] = to_utc(comp["date"])
    path.write_text(json.dumps(md, indent=2) + "\n", encoding="utf-8")
    return md


def _looks_absolute(value: object) -> bool:
    return isinstance(value, str) and (value.startswith(("/", "~", "\\")) or
                                       (len(value) > 2 and value[1] == ":" and value[2] in "/\\"))


def strip_name_or_path(bundle: Path) -> list[str]:
    """Remove name_or_path keys that hold an absolute path from tokenizer*/ configs."""
    changed = []
    for tok in sorted(p for p in bundle.iterdir() if p.is_dir() and p.name.startswith("tokenizer")):
        for name in ("tokenizer_config.json", "config.json"):
            f = tok / name
            if not f.is_file():
                continue
            data = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            hits = [k for k in NAME_OR_PATH_KEYS if _looks_absolute(data.get(k))]
            if hits:
                for k in hits:
                    del data[k]
                f.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                changed.append(f.relative_to(bundle).as_posix())
    return changed


def remove_ds_store(bundle: Path) -> int:
    """Delete only .DS_Store files. Never weights: every package keeps its main.mlirb."""
    n = 0
    for p in bundle.rglob(".DS_Store"):
        if p.is_file():
            p.unlink()
            n += 1
    return n


def _needles(paths: list[str | Path]) -> list[str]:
    out = []
    for p in paths:
        s = str(p).rstrip("/")
        if len(s) > 1 and s not in out:
            out.append(s)
    return out


def scan_text(bundle: Path, needles: list[str | Path]) -> None:
    """Fail (exit 4) when a text file contains any absolute machine path in ``needles``."""
    needles_s = _needles(needles)
    for p in sorted(bundle.rglob("*")):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            text = p.read_text(encoding="utf-8", errors="replace")
            for n in needles_s:
                if n in text:
                    raise ExportError(f"{p.relative_to(bundle).as_posix()} contains a local machine path")


def scan_binary(bundle: Path, home: str | Path) -> list[str]:
    """Warn about .aimodel files that contain the home directory path. Streams with overlap."""
    needle = str(home).rstrip("/").encode()
    if len(needle) < 2:
        return []
    hits = []
    for p in sorted(bundle.rglob("*")):
        if not p.is_file() or ".aimodel" not in p.relative_to(bundle).as_posix():
            continue
        tail = b""
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                buf = tail + chunk
                if needle in buf:
                    hits.append(p.relative_to(bundle).as_posix())
                    break
                tail = buf[-_OVERLAP:]
    for h in hits:
        LOG.warning("%s contains the home directory path", h)
    return hits


def check_prediction_type(metadata: dict, expected: str) -> None:
    got = (metadata.get("diffusion") or {}).get("prediction_type")
    if got != expected:
        raise ExportError(f"metadata.json prediction_type is {got!r}, expected {expected!r}")


def hidden_files(bundle: Path) -> list[str]:
    """Hidden files left after clean-up (reported; validation rule 6 refuses them)."""
    return sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
                  if any(part.startswith(".") for part in p.relative_to(bundle).parts))
