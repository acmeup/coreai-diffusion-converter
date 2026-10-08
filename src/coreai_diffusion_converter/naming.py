# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pack id slugs and output file names."""

from __future__ import annotations

import re
from pathlib import Path, PurePath

MAX_ID = 48
MAX_NAME = 80

_NON_ID = re.compile(r"[^a-z0-9]")
_DASHES = re.compile(r"-{2,}")
_NON_FILE = re.compile(r"[^A-Za-z0-9._ -]")


def pack_id(name: str) -> str:
    """Lowercase; every char outside [a-z0-9] -> '-'; collapse runs of '-'; trim '-' at both ends;
    cut to 48 chars and trim again. Empty result -> 'model'. Always matches
    ^[a-z0-9][a-z0-9-]{0,47}$."""
    slug = _NON_ID.sub("-", name.lower())
    slug = _DASHES.sub("-", slug).strip("-")
    slug = slug[:MAX_ID].strip("-")
    return slug or "model"


def output_file_name(name: str, target: str) -> str:
    """<display name with chars outside [A-Za-z0-9._ -] -> '-'>.<target>.caipack"""
    safe = _NON_FILE.sub("-", name).strip(" .") or "model"
    return f"{safe}.{target}.caipack"


def default_name(source: str) -> str:
    """The source's last path component (a folder or Hub repo name) or a file's stem, cut to 80."""
    local = Path(source).expanduser()
    p = local.resolve() if local.exists() else PurePath(source.rstrip("/"))
    base = p.stem if p.suffix == ".safetensors" else p.name
    return (base or "model")[:MAX_NAME]
