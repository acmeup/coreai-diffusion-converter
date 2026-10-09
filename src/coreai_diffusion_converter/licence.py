# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Licence discovery, naming, the Stability AI notice, and the CHANGES.md statement."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .errors import UsageError

LICENCE_CANDIDATES = ("license", "license.md", "license.txt", "license-model")
NOTICE_CANDIDATES = ("notice", "notice.md")

CARD_NAMES = {
    "creativeml-openrail-m": "CreativeML OpenRAIL-M",
    "openrail++": "CreativeML Open RAIL++-M",
    "apache-2.0": "Apache License 2.0",
    "mit": "MIT License",
}
STABILITY_NAME = "Stability AI Community License"
STABILITY_NOTICE = ("This Stability AI Model is licensed under the Stability AI Community License, "
                    "Copyright © Stability AI Ltd. All Rights Reserved")
STABILITY_ATTRIBUTION = "Powered by Stability AI"

MISSING_MESSAGE = ("a licence file is required; pass --license-file, or --allow-missing-license "
                   "to pack without one")
CIVITAI_MISSING_MESSAGE = (MISSING_MESSAGE + ". Civitai's API carries no licence text; pass the model's "
                           "licence with --license-file (and --license-name).")
LORA_TERMS_LINE = ("LoRA weights were fused into the model before export; each LoRA's own terms apply in "
                   "addition to the model licence (see NOTICE).")
SDXL_VAE_LINE = "VAE decoder: float32 weights and compute (the stock SDXL VAE overflows in float16)"


def derivatives_warning(name: str) -> str:
    return (f"{name}: the creator does not allow derivatives; a converted or merged pack is a derivative. "
            "Keep it for your own use.")


def discover(directory: Path | None, candidates: tuple[str, ...] = LICENCE_CANDIDATES) -> Path | None:
    """The first file in ``directory`` whose lowercased name is in ``candidates`` (in that order)."""
    if directory is None or not directory.is_dir():
        return None
    by_lower = {p.name.lower(): p for p in sorted(directory.iterdir()) if p.is_file()}
    for name in candidates:
        if name in by_lower:
            return by_lower[name]
    return None


def has_licence_name(names: list[str]) -> bool:
    """Whether a Hub file listing holds a top-level licence file."""
    return any("/" not in n and n.lower() in LICENCE_CANDIDATES for n in names)


def read_card_license(readme: Path | None) -> tuple[str | None, str | None]:
    """(license, license_name) from a model card's YAML front matter, without a YAML dependency."""
    if readme is None or not readme.is_file():
        return None, None
    text = readme.read_text(encoding="utf-8", errors="replace")
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not m:
        return None, None
    found: dict[str, str] = {}
    for line in m.group(1).splitlines():
        km = re.match(r"^(license|license_name)\s*:\s*(.+?)\s*$", line)
        if km:
            found[km.group(1)] = km.group(2).strip("'\"")
    return found.get("license"), found.get("license_name")


def licence_display_name(override: str | None, card: tuple[str | None, str | None]) -> str:
    if override:
        return override
    lic, lic_name = card
    if lic == "other" and lic_name and lic_name.lower() == "stabilityai-ai-community":
        return STABILITY_NAME
    if lic and lic.lower() in CARD_NAMES:
        return CARD_NAMES[lic.lower()]
    return lic or "See LICENSE"


def is_stability(name: str, card: tuple[str | None, str | None]) -> bool:
    lic, lic_name = card
    return "stability ai community" in name.lower() or (
        lic == "other" and (lic_name or "").lower() == "stabilityai-ai-community")


@dataclass
class LicenceInfo:
    name: str
    source_file: Path | None
    notice_source: Path | None
    stability: bool
    file: str | None = None          # "LICENSE" once written
    notice_file: str | None = None   # "NOTICE" once written
    attribution: str = ""
    warnings: list[str] = field(default_factory=list)
    extra_notice: list[str] = field(default_factory=list)  # e.g. Civitai permission summaries


def resolve(*, licence_file: Path | None, licence_dir: Path | None, name_override: str | None,
            card: tuple[str | None, str | None], allow_missing: bool) -> LicenceInfo:
    if licence_file is not None:
        if not licence_file.is_file():
            raise UsageError(f"--license-file {licence_file.name} does not exist")
        source = licence_file
        notice = discover(licence_file.parent, NOTICE_CANDIDATES)
    else:
        source = discover(licence_dir)
        notice = discover(licence_dir, NOTICE_CANDIDATES)
    if source is None and not allow_missing:
        raise UsageError(MISSING_MESSAGE)
    name = licence_display_name(name_override, card)
    stability = is_stability(name, card)
    return LicenceInfo(name=name, source_file=source, notice_source=notice, stability=stability,
                       attribution=STABILITY_ATTRIBUTION if stability else "")


def write_files(info: LicenceInfo, bundle: Path) -> LicenceInfo:
    """Copy the licence as LICENSE and write NOTICE (always for the Stability licence)."""
    if info.source_file is not None:
        shutil.copyfile(info.source_file, bundle / "LICENSE")
        info.file = "LICENSE"
    notice_parts: list[str] = []
    if info.stability:
        notice_parts.append(f"{STABILITY_NOTICE}\n\n{STABILITY_ATTRIBUTION}\n")
    if info.notice_source is not None:
        notice_parts.append(info.notice_source.read_text(encoding="utf-8", errors="replace"))
    notice_parts += info.extra_notice
    if notice_parts:
        (bundle / "NOTICE").write_text("\n".join(notice_parts), encoding="utf-8")
        info.notice_file = "NOTICE"
    return info


@dataclass(frozen=True)
class ChangesInfo:
    exporter_commit: str
    components: list[str]
    compression: str
    compute_precision: str
    size: int
    vae: str | None
    clip_skip: int
    prediction_type: str | None
    prediction_type_overridden: bool
    family: str
    loras: tuple[dict, ...] = ()       # pack.json conversion.loras entries
    notes: tuple[str, ...] = ()        # extra lines (warnings the user should keep with the pack)
    vae_precision: str | None = None   # "float32" for SDXL


def changes_text(info: ChangesInfo) -> str:
    lines = [
        "# Changes",
        "",
        f"Converted to Apple Core AI format with coreai-diffusion-converter {__version__} "
        f"(apple/coreai-models {info.exporter_commit}).",
        "",
        f"- Components: {', '.join(info.components)}",
        f"- Weight compression: {info.compression}",
        f"- Compute precision: {info.compute_precision}",
        f"- Traced image size: {info.size} x {info.size}",
    ]
    if info.vae:
        lines.append(f"- VAE replaced with: {info.vae}")
    if info.clip_skip > 1:
        lines.append(f"- Clip skip {info.clip_skip}: the text encoder's last {info.clip_skip - 1} "
                     "layer(s) were removed")
    if info.prediction_type:
        how = "set explicitly" if info.prediction_type_overridden else "as configured"
        lines.append(f"- Prediction type: {info.prediction_type} ({how})")
    if info.vae_precision == "float32":
        lines.append(f"- {SDXL_VAE_LINE}")
    for lo in info.loras:
        src = lo.get("source") or {}
        lines.append(f"- LoRA merged: {lo.get('name')} (sha256 {str(lo.get('sha256', ''))[:12]}), "
                     f"scale {lo.get('scale'):g}, from {src.get('kind')} {src.get('ref')}")
        words = [w for w in lo.get("trained_words") or [] if w]
        if words:
            lines.append(f"  trigger words: {', '.join(words)}")
    for note in info.notes:
        lines.append(f"- {note}")
    if info.loras:
        lines += ["", LORA_TERMS_LINE]
    dropped = ["safety checker", "feature extractor", "VAE encoder"]
    if info.family == "sd3":
        dropped.append("T5 text encoder (text_encoder_3)")
    lines += ["", f"Not included: {', '.join(dropped)}.", ""]
    return "\n".join(lines)


def _strip_path(value: str) -> str:
    """Never record a filesystem path in CHANGES.md: keep only the last component."""
    return Path(value).name if ("/" in value and Path(value).exists()) or value.startswith(("/", "~")) else value


def write_changes(info: ChangesInfo, bundle: Path) -> None:
    safe = ChangesInfo(**{**info.__dict__, "vae": _strip_path(info.vae) if info.vae else None})
    (bundle / "CHANGES.md").write_text(changes_text(safe), encoding="utf-8")
