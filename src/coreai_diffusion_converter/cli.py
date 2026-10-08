# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""``caipack``: convert, validate and inspect Core AI diffusion packs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from . import __version__, exporter, licence, naming, pack, privacy, sources, validate
from .errors import EXIT_OK, ConverterError, UsageError, ValidationError
from .families import Tuning, make_plan

LOG = logging.getLogger("caipack")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="caipack", description="Convert diffusers models into Core AI packs.")
    parser.add_argument("--version", action="version", version=f"caipack {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="convert a model into a .caipack")
    c.add_argument("source", metavar="SOURCE", help="Hub id, diffusers folder or .safetensors file")
    c.add_argument("--target", required=True, choices=("ios", "macos"))
    c.add_argument("--size", type=int, choices=(512, 768, 1024))
    c.add_argument("--precision", choices=("fp16", "4bit"))
    c.add_argument("--name")
    c.add_argument("--license-file", "--licence-file", dest="license_file", type=Path)
    c.add_argument("--license-name", "--licence-name", dest="license_name")
    c.add_argument("--allow-missing-license", "--allow-missing-licence", dest="allow_missing_license",
                   action="store_true")
    c.add_argument("--steps", type=int, help="default steps")
    c.add_argument("--guidance", type=float, help="default guidance scale")
    c.add_argument("--revision", help="Hub revision")
    c.add_argument("--base", help="Hub id or folder supplying configs / text encoders for a single file")
    c.add_argument("--vae", help="replacement VAE: .safetensors file, folder or Hub id (SD 1.x / 2.x)")
    c.add_argument("--clip-skip", type=int, default=1, help="1-4; 2 = penultimate layer (SD 1.x / 2.x)")
    c.add_argument("--prediction-type", choices=("epsilon", "v_prediction"))
    c.add_argument("--output-dir", type=Path, default=Path("."))
    c.add_argument("--work-dir", type=Path)
    c.add_argument("--keep-work", action="store_true")
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("-v", "--verbose", action="store_true")

    v = sub.add_parser("validate", help="fully validate a pack (re-hashes every file)")
    v.add_argument("pack", type=Path)
    i = sub.add_parser("inspect", help="print pack.json and a summary (no hashing)")
    i.add_argument("pack", type=Path)
    return parser


def _header(args: argparse.Namespace, *, pack_id: str, name: str, spec, plan, src_kind: str, src_ref: str,
            revision: str | None, lic: licence.LicenceInfo, prediction_type: str | None) -> dict:
    return {
        "format": validate.FORMAT, "format_version": validate.SUPPORTED_FORMAT_VERSION,
        "id": pack_id, "name": name, "description": "",
        "family": spec.family, "pipeline": spec.pipeline, "target": args.target,
        "supported_sizes": [plan.size], "default_size": plan.size,
        "default_steps": args.steps if args.steps is not None else spec.default_steps,
        "max_steps": spec.max_steps,
        "guidance_scale": args.guidance if args.guidance is not None else spec.guidance,
        "scheduler": spec.scheduler, "precision": plan.precision, "compute_precision": "float16",
        "lazy_model_loading": True, "excluded_architectures": list(spec.excluded_architectures),
        "assets": [],
        "source": {"kind": src_kind, "ref": src_ref, "revision": revision},
        "conversion": {"clip_skip": plan.clip_skip, "vae": _vae_label(plan.vae),
                       "prediction_type": prediction_type},
        "license": {"name": lic.name, "file": "LICENSE" if lic.source_file else None,
                    "notice_file": "NOTICE" if (lic.stability or lic.notice_source) else None},
        "attribution": lic.attribution,
    }


def _vae_label(vae: str | None) -> str | None:
    if not vae:
        return None
    p = Path(vae).expanduser()
    return p.name if p.exists() else vae


def _check_header(header: dict) -> None:
    """Rules 1-5 on the header, before any download or export."""
    try:
        validate.check_header(validate.decode(header))
    except validate.PackError as err:
        raise UsageError(f"invalid pack header ({err.code}): {err.detail}") from err


def _early_checks(args: argparse.Namespace, name: str) -> None:
    if not 1 <= len(name) <= validate.MAX_NAME_CHARS:
        raise UsageError("--name must be 1-80 characters")
    if args.steps is not None and not 1 <= args.steps <= 100:
        raise UsageError("--steps must be between 1 and 100")
    if args.guidance is not None and not 0 <= args.guidance <= validate.MAX_GUIDANCE:
        raise UsageError("--guidance must be between 0 and 30")
    if not 1 <= args.clip_skip <= 4:
        raise UsageError("--clip-skip must be between 1 and 4")


def convert(args: argparse.Namespace) -> int:
    started = time.monotonic()
    # A default name is cut to 80 characters; an explicit --name is validated as given.
    name = args.name.strip() if args.name is not None else naming.default_name(args.source)
    _early_checks(args, name)
    pack_id = naming.pack_id(name)
    output_dir = args.output_dir.expanduser().resolve()
    out_path = output_dir / naming.output_file_name(name, args.target)
    if out_path.exists() and not args.overwrite and not args.dry_run:
        raise UsageError(f"{out_path.name} exists; pass --overwrite to replace it")

    probe = sources.probe_source(args.source, revision=args.revision)
    spec = probe.spec
    tuning = Tuning(vae=args.vae, clip_skip=args.clip_skip, prediction_type=args.prediction_type)
    plan = make_plan(spec, probe.config_tree, args.target, args.size, args.precision, tuning)
    if args.steps is not None and args.steps > spec.max_steps:
        raise UsageError(f"--steps must be between 1 and {spec.max_steps} for {spec.family}")

    licence_missing = args.license_file is None and not probe.has_licence and not args.allow_missing_license
    if licence_missing and not args.dry_run:
        raise UsageError(licence.MISSING_MESSAGE)
    lic_preview = licence.LicenceInfo(
        name=licence.licence_display_name(args.license_name, probe.card_license),
        source_file=args.license_file or (Path("LICENSE") if probe.has_licence else None),
        notice_source=None, stability=False)
    lic_preview.stability = licence.is_stability(lic_preview.name, probe.card_license)
    lic_preview.attribution = licence.STABILITY_ATTRIBUTION if lic_preview.stability else ""
    header = _header(args, pack_id=pack_id, name=name, spec=spec, plan=plan, src_kind=probe.kind,
                     src_ref=probe.ref, revision=probe.revision, lic=lic_preview,
                     prediction_type=plan.prediction_type)
    # Rules 1-5 need assets for the FLUX.2 size check: give the expected packages.
    header["assets"] = _expected_assets(plan)
    _check_header(header)
    for w in plan.warnings:
        LOG.warning("%s", w)

    if args.dry_run:
        if licence_missing:
            plan.warnings.append(f"no licence file found: {licence.MISSING_MESSAGE}")
        _print_plan(probe, plan, header, out_path)
        return EXIT_OK

    work_dir = (args.work_dir.expanduser().resolve() if args.work_dir
                else output_dir / ".caipack-work" / pack_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        src = sources.resolve_source(probe, base=args.base, work_dir=work_dir, pack_id=pack_id)
        if src.kind == "single_file":
            # The tree now exists: re-plan against its real configs (native size, variant checks).
            plan = make_plan(spec, src.tree, args.target, args.size, args.precision, tuning)
        lic = licence.resolve(licence_file=args.license_file, licence_dir=src.licence_dir,
                              name_override=args.license_name, card=src.card_license,
                              allow_missing=args.allow_missing_license)
        # The worker is offline: a Hub-id VAE is fetched here. pack.json and CHANGES.md keep the
        # reference the user gave (plan.vae), never the local cache path.
        export_plan = dataclasses.replace(plan, vae=sources.fetch_vae(plan.vae))
        bundle, result = exporter.run_export(src.tree, export_plan, pack_id=pack_id, work_dir=work_dir,
                                             licence_name=lic.name, verbose=args.verbose)
        prediction_type = result.get("prediction_type") if spec.family in ("sd1", "sd2") else None
        local_inputs = [probe.path.parent if probe.path else None, args.base, export_plan.vae,
                        args.license_file]
        _finish_bundle(bundle, src=src, spec=spec, plan=plan, lic=lic, work_dir=work_dir,
                       prediction_type=prediction_type, local_inputs=local_inputs)
        header = _header(args, pack_id=pack_id, name=name, spec=spec, plan=plan, src_kind=src.kind,
                         src_ref=src.ref, revision=src.revision, lic=lic, prediction_type=prediction_type)
        header["license"] = {"name": lic.name, "file": lic.file, "notice_file": lic.notice_file}
        built = pack.build_pack_json(header, bundle)
        output_dir.mkdir(parents=True, exist_ok=True)
        pack.write_pack(bundle, built, out_path)
        try:
            validate.validate_pack(out_path, full=True)
        except validate.PackError as err:
            out_path.unlink(missing_ok=True)
            raise ValidationError(f"the written pack failed validation ({err.code}): {err.detail}") from err
    except BaseException:
        LOG.error("conversion failed; work dir kept at %s", work_dir)
        raise
    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
        parent = work_dir.parent
        if parent.name == ".caipack-work" and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    size = out_path.stat().st_size
    print(f"wrote {out_path.name} ({size / 1e9:.2f} GB) in {time.monotonic() - started:.0f} s; validation passed")
    return EXIT_OK


def _expected_assets(plan) -> list[str]:
    names = {"text_encoder": "TextEncoder", "text_encoder_2": "TextEncoder2", "unet": "Unet",
             "transformer": "MMDiT" if plan.spec.family == "sd3" else "Transformer",
             "transformer_512": "Transformer_512", "vae_decoder": "VAEDecoder",
             "vae_decoder_half": "VAEDecoder_half"}
    return sorted(names[c] + ".aimodel" for c in plan.components) + ["metadata.json"]


def _finish_bundle(bundle: Path, *, src: sources.ResolvedSource, spec, plan, lic: licence.LicenceInfo,
                   work_dir: Path, prediction_type: str | None,
                   local_inputs: list[str | Path | None] = ()) -> None:
    metadata = privacy.normalise_metadata(bundle, src.ref)
    privacy.strip_name_or_path(bundle)
    privacy.remove_ds_store(bundle)
    licence.write_files(lic, bundle)
    licence.write_changes(licence.ChangesInfo(
        exporter_commit=pack.EXPORTER_COMMIT[:12], components=plan.components, compression=plan.compression,
        compute_precision="float16", size=plan.size, vae=plan.vae, clip_skip=plan.clip_skip,
        prediction_type=prediction_type, prediction_type_overridden=plan.prediction_type is not None,
        family=spec.family), bundle)
    hidden = privacy.hidden_files(bundle)
    if hidden:
        LOG.warning("hidden files in the export (validation will refuse them): %s", ", ".join(hidden))
    needles: list[str | Path] = [work_dir, Path.home()]
    if src.kind != "hf":
        needles.append(src.tree)
    for p in local_inputs:  # where the source file, --base, --vae and --license-file live
        if p and Path(p).expanduser().exists():
            q = Path(p).expanduser().resolve()
            needles.append(q if q.is_dir() else q.parent)
    hf_cache = _hf_cache_dir()
    if hf_cache:
        needles.append(hf_cache)
    privacy.scan_text(bundle, needles)
    privacy.scan_binary(bundle, Path.home())
    if spec.family in ("sd1", "sd2") and prediction_type:
        privacy.check_prediction_type(metadata, prediction_type)


def _hf_cache_dir() -> str | None:
    try:
        from huggingface_hub import constants

        return str(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        return os.environ.get("HF_HUB_CACHE")


def _print_plan(probe: sources.SourceProbe, plan, header: dict, out_path: Path) -> None:
    lines = [
        f"source:          {probe.kind} {probe.ref}" + (f" @ {probe.revision}" if probe.revision else ""),
        f"family:          {plan.spec.family} (pipeline {plan.spec.pipeline})",
        f"components:      {', '.join(plan.components)}",
        f"size:            {plan.size}" + (f" (sample_size {plan.sample_size})" if plan.sample_size else ""),
        f"precision:       {plan.precision} (compression {plan.compression})",
        f"steps:           default {header['default_steps']}, max {header['max_steps']}",
        f"guidance:        {header['guidance_scale']}",
        f"scheduler:       {plan.spec.scheduler}",
        f"vae:             {_vae_label(plan.vae) or 'model default'}",
        f"clip skip:       {plan.clip_skip}",
        f"prediction type: {plan.prediction_type or 'as configured'}",
        f"licence:         {header['license']['name']}",
        f"pack id:         {header['id']}",
        f"output:          {out_path.name}",
    ]
    if probe.kind == "hf":
        lines.append("download filter:")
        lines += [f"  {f}" for f in probe.download_files]
    for w in plan.warnings:
        lines.append(f"warning:         {w}")
    print("\n".join(lines))


def cmd_validate(path: Path) -> int:
    try:
        p = validate.validate_pack(path, full=True)
    except validate.PackError as err:
        print(f"invalid: {err.code}" + (f" ({err.detail})" if err.detail else ""))
        return ValidationError.exit_code
    print(f"valid: {p['id']} ({len(p['files'])} files)")
    return EXIT_OK


def cmd_inspect(path: Path) -> int:
    try:
        with zipfile.ZipFile(path) as zf:
            raw = json.loads(zf.read(validate.PACK_JSON))
    except (zipfile.BadZipFile, KeyError, OSError, json.JSONDecodeError) as err:
        print(f"not a pack: {err}")
        return ValidationError.exit_code
    if not isinstance(raw, dict):
        print("not a pack: pack.json is not an object")
        return ValidationError.exit_code
    print(json.dumps(raw, indent=2, ensure_ascii=False))
    files = raw.get("files") if isinstance(raw.get("files"), list) else []
    total = sum(f["size"] for f in files if isinstance(f, dict) and isinstance(f.get("size"), int))
    print(f"\n{raw.get('name')} [{raw.get('family')} / {raw.get('target')}], "
          f"{len(files)} files, {total / 1e9:.2f} GB unpacked")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="%(levelname)s %(message)s")
    try:
        if args.command == "convert":
            return convert(args)
        if args.command == "validate":
            return cmd_validate(args.pack)
        return cmd_inspect(args.pack)
    except ConverterError as err:
        print(f"error: {err}", file=sys.stderr)
        return err.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
