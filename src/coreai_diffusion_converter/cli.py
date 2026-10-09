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

from . import __version__, civitai, exporter, licence, naming, pack, privacy, sources, validate
from . import lora as loramod
from .errors import EXIT_OK, ConverterError, ExportError, UsageError, ValidationError
from .families import Tuning, make_plan

LOG = logging.getLogger("caipack")
PREDICTION_FAMILIES = ("sd1", "sd2", "sdxl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="caipack", description="Convert diffusers models into Core AI packs.")
    parser.add_argument("--version", action="version", version=f"caipack {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="convert a model into a .caipack")
    c.add_argument("source", metavar="SOURCE",
                   help="Hub id, diffusers folder, .safetensors file, civitai:<id>[@<version>] or a Civitai "
                        "model URL")
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
    c.add_argument("--vae", help="replacement VAE: .safetensors file, folder, Hub id, civitai:<id>[@<version>] "
                                 "or a Civitai URL (SD 1.x / 2.x / SDXL)")
    c.add_argument("--lora", action="append", metavar="SOURCE[:SCALE]",
                   help="merge a LoRA before export (repeatable, applied in order; scale -4..4, default 1): "
                        "a .safetensors file, a Hub file (org/repo/file.safetensors or org/repo), or "
                        "civitai:<id>[@<version>] / a Civitai URL")
    c.add_argument("--civitai-token", help="Civitai API token, or set CIVITAI_API_TOKEN; never written anywhere")
    c.add_argument("--cache-dir", help="where Civitai downloads are kept (default: CAIPACK_CACHE_DIR, else "
                                       "~/Library/Caches/coreai-diffusion-converter)")
    c.add_argument("--allow-unverified-download", action="store_true",
                   help="accept a Civitai file for which Civitai publishes no SHA-256")
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


def _source_json(kind: str, ref: str, revision: str | None, version=None, file_sha256: str | None = None) -> dict:
    out: dict = {"kind": kind, "ref": ref, "revision": revision}
    if version is not None:
        out["file_sha256"] = file_sha256 if file_sha256 is not None else (version.file.sha256 or None)
        out["base_model"] = version.base_model
        out["permissions"] = version.permissions.to_json()
        if not version.file.sha256:
            out["verified"] = False
    return out


def _header(args: argparse.Namespace, *, pack_id: str, name: str, spec, plan, source: dict,
            lic: licence.LicenceInfo, prediction_type: str | None, loras: list) -> dict:
    conversion: dict = {"clip_skip": plan.clip_skip, "vae": _vae_label(plan.vae), "prediction_type": prediction_type}
    if loras:
        conversion["loras"] = [lo.pack_json() for lo in loras]
    if spec.family == "sdxl":
        conversion["vae_precision"] = "float32"
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
        "source": source,
        "conversion": conversion,
        "license": {"name": lic.name, "file": "LICENSE" if lic.source_file else None,
                    "notice_file": "NOTICE" if (lic.stability or lic.notice_source or lic.extra_notice) else None},
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


def _civitai_refs(args: argparse.Namespace, loras: list[tuple[str, float]]) -> bool:
    refs = [args.source, args.vae or "", *(raw for raw, _ in loras)]
    return any(civitai.is_civitai_ref(r) for r in refs if r)


def _default_name(args: argparse.Namespace, probe: sources.SourceProbe) -> str:
    if args.name is not None:
        return args.name.strip()  # an explicit --name is validated as given
    if probe.civitai is not None:
        return probe.civitai.display_name[: naming.MAX_NAME] or "model"
    return naming.default_name(args.source)  # cut to 80 characters


def _licence_name(args: argparse.Namespace, probe: sources.SourceProbe) -> str | None:
    if args.license_name:
        return args.license_name
    if probe.civitai is not None:
        return civitai.DEFAULT_LICENCE_NAMES.get(probe.civitai.base_model)
    return None


def _derivative_notes(probe: sources.SourceProbe, loras: list[loramod.LoraSpec]) -> list[str]:
    """Warn and continue (user decision) for every Civitai item whose creator forbids derivatives."""
    notes = []
    items = []
    if probe.civitai is not None:
        items.append((probe.civitai.display_name, probe.civitai.permissions))
    items += [(lo.label(), lo.permissions) for lo in loras if lo.permissions is not None]
    for name, perms in items:
        if not perms.allow_derivatives:
            text = licence.derivatives_warning(name)
            LOG.warning("%s", text)
            notes.append(text)
    return notes


def _civitai_notice(probe: sources.SourceProbe, loras: list[loramod.LoraSpec], vae_version) -> list[str]:
    versions = [probe.civitai] if probe.civitai is not None else []
    versions += [lo.civitai for lo in loras if lo.civitai is not None]
    if vae_version is not None:
        versions.append(vae_version)
    return [civitai.notice_section(v) for v in versions]


def convert(args: argparse.Namespace) -> int:
    lora_args = loramod.parse_lora_args(args.lora)
    token = args.civitai_token or os.environ.get(civitai.TOKEN_ENV) or None
    client = civitai.CivitaiClient(token) if _civitai_refs(args, lora_args) else None
    try:
        return _convert(args, lora_args, client, token)
    finally:
        if client is not None:
            client.close()


def _convert(args: argparse.Namespace, lora_args: list[tuple[str, float]], client, token: str | None) -> int:
    started = time.monotonic()
    cache_dir = civitai.default_cache_dir(args.cache_dir)
    # 1. Probe the base (Civitai metadata or Hub configs; no weights).
    probe = sources.probe_source(args.source, revision=args.revision, client=client, target=args.target)
    spec = probe.spec
    # 2. Name, pack id and output path depend on the probe for a Civitai source.
    name = _default_name(args, probe)
    _early_checks(args, name)
    pack_id = naming.pack_id(name)
    output_dir = args.output_dir.expanduser().resolve()
    out_path = output_dir / naming.output_file_name(name, args.target)
    if out_path.exists() and not args.overwrite and not args.dry_run:
        raise UsageError(f"{out_path.name} exists; pass --overwrite to replace it")
    # 3. LoRA and --vae metadata, early family checks.
    loras = [sources.probe_lora(raw, scale, client=client) for raw, scale in lora_args]
    for lo in loras:
        loramod.check_family(lo, spec.family)
        if probe.civitai is not None and lo.civitai is not None and lo.civitai.base_model != probe.civitai.base_model:
            msg = (f"{lo.label()} was trained on {lo.civitai.base_model}, the checkpoint is "
                   f"{probe.civitai.base_model}; results may differ")
            LOG.warning("%s", msg)
    vae_version = sources.probe_vae(args.vae, client=client, family=spec.family)
    notes = _derivative_notes(probe, loras)
    # 4. Plan.
    tuning = Tuning(vae=args.vae, clip_skip=args.clip_skip, prediction_type=args.prediction_type)
    plan = make_plan(spec, probe.config_tree, args.target, args.size, args.precision, tuning)
    if args.steps is not None and args.steps > spec.max_steps:
        raise UsageError(f"--steps must be between 1 and {spec.max_steps} for {spec.family}")

    missing_message = licence.CIVITAI_MISSING_MESSAGE if probe.kind == "civitai" else licence.MISSING_MESSAGE
    licence_missing = args.license_file is None and not probe.has_licence and not args.allow_missing_license
    if licence_missing and not args.dry_run:
        raise UsageError(missing_message)
    lic_preview = licence.LicenceInfo(
        name=licence.licence_display_name(_licence_name(args, probe), probe.card_license),
        source_file=args.license_file or (Path("LICENSE") if probe.has_licence else None),
        notice_source=None, stability=False, extra_notice=_civitai_notice(probe, loras, vae_version))
    lic_preview.stability = licence.is_stability(lic_preview.name, probe.card_license)
    lic_preview.attribution = licence.STABILITY_ATTRIBUTION if lic_preview.stability else ""
    header = _header(args, pack_id=pack_id, name=name, spec=spec, plan=plan,
                     source=_source_json(probe.kind, probe.ref, probe.revision, probe.civitai), lic=lic_preview,
                     prediction_type=plan.prediction_type, loras=loras)
    # Rules 1-5 need assets for the FLUX.2 size check: give the expected packages.
    header["assets"] = _expected_assets(plan)
    _check_header(header)
    for w in plan.warnings:
        LOG.warning("%s", w)

    if args.dry_run:
        if licence_missing:
            plan.warnings.append(f"no licence file found: {missing_message}")
        _print_plan(probe, plan, header, out_path, loras=loras, vae_version=vae_version, notes=notes)
        return EXIT_OK

    work_dir = (args.work_dir.expanduser().resolve() if args.work_dir
                else output_dir / ".caipack-work" / pack_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        # 5. Small files first: every LoRA, then --vae, before the checkpoint.
        for lo in loras:
            sources.fetch_lora(lo, spec.family, client=client, cache_dir=cache_dir,
                               allow_unverified=args.allow_unverified_download)
        # The worker is offline: a Hub-id or Civitai VAE is fetched here. pack.json and CHANGES.md
        # keep the reference the user gave (plan.vae), never the local cache path.
        local_vae = sources.fetch_vae(plan.vae, client=client, cache_dir=cache_dir, family=spec.family,
                                      version=vae_version, allow_unverified=args.allow_unverified_download)
        # 6. The base model.
        src = sources.resolve_source(probe, base=args.base, work_dir=work_dir, pack_id=pack_id, client=client,
                                     cache_dir=cache_dir, prediction_type=args.prediction_type,
                                     allow_unverified=args.allow_unverified_download)
        if src.kind in ("single_file", "civitai"):
            # The tree now exists: re-plan against its real configs (native size, variant checks).
            plan = make_plan(spec, src.tree, args.target, args.size, args.precision, tuning)
        export_plan = dataclasses.replace(plan, vae=local_vae, loras=tuple(lo.worker_form() for lo in loras))
        lic = licence.resolve(licence_file=args.license_file, licence_dir=src.licence_dir,
                              name_override=_licence_name(args, probe), card=src.card_license,
                              allow_missing=args.allow_missing_license)
        lic.extra_notice = _civitai_notice(probe, loras, vae_version)
        bundle, result = exporter.run_export(src.tree, export_plan, pack_id=pack_id, work_dir=work_dir,
                                             licence_name=lic.name, verbose=args.verbose)
        _check_lora_result(loras, result)
        prediction_type = result.get("prediction_type") if spec.family in PREDICTION_FAMILIES else None
        local_inputs = [probe.path.parent if probe.path else None, args.base, export_plan.vae,
                        args.license_file, cache_dir, src.file_path.parent if src.file_path else None]
        local_inputs += [lo.path.parent for lo in loras if lo.path is not None]
        _finish_bundle(bundle, src=src, spec=spec, plan=plan, lic=lic, work_dir=work_dir,
                       prediction_type=prediction_type, local_inputs=local_inputs, loras=loras,
                       notes=[*src.notes, *notes], secrets=[token] if token else [])
        header = _header(args, pack_id=pack_id, name=name, spec=spec, plan=plan,
                         source=_source_json(src.kind, src.ref, src.revision, probe.civitai, src.file_sha256),
                         lic=lic, prediction_type=prediction_type, loras=loras)
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
    for lo in loras:
        if lo.trained_words:
            print(f"trigger words ({lo.name}): {', '.join(lo.trained_words)}")
    return EXIT_OK


def _check_lora_result(loras: list[loramod.LoraSpec], result: dict) -> None:
    """The pack must never claim a merge that did not happen."""
    if not loras:
        return
    merged = result.get("loras") or []
    if len(merged) != len(loras):
        raise ExportError(f"the export merged {len(merged)} of {len(loras)} LoRAs")
    for lo, m in zip(loras, merged):
        changed = m.get("changed") or {}
        if not changed or sum(int(n) for n in changed.values()) == 0:
            raise ExportError(f"{lo.label()}: the LoRA changed no layer")


def _expected_assets(plan) -> list[str]:
    names = {"text_encoder": "TextEncoder", "text_encoder_2": "TextEncoder2", "unet": "Unet",
             "transformer": "MMDiT" if plan.spec.family == "sd3" else "Transformer",
             "transformer_512": "Transformer_512", "vae_decoder": "VAEDecoder",
             "vae_decoder_half": "VAEDecoder_half"}
    return sorted(names[c] + ".aimodel" for c in plan.components) + ["metadata.json"]


def _finish_bundle(bundle: Path, *, src: sources.ResolvedSource, spec, plan, lic: licence.LicenceInfo,
                   work_dir: Path, prediction_type: str | None,
                   local_inputs: list[str | Path | None] = (), loras: list = (), notes: list[str] = (),
                   secrets: list[str] = ()) -> None:
    metadata = privacy.normalise_metadata(bundle, src.ref)
    privacy.strip_name_or_path(bundle)
    privacy.remove_ds_store(bundle)
    licence.write_files(lic, bundle)
    licence.write_changes(licence.ChangesInfo(
        exporter_commit=pack.EXPORTER_COMMIT[:12], components=plan.components, compression=plan.compression,
        compute_precision="float16", size=plan.size, vae=plan.vae, clip_skip=plan.clip_skip,
        prediction_type=prediction_type, prediction_type_overridden=plan.prediction_type is not None,
        family=spec.family, loras=tuple(lo.pack_json() for lo in loras), notes=tuple(notes),
        vae_precision="float32" if spec.family == "sdxl" else None), bundle)
    hidden = privacy.hidden_files(bundle)
    if hidden:
        LOG.warning("hidden files in the export (validation will refuse them): %s", ", ".join(hidden))
    needles: list[str | Path] = [work_dir, Path.home()]
    if src.kind != "hf":
        needles.append(src.tree)
    for p in local_inputs:  # source file, --base, --vae, --license-file, cache dir, LoRA folders
        if p and Path(p).expanduser().exists():
            q = Path(p).expanduser().resolve()
            needles.append(q if q.is_dir() else q.parent)
    hf_cache = _hf_cache_dir()
    if hf_cache:
        needles.append(hf_cache)
    privacy.scan_text(bundle, needles, secrets=list(secrets))
    privacy.scan_binary(bundle, Path.home())
    if spec.family in PREDICTION_FAMILIES and prediction_type:
        privacy.check_prediction_type(metadata, prediction_type)


def _hf_cache_dir() -> str | None:
    try:
        from huggingface_hub import constants

        return str(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        return os.environ.get("HF_HUB_CACHE")


def _print_plan(probe: sources.SourceProbe, plan, header: dict, out_path: Path, *, loras: list = (),
                vae_version=None, notes: list[str] = ()) -> None:
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
    versions = [("model", probe.civitai)] if probe.civitai is not None else []
    if vae_version is not None:
        versions.append(("vae", vae_version))
    for slot, v in versions:
        lines += _civitai_lines(slot, v)
    for i, lo in enumerate(loras, 1):
        fam = lo.family or "unknown until downloaded"
        lines.append(f"lora {i}:          {lo.label()} ({lo.source.kind} {lo.source.ref}), scale {lo.scale:g}, "
                     f"family {fam}")
        if lo.civitai is not None:
            lines += ["  " + line for line in _civitai_lines("lora", lo.civitai)]
        if lo.trained_words:
            lines.append(f"  trigger words: {', '.join(lo.trained_words)}")
    for w in [*plan.warnings, *notes]:
        lines.append(f"warning:         {w}")
    print("\n".join(lines))


def _civitai_lines(slot: str, v) -> list[str]:
    f = v.file
    sha = f.sha256[:12] if f.sha256 else "not published"
    lines = [
        f"civitai {slot}:   {v.model_name} / {v.version_name} ({v.ref}) by {v.creator or 'unknown'}",
        f"  base model:    {v.base_model} -> {v.family}",
        f"  file:          {f.name}, {f.size_bytes / 1e9:.2f} GB, {f.fp or 'unknown precision'}, sha256 {sha}",
    ]
    lines += [f"  permission:    {line}" for line in v.permissions.summary_lines()]
    return lines


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
    # httpx logs full request URLs; a Civitai download URL carries a credential in its query.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
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
