# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run Apple's exporter in a subprocess with the Hugging Face Hub offline."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .errors import EXIT_UNSUPPORTED, EXIT_USAGE, ExportError, UnsupportedModelError, UsageError
from .families import ExportPlan

LOG = logging.getLogger(__name__)
WORKER_MODULE = "coreai_diffusion_converter._export_worker"
RESULT_FILE = "worker_result.json"


def worker_plan(plan: ExportPlan, *, tree: Path, pack_id: str, out_root: Path, licence_name: str,
                work_dir: Path) -> dict:
    return {
        "pack_id": pack_id,
        "tree": str(tree),
        "out_root": str(out_root),
        "pipeline_type": plan.spec.pipeline_type,
        "family": plan.spec.family,
        "components": plan.components,
        "compression": plan.compression,
        "multifunction": plan.multifunction,
        "sample_size": plan.sample_size,
        "variant": plan.variant,
        "tuning": {"vae": plan.vae, "clip_skip": plan.clip_skip, "prediction_type": plan.prediction_type},
        "loras": list(plan.loras),
        "licence_name": licence_name,
        "result_path": str(work_dir / RESULT_FILE),
    }


def worker_env() -> dict[str, str]:
    """The worker's environment: offline before huggingface_hub is ever imported there, and
    without the Civitai token (the worker needs no network)."""
    env = {k: v for k, v in os.environ.items() if k != "CIVITAI_API_TOKEN"}
    return {**env, "HF_HUB_OFFLINE": "1"}


def run_export(tree: Path, plan: ExportPlan, *, pack_id: str, work_dir: Path, licence_name: str,
               verbose: bool = False) -> tuple[Path, dict]:
    """Export into ``work_dir/export/<pack_id>/``. Returns (bundle dir, worker result)."""
    out_root = work_dir / "export"
    if out_root.exists():
        shutil.rmtree(out_root)  # the exporter merges a previous metadata.json's assets otherwise
    out_root.mkdir(parents=True)
    spec = worker_plan(plan, tree=tree, pack_id=pack_id, out_root=out_root, licence_name=licence_name,
                       work_dir=work_dir)
    plan_path = work_dir / "plan.json"
    plan_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    cmd = [sys.executable, "-m", WORKER_MODULE, str(plan_path)]
    if verbose:
        cmd.append("-v")
    LOG.info("exporting %s (%s) ...", pack_id, ", ".join(plan.components))
    proc = subprocess.run(cmd, env=worker_env(), check=False)
    if proc.returncode == EXIT_USAGE:
        raise UsageError("the export rejected a tuning option (see the log above)")
    if proc.returncode == EXIT_UNSUPPORTED:
        raise UnsupportedModelError("the exporter could not load this model (see the log above)")
    if proc.returncode != 0:
        raise ExportError(f"the export failed with exit code {proc.returncode}")
    bundle = out_root / pack_id
    if not (bundle / "metadata.json").is_file():
        raise ExportError("the export produced no metadata.json")
    result_path = Path(spec["result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    return bundle, result
