# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""The export subprocess: adapts Apple's coreai_models diffusion exporter at run time.

This is the only module that imports ``coreai_models``. It runs with HF_HUB_OFFLINE=1 already in
its environment (exporter.py), so huggingface_hub reads offline mode at import time. Each patch
is a context manager that restores the original attribute; no Apple source is copied.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .errors import ConverterError

LOG = logging.getLogger("caipack.worker")
PIPELINE_CLASSES = ("StableDiffusionPipeline", "StableDiffusion3Pipeline", "Flux2KleinPipeline")
_MISSING = object()


@contextlib.contextmanager
def patch_attr(obj: Any, name: str, value: Any) -> Iterator[None]:
    """Set ``obj.name = value`` for the duration; restore (or delete) it afterwards."""
    had_own = name in vars(obj) if isinstance(obj, type) else hasattr(obj, name)
    original = vars(obj)[name] if isinstance(obj, type) and had_own else getattr(obj, name, _MISSING)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if isinstance(obj, type) and not had_own:
            delattr(obj, name)
        elif original is _MISSING:
            delattr(obj, name)
        else:
            setattr(obj, name, original)


def make_loader(original: Callable[..., Any], *, pack_id: str, tree: str, variant: str | None,
                sample_size: int | None, family: str, tuning: dict, result: dict) -> Callable[..., Any]:
    def load(model_id: Any, *args: Any, **kwargs: Any) -> Any:
        if str(model_id) != pack_id:
            LOG.warning("from_pretrained(%s) is not the pack id; loading the source tree anyway", model_id)
        if variant and "variant" not in kwargs:
            kwargs["variant"] = variant
        if family in ("sd1", "sd2"):
            kwargs.setdefault("feature_extractor", None)
            kwargs.setdefault("safety_checker", None)
        pipe = original(tree, *args, **kwargs)
        if sample_size:
            denoiser = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
            LOG.info("sample_size %s -> %d (image edge %d)", denoiser.config.sample_size, sample_size,
                     sample_size * 8)
            denoiser.register_to_config(sample_size=sample_size)
        if family in ("sd1", "sd2"):
            from .tuning import apply_tuning

            result.update(apply_tuning(pipe, vae=tuning.get("vae"), clip_skip=int(tuning.get("clip_skip", 1)),
                                       prediction_type=tuning.get("prediction_type"), config_tree=tree))
        return pipe

    return load


@contextlib.contextmanager
def patch_from_pretrained(**kw: Any) -> Iterator[None]:
    import diffusers

    with contextlib.ExitStack() as stack:
        for name in PIPELINE_CLASSES:
            cls = getattr(diffusers, name)
            loader = make_loader(cls.from_pretrained, **kw)
            stack.enter_context(patch_attr(cls, "from_pretrained", staticmethod(functools.partial(loader))))
        yield


def wrap_metadata(original: Callable[..., Any], licence_name: str) -> Callable[..., Any]:
    """Fill the .aimodel ``license`` field; author and description stay blank."""

    def build(hf_model_id: str, component: str | None = None) -> Any:
        md_logger = logging.getLogger(original.__module__)
        level = md_logger.level
        md_logger.setLevel(logging.ERROR)  # the "no metadata registered" banner is expected here
        try:
            metadata = original(hf_model_id, component=component)
        finally:
            md_logger.setLevel(level)
        metadata.license = licence_name
        return metadata

    return build


def run(plan: dict) -> int:
    import coreai_models.diffusion.pipeline as P

    tree = plan["tree"]
    result: dict = {}
    with patch_attr(P, "get_pipeline_type", lambda _id: plan["pipeline_type"]), \
         patch_attr(P, "snapshot_download", lambda repo_id, *a, **kw: tree), \
         patch_from_pretrained(pack_id=plan["pack_id"], tree=tree, variant=plan["variant"],
                               sample_size=plan["sample_size"], family=plan["family"],
                               tuning=plan["tuning"], result=result), \
         patch_attr(P, "build_aimodel_metadata", wrap_metadata(P.build_aimodel_metadata, plan["licence_name"])):
        P.export_diffusion(P.DiffusionExportConfig(
            hf_model_id=plan["pack_id"],  # drives out_root/<pack_id> and metadata.json "name"
            output_dir=plan["out_root"], components=plan["components"],
            compute_precision="float16", compression=plan["compression"],
            overwrite=True, multifunction=plan["multifunction"]))
    Path(plan["result_path"]).write_text(json.dumps(result), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = "-v" in argv
    argv = [a for a in argv if a != "-v"]
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    plan = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    try:
        return run(plan)
    except ConverterError as err:
        LOG.error("%s", err)
        return err.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
