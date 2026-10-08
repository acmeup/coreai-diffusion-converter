# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a Hub id, a diffusers folder or a single .safetensors file into a local diffusers tree.

The network is used only here, in the parent process. The export itself runs offline (exporter.py).
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import licence
from .errors import UnsupportedModelError, UsageError
from .families import WEIGHT_COMPONENTS, FamilySpec, FAMILIES, detect_family, detect_single_file_family

LOG = logging.getLogger(__name__)

HUB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
PICKLE_SUFFIXES = (".ckpt", ".pt", ".pth", ".bin")
LOGIN_MESSAGE = ("access denied: accept the licence on the model page and run `hf auth login`")
NO_SAFETENSORS_MESSAGE = ("this repository has no .safetensors weights; pickle (.bin) weights are "
                          "not loaded")

# Top-level and per-directory download rules (Hub ids).
_ALWAYS_SKIP_DIRS = ("safety_checker", "feature_extractor", "text_encoder_3", "tokenizer_3")
_SKIP_SUFFIXES = (".bin", ".ckpt", ".pt", ".pth", ".onnx", ".onnx_data", ".msgpack", ".h5", ".pb",
                  ".ot", ".png", ".jpg", ".jpeg", ".webp", ".gif")
_CONFIG_DIRS = ("scheduler", "tokenizer", "tokenizer_2")
_TOP_LEVEL = ("model_index.json", "README.md")
_TOP_LEVEL_GLOBS = ("LICENSE*", "license*", "NOTICE*", "notice*")


@dataclass(frozen=True)
class ResolvedSource:
    tree: Path             # local diffusers tree the exporter reads
    kind: str              # "hf" | "folder" | "single_file"
    ref: str               # Hub id, folder name or file name (never an absolute path)
    revision: str | None   # Hub commit sha when kind == "hf"
    licence_dir: Path | None
    card_license: tuple[str | None, str | None]


@dataclass
class SourceProbe:
    """What can be learned before downloading weights or building a tree."""

    kind: str
    ref: str
    spec: FamilySpec
    config_tree: Path | None            # a tree with model_index.json and configs (None: single file)
    revision: str | None = None
    card_license: tuple[str | None, str | None] = (None, None)
    hub_files: list[str] = field(default_factory=list)
    download_files: list[str] = field(default_factory=list)
    has_licence: bool = False
    path: Path | None = None            # folder or single-file path


def classify(source: str) -> str:
    p = Path(source).expanduser()
    if p.is_dir():
        if not (p / "model_index.json").is_file():
            raise UsageError(f"{p.name} has no model_index.json; not a diffusers folder")
        return "folder"
    if p.is_file():
        if p.suffix.lower() in PICKLE_SUFFIXES:
            raise UnsupportedModelError(f"{p.suffix} checkpoints are pickle files and are not loaded; "
                                        "use a .safetensors checkpoint")
        if p.suffix.lower() != ".safetensors":
            raise UsageError("SOURCE must be a Hub id, a diffusers folder or a .safetensors file")
        return "single_file"
    if HUB_ID_RE.match(source):
        return "hf"
    raise UsageError(f"SOURCE {source!r} is not an existing folder, a .safetensors file or a Hub id")


# ---------------------------------------------------------------------------
# Hub download filter
# ---------------------------------------------------------------------------


def select_download_files(files: list[str], family: str) -> list[str]:
    """The exact Hub files the export needs. Prefers *.fp16.safetensors when every weight component
    has them. Raises UnsupportedModelError when a needed component has no .safetensors weights."""
    weight_dirs = WEIGHT_COMPONENTS[family]
    selected: list[str] = []
    for f in files:
        if "/" not in f:
            if f in _TOP_LEVEL or any(fnmatch.fnmatchcase(f, g) for g in _TOP_LEVEL_GLOBS):
                selected.append(f)
            continue
        top = f.split("/", 1)[0]
        if top in _ALWAYS_SKIP_DIRS or top.startswith("flax") or f.lower().endswith(_SKIP_SUFFIXES):
            continue
        if f.count("/") != 1:
            continue
        if top in _CONFIG_DIRS:
            selected.append(f)
        elif top in weight_dirs and (f.endswith(".json") or f.endswith(".safetensors")):
            selected.append(f)

    def weights(comp: str, fp16: bool) -> list[str]:
        return [f for f in selected if f.startswith(comp + "/") and f.endswith(".safetensors")
                and f.endswith(".fp16.safetensors") == fp16]

    for comp in weight_dirs:
        if not weights(comp, True) and not weights(comp, False):
            raise UnsupportedModelError(f"{NO_SAFETENSORS_MESSAGE} (component {comp})")
    use_fp16 = all(weights(comp, True) for comp in weight_dirs)
    out = []
    for f in selected:
        if f.endswith(".safetensors") and f.split("/", 1)[0] in weight_dirs:
            if f.endswith(".fp16.safetensors") != use_fp16:
                continue
        if ".safetensors.index" in f and f.split("/", 1)[0] in weight_dirs:
            if (".fp16." in f) != use_fp16:
                continue
        out.append(f)
    return sorted(out)


def _hub_error(err: Exception) -> Exception:
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    if isinstance(err, (GatedRepoError, RepositoryNotFoundError)):
        return UnsupportedModelError(LOGIN_MESSAGE if isinstance(err, GatedRepoError) else
                                     f"repository not found or not accessible; {LOGIN_MESSAGE}")
    if isinstance(err, HfHubHTTPError):
        status = getattr(getattr(err, "response", None), "status_code", None)
        if status in (401, 403):
            return UnsupportedModelError(LOGIN_MESSAGE)
    return err


def probe_hub(repo_id: str, revision: str | None) -> SourceProbe:
    from huggingface_hub import HfApi, snapshot_download

    try:
        info = HfApi().model_info(repo_id, revision=revision)
        files = sorted(s.rfilename for s in (info.siblings or []))
        configs = snapshot_download(repo_id, revision=info.sha, allow_patterns=[
            "model_index.json", "*/config.json", "scheduler/scheduler_config.json"])
    except Exception as err:  # noqa: BLE001 - mapped to a clear message where possible
        raise _hub_error(err) from err
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    card_license = (None, None)
    if card is not None:
        get = card.get if hasattr(card, "get") else (lambda k, d=None: getattr(card, k, d))
        card_license = (get("license", None), get("license_name", None))
    spec = detect_family(Path(configs))
    return SourceProbe(kind="hf", ref=repo_id, spec=spec, config_tree=Path(configs), revision=info.sha,
                       card_license=card_license, hub_files=files,
                       download_files=select_download_files(files, spec.family),
                       has_licence=licence.has_licence_name(files))


def probe_source(source: str, *, revision: str | None = None) -> SourceProbe:
    kind = classify(source)
    if kind == "hf":
        return probe_hub(source, revision)
    path = Path(source).expanduser().resolve()
    if kind == "folder":
        spec = detect_family(path)
        return SourceProbe(kind="folder", ref=path.name, spec=spec, config_tree=path,
                           card_license=licence.read_card_license(path / "README.md"),
                           has_licence=licence.discover(path) is not None, path=path)
    family = detect_single_file_family(path)
    LOG.info("single-file checkpoint: %s", family)
    return SourceProbe(kind="single_file", ref=path.name, spec=FAMILIES[family], config_tree=None, path=path)


# ---------------------------------------------------------------------------
# Materialise
# ---------------------------------------------------------------------------


def _has_text_encoders(path: Path) -> bool:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as fh:
        return any(k.startswith(("text_encoders.", "cond_stage_model.", "conditioner.")) for k in fh.keys())


# Diffusers configs used when a single-file checkpoint is loaded without --base. SD 2.x points at the
# community re-host because the original repository is no longer on the Hub.
DEFAULT_CONFIG_REPOS = {"sd1": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                        "sd2": "sd2-community/stable-diffusion-2-1"}


def _read_prefixed(path: Path, prefixes: tuple[str, ...]) -> dict:
    from safetensors import safe_open

    out = {}
    with safe_open(str(path), framework="pt") as fh:
        for k in fh.keys():
            if k.startswith(prefixes):
                out[k] = fh.get_tensor(k)
    return out


def load_single_file_text_encoder(path: Path, family: str, config_source: str, dtype) -> object:
    """The CLIP text encoder of an SD 1.x / 2.x single-file checkpoint.

    Diffusers' own single-file CLIP loader addresses ``model.text_model``, which transformers 5
    removed (the CLIP text model is now flat). This loader reuses diffusers' key converters and
    drops the ``text_model.`` prefix when the installed transformers has no such submodule."""
    from diffusers.loaders.single_file_utils import convert_ldm_clip_checkpoint, convert_open_clip_checkpoint
    from transformers import CLIPTextConfig, CLIPTextModel

    config = CLIPTextConfig.from_pretrained(config_source, subfolder="text_encoder")
    model = CLIPTextModel(config)
    if family == "sd1":
        checkpoint = _read_prefixed(path, ("cond_stage_model.transformer.",))
        state = convert_ldm_clip_checkpoint(checkpoint)
    else:
        checkpoint = _read_prefixed(path, ("cond_stage_model.model.",))
        state = convert_open_clip_checkpoint(model, checkpoint, prefix="cond_stage_model.model.")
    if not hasattr(model, "text_model"):
        state = {k.removeprefix("text_model."): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if k in model.state_dict()}
    missing = [k for k in model.state_dict() if k not in state and not k.endswith("position_ids")]
    if missing:
        raise UnsupportedModelError(f"the checkpoint's text encoder is incomplete ({len(missing)} tensors missing)")
    model.load_state_dict(state, strict=False)
    return model.to(dtype).eval()


def build_single_file_tree(probe: SourceProbe, *, base: str | None, out_dir: Path) -> tuple[Path, str]:
    """Load a single-file checkpoint with diffusers and save it as a diffusers tree.

    Returns (tree, inferred prediction type)."""
    import torch
    import diffusers

    path = probe.path
    assert path is not None
    family = probe.spec.family
    kwargs: dict = {"torch_dtype": torch.float16}
    if base:
        kwargs["config"] = base
    if family in ("sd1", "sd2"):
        config_source = base or DEFAULT_CONFIG_REPOS[family]
        kwargs["config"] = config_source
        text_encoder = load_single_file_text_encoder(path, family, config_source, torch.float16)
        pipe = diffusers.StableDiffusionPipeline.from_single_file(
            str(path), safety_checker=None, feature_extractor=None, text_encoder=text_encoder, **kwargs)
    else:
        if not _has_text_encoders(path):
            if not base:
                raise UsageError("this SD 3.x checkpoint has no text encoders; pass --base with the "
                                 "matching diffusers model (Hub id or folder) to supply them")
            transformer = diffusers.SD3Transformer2DModel.from_single_file(
                str(path), config=base, subfolder="transformer", torch_dtype=torch.float16)
            pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
                base, transformer=transformer, text_encoder_3=None, tokenizer_3=None,
                torch_dtype=torch.float16)
        else:
            pipe = diffusers.StableDiffusion3Pipeline.from_single_file(
                str(path), text_encoder_3=None, tokenizer_3=None, **kwargs)
    prediction_type = getattr(pipe.scheduler.config, "prediction_type", None) or "epsilon"
    if family in ("sd1", "sd2"):
        LOG.info("prediction type: %s (inferred)", prediction_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(str(out_dir), safe_serialization=True)
    return out_dir, prediction_type


def select_vae_files(files: list[str]) -> list[str]:
    """The VAE files of a Hub repo: its ``vae/`` configs and weights, else the root ones."""
    def wanted(f: str) -> bool:
        return f.endswith((".json", ".safetensors"))

    sub = [f for f in files if f.startswith("vae/") and f.count("/") == 1 and wanted(f)]
    picked = sub or [f for f in files if "/" not in f and wanted(f)]
    if not any(f.endswith(".safetensors") for f in picked):
        raise UsageError("--vae: the repository has no .safetensors VAE weights")
    return sorted(picked)


def fetch_vae(ref: str | None) -> str | None:
    """A local path for ``--vae``. A Hub id is downloaded here, because the export worker runs
    offline; a local file or folder is returned unchanged."""
    if not ref or Path(ref).expanduser().exists():
        return ref
    if not HUB_ID_RE.match(ref):
        raise UsageError(f"--vae {ref!r} is not an existing file or folder, or a Hub id")
    from huggingface_hub import HfApi, snapshot_download

    try:
        files = HfApi().list_repo_files(ref)
        return snapshot_download(ref, allow_patterns=select_vae_files(files))
    except UsageError:
        raise
    except Exception as err:  # noqa: BLE001
        raise _hub_error(err) from err


def resolve_source(probe: SourceProbe, *, base: str | None, work_dir: Path,
                   pack_id: str) -> ResolvedSource:
    if probe.kind == "folder":
        assert probe.path is not None
        return ResolvedSource(tree=probe.path, kind="folder", ref=probe.ref, revision=None,
                              licence_dir=probe.path, card_license=probe.card_license)
    if probe.kind == "hf":
        from huggingface_hub import snapshot_download

        try:
            tree = Path(snapshot_download(probe.ref, revision=probe.revision,
                                          allow_patterns=probe.download_files))
        except Exception as err:  # noqa: BLE001
            raise _hub_error(err) from err
        return ResolvedSource(tree=tree, kind="hf", ref=probe.ref, revision=probe.revision,
                              licence_dir=tree, card_license=probe.card_license)
    tree, _ = build_single_file_tree(probe, base=base, out_dir=work_dir / "tree" / pack_id)
    return ResolvedSource(tree=tree, kind="single_file", ref=probe.ref, revision=None,
                          licence_dir=None, card_license=(None, None))
