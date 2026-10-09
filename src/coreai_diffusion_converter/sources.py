# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resolve a Hub id, a diffusers folder, a single .safetensors file or a Civitai checkpoint into a
local diffusers tree; fetch --vae and --lora inputs.

The network is used only here, in the parent process. The export itself runs offline (exporter.py).
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import civitai, licence
from . import lora as loramod
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
    kind: str              # "hf" | "folder" | "single_file" | "civitai"
    ref: str               # Hub id, folder name, file name or "<model>@<version>" (never a path)
    revision: str | None   # Hub commit sha when kind == "hf"
    licence_dir: Path | None
    card_license: tuple[str | None, str | None]
    prediction_type: str | None = None   # single files: inferred while building the tree
    notes: tuple[str, ...] = ()          # extra CHANGES.md lines (e.g. a zero-terminal-SNR warning)
    file_sha256: str | None = None       # civitai: the verified checkpoint hash
    file_path: Path | None = None        # civitai: the cached checkpoint (never written to the pack)


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
    civitai: Any = None                 # civitai.CivitaiVersion for kind "civitai"


def classify(source: str) -> str:
    if civitai.is_civitai_ref(source):
        return "civitai"
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


def probe_civitai(source: str, client: Any, target: str) -> SourceProbe:
    """Metadata only: the version, its chosen file and the family from its baseModel."""
    from .families import check_variant

    if client is None:
        raise UsageError("a Civitai source needs a Civitai client")
    version = client.resolve(civitai.parse_ref(source), "model")
    family = version.family
    if family == "flux2":
        raise UnsupportedModelError("FLUX single-file checkpoints are not supported; convert a FLUX.2 "
                                    "Klein diffusers folder or Hub id instead")
    if family == "sd3" and version.base_model == "SD 3.5 Large" and target == "ios":
        raise UnsupportedModelError("SD 3.5 Large-sized transformers are not supported for --target ios")
    spec = FAMILIES[family]
    check_variant(spec, None, target)
    return SourceProbe(kind="civitai", ref=version.ref, spec=spec, config_tree=None, civitai=version)


def probe_source(source: str, *, revision: str | None = None, client: Any = None,
                 target: str = "macos") -> SourceProbe:
    kind = classify(source)
    if kind == "civitai":
        return probe_civitai(source, client, target)
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
                        "sd2": "sd2-community/stable-diffusion-2-1",
                        # ungated, OpenRAIL++; only configs and tokenizers are fetched from it
                        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0"}
ZTSNR_WARNING = ("zero-terminal-SNR model: the app's scheduler does not rescale betas; expect lower "
                 "contrast")


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
    return _load_clip_state(model, state, dtype)


def _load_clip_state(model: Any, state: dict, dtype: Any) -> Any:
    """Load converted CLIP weights, dropping ``text_model.`` when the model is flat (transformers 5
    CLIPTextModel); refuse an incomplete encoder."""
    if not hasattr(model, "text_model"):
        state = {k.removeprefix("text_model."): v for k, v in state.items()}
    own = model.state_dict()
    state = {k: v for k, v in state.items() if k in own}
    missing = [k for k in own if k not in state and not k.endswith("position_ids")]
    if missing:
        raise UnsupportedModelError(f"the checkpoint's text encoder is incomplete ({len(missing)} tensors missing)")
    model.load_state_dict(state, strict=False)
    return model.to(dtype).eval()


def load_single_file_sdxl_text_encoders(path: Path, config_source: str, dtype) -> tuple[Any, Any]:
    """(CLIPTextModel, CLIPTextModelWithProjection) of an SDXL single-file checkpoint, with the same
    transformers-5 workaround as ``load_single_file_text_encoder``."""
    from diffusers.loaders.single_file_utils import convert_ldm_clip_checkpoint, convert_open_clip_checkpoint
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    te1 = CLIPTextModel(CLIPTextConfig.from_pretrained(config_source, subfolder="text_encoder"))
    p1 = "conditioner.embedders.0.transformer."
    state1 = convert_ldm_clip_checkpoint(_read_prefixed(path, (p1,)), remove_prefix=p1)
    te2 = CLIPTextModelWithProjection(CLIPTextConfig.from_pretrained(config_source, subfolder="text_encoder_2"))
    p2 = "conditioner.embedders.1.model."
    state2 = convert_open_clip_checkpoint(te2, _read_prefixed(path, (p2,)), prefix=p2)
    return _load_clip_state(te1, state1, dtype), _load_clip_state(te2, state2, dtype)


def _checkpoint_flags(path: Path) -> tuple[bool, bool]:
    """(has a ``v_pred`` key, has a ``ztsnr`` key) -- the NoobAI v-prediction convention."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as fh:
        keys = set(fh.keys())
    return "v_pred" in keys, "ztsnr" in keys


def build_single_file_tree(probe: SourceProbe, *, base: str | None, out_dir: Path,
                           prediction_type: str | None = None, notes: list[str] | None = None,
                           path: Path | None = None) -> tuple[Path, str]:
    """Load a single-file checkpoint with diffusers and save it as a diffusers tree.

    Returns (tree, inferred prediction type)."""
    import torch
    import diffusers

    path = path or probe.path
    assert path is not None
    family = probe.spec.family
    if family == "sdxl":
        return _build_sdxl_tree(path, base=base, out_dir=out_dir, prediction_type=prediction_type,
                                notes=notes if notes is not None else [])
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
    ensure_slow_tokenizer_files(out_dir)
    return out_dir, prediction_type


def _build_sdxl_tree(path: Path, *, base: str | None, out_dir: Path, prediction_type: str | None,
                     notes: list[str]) -> tuple[Path, str]:
    import torch
    import diffusers

    if _has_text_encoders(path):
        config_source = base or DEFAULT_CONFIG_REPOS["sdxl"]
        te1, te2 = load_single_file_sdxl_text_encoders(path, config_source, torch.float16)
        pipe = diffusers.StableDiffusionXLPipeline.from_single_file(
            str(path), config=config_source, text_encoder=te1, text_encoder_2=te2, torch_dtype=torch.float16)
    else:
        if not base:
            raise UsageError("this SDXL checkpoint has no text encoders; pass --base with a matching diffusers "
                             "SDXL model (Hub id or folder) to supply them")
        unet = diffusers.UNet2DConditionModel.from_single_file(str(path), config=base, subfolder="unet",
                                                               torch_dtype=torch.float16)
        pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16)
    v_pred, ztsnr = _checkpoint_flags(path)
    if prediction_type:
        how = "set explicitly"
    elif v_pred:
        prediction_type, how = "v_prediction", "inferred from the checkpoint"
    else:
        prediction_type = getattr(pipe.scheduler.config, "prediction_type", None) or "epsilon"
        how = "inferred"
    # diffusers' single-file loader has no v_pred handling: set it before the tree is saved, so
    # scheduler_config.json (and the exported metadata.json) carry it.
    pipe.scheduler.register_to_config(prediction_type=prediction_type)
    LOG.info("prediction type: %s (%s)", prediction_type, how)
    if ztsnr:
        LOG.warning("%s", ZTSNR_WARNING)
        notes.append(ZTSNR_WARNING[0].upper() + ZTSNR_WARNING[1:])
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(str(out_dir), safe_serialization=True)
    ensure_slow_tokenizer_files(out_dir)
    return out_dir, prediction_type


def ensure_slow_tokenizer_files(tree: Path) -> list[str]:
    """Write ``vocab.json`` and ``merges.txt`` beside ``tokenizer.json`` in every ``tokenizer*/``
    folder that lacks them. transformers 5's ``save_pretrained`` writes only the fast-tokenizer
    files, while the app's BPE tokenizer reads vocab.json and merges.txt (a single-file tree built
    without them exports a pack whose tokenizer cannot load). Returns the folders completed."""
    import json

    done = []
    for tok in sorted(p for p in tree.iterdir() if p.is_dir() and p.name.startswith("tokenizer")):
        vocab, merges, fast = tok / "vocab.json", tok / "merges.txt", tok / "tokenizer.json"
        if (vocab.is_file() and merges.is_file()) or not fast.is_file():
            continue
        model = json.loads(fast.read_text(encoding="utf-8")).get("model") or {}
        if model.get("type") != "BPE" or not isinstance(model.get("vocab"), dict):
            continue  # not a CLIP/GPT-2 style BPE tokenizer (e.g. FLUX.2's Qwen tokenizer is saved elsewhere)
        lines = [m if isinstance(m, str) else " ".join(m) for m in model.get("merges") or []]
        vocab.write_text(json.dumps(model["vocab"], ensure_ascii=False), encoding="utf-8")
        merges.write_text("#version: 0.2\n" + "\n".join(lines) + "\n", encoding="utf-8")
        special, config = tok / "special_tokens_map.json", tok / "tokenizer_config.json"
        if not special.is_file() and config.is_file():
            cfg = json.loads(config.read_text(encoding="utf-8"))
            keys = ("bos_token", "eos_token", "unk_token", "pad_token")
            special.write_text(json.dumps({k: cfg[k] for k in keys if k in cfg}, indent=2), encoding="utf-8")
        done.append(tok.name)
    if done:
        LOG.info("wrote vocab.json and merges.txt for %s", ", ".join(done))
    return done


def select_vae_files(files: list[str]) -> list[str]:
    """The VAE files of a Hub repo: its ``vae/`` configs and weights, else the root ones."""
    def wanted(f: str) -> bool:
        return f.endswith((".json", ".safetensors"))

    sub = [f for f in files if f.startswith("vae/") and f.count("/") == 1 and wanted(f)]
    picked = sub or [f for f in files if "/" not in f and wanted(f)]
    if not any(f.endswith(".safetensors") for f in picked):
        raise UsageError("--vae: the repository has no .safetensors VAE weights")
    return sorted(picked)


def probe_vae(ref: str | None, *, client: Any, family: str) -> Any:
    """Civitai metadata for a ``--vae`` reference (None for other forms). The VAE's baseModel must
    be the model's family: an SD 1.x VAE has the same block layout as an SDXL one, so the layout
    check alone cannot tell them apart."""
    if not ref or not civitai.is_civitai_ref(ref):
        return None
    if client is None:
        raise UsageError("a Civitai --vae needs a Civitai client")
    version = client.resolve(civitai.parse_ref(ref), "vae")
    if version.family != family:
        raise UnsupportedModelError(f"--vae is a {loramod.FAMILY_LABELS.get(version.family, version.family)} VAE; "
                                    f"the model is {loramod.FAMILY_LABELS.get(family, family)}")
    return version


def fetch_vae(ref: str | None, *, client: Any = None, cache_dir: Path | None = None, family: str | None = None,
              version: Any = None, allow_unverified: bool = False) -> str | None:
    """A local path for ``--vae``. A Hub id or a Civitai reference is downloaded here, because the
    export worker runs offline; a local file or folder is returned unchanged."""
    if not ref or Path(ref).expanduser().exists():
        return ref
    if civitai.is_civitai_ref(ref):
        if version is None:
            version = probe_vae(ref, client=client, family=family or "")
        if cache_dir is None:
            raise UsageError("a Civitai --vae needs a cache directory")
        return str(client.download(version, cache_dir, allow_unverified=allow_unverified))
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


def resolve_source(probe: SourceProbe, *, base: str | None, work_dir: Path, pack_id: str,
                   client: Any = None, cache_dir: Path | None = None, prediction_type: str | None = None,
                   allow_unverified: bool = False) -> ResolvedSource:
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
    notes: list[str] = []
    if probe.kind == "civitai":
        version = probe.civitai
        if client is None or cache_dir is None:
            raise UsageError("a Civitai source needs a Civitai client and a cache directory")
        path = client.download(version, cache_dir, allow_unverified=allow_unverified)
        file_family = detect_single_file_family(path)
        if file_family != probe.spec.family:
            raise UnsupportedModelError(f"Civitai lists {version.base_model} but the file is {file_family}")
        _, digest = loramod.sha256_file(path)
        tree, prediction = build_single_file_tree(probe, base=base, out_dir=work_dir / "tree" / pack_id,
                                                  prediction_type=prediction_type, notes=notes, path=path)
        return ResolvedSource(tree=tree, kind="civitai", ref=probe.ref, revision=None, licence_dir=None,
                              card_license=(None, None), prediction_type=prediction, notes=tuple(notes),
                              file_sha256=digest, file_path=path)
    tree, prediction = build_single_file_tree(probe, base=base, out_dir=work_dir / "tree" / pack_id,
                                              prediction_type=prediction_type, notes=notes)
    LOG.info("single-file prediction type: %s", prediction)
    return ResolvedSource(tree=tree, kind="single_file", ref=probe.ref, revision=None,
                          licence_dir=None, card_license=(None, None), prediction_type=prediction,
                          notes=tuple(notes))


# ---------------------------------------------------------------------------
# LoRAs (parent side: metadata first, then downloads)
# ---------------------------------------------------------------------------


def probe_lora(raw: str, scale: float, *, client: Any, revision: str | None = None) -> loramod.LoraSpec:
    """Metadata for one --lora, before any weight byte is fetched (a local file is read from its
    header only)."""
    if civitai.is_civitai_ref(raw):
        if client is None:
            raise UsageError("a Civitai --lora needs a Civitai client")
        version = client.resolve(civitai.parse_ref(raw), "lora")
        return loramod.LoraSpec(raw=raw, scale=scale, source=loramod.LoraSource("civitai", version.ref, None),
                                name=version.file.name, trained_words=list(version.trained_words),
                                permissions=version.permissions, family=version.family, civitai=version,
                                sha256=version.file.sha256, verified=bool(version.file.sha256))
    p = Path(raw).expanduser()
    if p.suffix.lower() in loramod.PICKLE_SUFFIXES:
        raise UnsupportedModelError(f"--lora: {p.suffix} files are pickle and are not loaded; use .safetensors")
    if p.exists():
        loramod.check_local_file(p)
        p = p.resolve()
        loramod.classify_lora_keys(loramod.header_shapes(p))
        return loramod.LoraSpec(raw=raw, scale=scale, source=loramod.LoraSource("file", p.name, None), path=p,
                                name=p.name, family=loramod.detect_lora_family(p))
    if loramod.HUB_FILE_RE.match(raw) or loramod.HUB_ID_RE.match(raw):
        return _probe_hub_lora(raw, scale, revision)
    raise UsageError(f"--lora {raw!r} is not an existing .safetensors file, a Hub id or file, or a Civitai reference")


def _probe_hub_lora(raw: str, scale: float, revision: str | None) -> loramod.LoraSpec:
    from huggingface_hub import HfApi

    parts = raw.split("/", 2)
    repo = "/".join(parts[:2])
    try:
        info = HfApi().model_info(repo, revision=revision)
    except Exception as err:  # noqa: BLE001
        raise _hub_error(err) from err
    files = sorted(s.rfilename for s in (info.siblings or []))
    if len(parts) == 3:
        filename = parts[2]
        if filename not in files:
            raise UsageError(f"--lora: {filename} is not in {repo}")
    else:
        top = [f for f in files if "/" not in f and f.endswith(".safetensors")]
        if len(top) != 1:
            listing = ", ".join(top) or "none"
            raise UsageError(f"--lora {repo}: name one .safetensors file as {repo}/<file> (found: {listing})")
        filename = top[0]
    return loramod.LoraSpec(raw=raw, scale=scale, source=loramod.LoraSource("hf", f"{repo}/{filename}", info.sha),
                            name=Path(filename).name, hf_filename=filename)


def fetch_lora(spec: loramod.LoraSpec, family: str, *, client: Any, cache_dir: Path | None,
               allow_unverified: bool = False) -> loramod.LoraSpec:
    """Download (Civitai / Hub), hash and check one LoRA against the model's family."""
    if spec.source.kind == "civitai":
        if client is None or cache_dir is None:
            raise UsageError("a Civitai --lora needs a Civitai client and a cache directory")
        spec.path = client.download(spec.civitai, cache_dir, allow_unverified=allow_unverified)
    elif spec.source.kind == "hf":
        from huggingface_hub import hf_hub_download

        repo = "/".join(spec.source.ref.split("/", 2)[:2])
        try:
            spec.path = Path(hf_hub_download(repo, spec.hf_filename, revision=spec.source.revision))
        except Exception as err:  # noqa: BLE001
            raise _hub_error(err) from err
    assert spec.path is not None
    loramod.classify_lora_keys(loramod.header_shapes(spec.path))
    detected = loramod.detect_lora_family(spec.path)
    if detected is not None and spec.family is not None and detected != spec.family:
        raise UnsupportedModelError(f"{spec.label()}: Civitai lists it for {loramod.FAMILY_LABELS.get(spec.family)} "
                                    f"but its weights are {loramod.FAMILY_LABELS.get(detected)}")
    spec.family = detected or spec.family
    loramod.check_family(spec, family)
    _, spec.sha256 = loramod.sha256_file(spec.path)
    return spec
