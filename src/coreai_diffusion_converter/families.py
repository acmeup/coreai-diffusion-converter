# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model family detection and the per-family export plan."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .errors import UnsupportedModelError, UsageError

LOG = logging.getLogger(__name__)

SDXL_MESSAGE = ("SDXL-based models are not supported: Apple's Core AI diffusion pipelines implement "
                "SD 1.x, SD 2.x, SD 3.x and FLUX.2 Klein only")
INPAINT_MESSAGE = "inpainting checkpoints (9-channel UNet) are not supported"


def generic_unsupported(what: str) -> str:
    return (f"{what} is not supported: Apple's Core AI diffusion pipelines implement "
            "SD 1.x, SD 2.x, SD 3.x and FLUX.2 Klein text-to-image only")


@dataclass(frozen=True)
class FamilySpec:
    family: str            # "sd1" | "sd2" | "sd3" | "flux2"
    pipeline: str          # app pipeline kind: "stable_diffusion" | "sd3" | "flux2"
    pipeline_type: str     # coreai_models type: "sd" | "sd3" | "flux2"
    sizes: tuple[int, ...]
    default_steps: int
    max_steps: int
    guidance: float
    scheduler: str
    default_precision: str
    excluded_architectures: tuple[str, ...]


FAMILIES: dict[str, FamilySpec] = {
    "sd1": FamilySpec("sd1", "stable_diffusion", "sd", (512,), 25, 50, 7.5, "dpmpp", "fp16", ("h13",)),
    "sd2": FamilySpec("sd2", "stable_diffusion", "sd", (512, 768), 20, 50, 7.5, "dpmpp", "fp16", ()),
    "sd3": FamilySpec("sd3", "sd3", "sd3", (512, 1024), 28, 50, 4.5, "flow_match_euler", "4bit", ()),
    "flux2": FamilySpec("flux2", "flux2", "flux2", (512, 1024), 4, 8, 1.0, "flow_match_euler", "4bit", ()),
}

# Weight-bearing components the export loads, per family (used for the download filter and the
# .safetensors check).
WEIGHT_COMPONENTS: dict[str, tuple[str, ...]] = {
    "sd1": ("text_encoder", "unet", "vae"),
    "sd2": ("text_encoder", "unet", "vae"),
    "sd3": ("text_encoder", "text_encoder_2", "transformer", "vae"),
    "flux2": ("text_encoder", "transformer", "vae"),
}

# FLUX.2 Klein 4B transformer geometry (from its public transformer/config.json). Any other
# geometry is an untested Klein variant.
KLEIN_4B_GEOMETRY = {"num_layers": 5, "num_single_layers": 20, "inner_dim": 24 * 128,
                     "joint_attention_dim": 7680}
SD3_MAX_TESTED_LAYERS = 24  # SD 3.5 Medium; Large has 38


@dataclass(frozen=True)
class Tuning:
    vae: str | None = None
    clip_skip: int = 1
    prediction_type: str | None = None

    @property
    def is_default(self) -> bool:
        return self.vae is None and self.clip_skip == 1 and self.prediction_type is None


@dataclass(frozen=True)
class ExportPlan:
    spec: FamilySpec
    target: str
    size: int
    precision: str
    components: list[str]
    multifunction: bool
    sample_size: int | None
    compression: str       # "none" | "4bit"
    variant: str | None    # "fp16" when the tree holds only *.fp16.safetensors
    vae: str | None
    clip_skip: int
    prediction_type: str | None
    warnings: list[str] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise UnsupportedModelError(f"{path.parent.name}/{path.name} is missing; not a diffusers tree") from err
    except json.JSONDecodeError as err:
        raise UnsupportedModelError(f"{path.name} is not valid JSON") from err


def detect_family(tree: Path) -> FamilySpec:
    """Exact detection from model_index.json and the denoiser config."""
    index = _read_json(tree / "model_index.json")
    cls = str(index.get("_class_name", ""))
    if cls.startswith("StableDiffusionXL"):
        raise UnsupportedModelError(SDXL_MESSAGE)
    if "Inpaint" in cls:
        raise UnsupportedModelError(INPAINT_MESSAGE)
    if cls == "StableDiffusionPipeline":
        unet = _read_json(tree / "unet" / "config.json")
        if unet.get("in_channels") == 9:
            raise UnsupportedModelError(INPAINT_MESSAGE)
        if unet.get("in_channels") != 4:
            raise UnsupportedModelError(generic_unsupported(f"a UNet with {unet.get('in_channels')} input channels"))
        dim = unet.get("cross_attention_dim")
        if dim == 768:
            return FAMILIES["sd1"]
        if dim == 1024:
            return FAMILIES["sd2"]
        raise UnsupportedModelError(generic_unsupported(f"StableDiffusionPipeline with cross_attention_dim {dim}"))
    if cls == "StableDiffusion3Pipeline":
        return FAMILIES["sd3"]
    if cls == "Flux2KleinPipeline":
        return FAMILIES["flux2"]
    raise UnsupportedModelError(generic_unsupported(cls or "a pipeline without _class_name"))


def check_variant(spec: FamilySpec, tree: Path | None, target: str) -> list[str]:
    """Refuse (ios) or warn (macos) on untested variants within a family. Returns warnings."""
    warnings: list[str] = []
    if tree is None:
        return warnings
    if spec.family == "sd3":
        cfg = _read_json(tree / "transformer" / "config.json")
        if int(cfg.get("num_layers", 0)) > SD3_MAX_TESTED_LAYERS:
            if target == "ios":
                raise UnsupportedModelError("SD 3.5 Large-sized transformers are not supported for --target ios")
            warnings.append("untested SD 3.x variant (more than 24 transformer layers)")
    if spec.family == "flux2":
        cfg = _read_json(tree / "transformer" / "config.json")
        geometry = {
            "num_layers": cfg.get("num_layers"),
            "num_single_layers": cfg.get("num_single_layers"),
            "inner_dim": int(cfg.get("num_attention_heads", 0)) * int(cfg.get("attention_head_dim", 0)),
            "joint_attention_dim": cfg.get("joint_attention_dim"),
        }
        if geometry != KLEIN_4B_GEOMETRY:
            if target == "ios":
                raise UnsupportedModelError("only the FLUX.2 Klein 4B geometry is supported for --target ios")
            warnings.append("untested FLUX.2 Klein variant (geometry differs from Klein 4B)")
    return warnings


NOMINAL_NATIVE = {"sd1": 512, "sd2": 768, "sd3": 1024, "flux2": 1024}


def native_size(spec: FamilySpec, tree: Path | None) -> int:
    """The denoiser's trained edge in pixels. Without a tree (a single-file dry run) the family's
    nominal size is assumed."""
    if spec.family == "flux2" or tree is None:
        return NOMINAL_NATIVE[spec.family]
    sub = "transformer" if spec.family == "sd3" else "unet"
    cfg = _read_json(tree / sub / "config.json")
    return int(cfg.get("sample_size", 64)) * 8


def default_size(spec: FamilySpec, tree: Path | None, target: str) -> int:
    if target == "ios":
        return 512
    if spec.family == "sd2":
        n = native_size(spec, tree)
        return n if n in spec.sizes else 512
    return {"sd1": 512, "sd3": 1024, "flux2": 1024}[spec.family]


def weight_variant(tree: Path, components: tuple[str, ...]) -> str | None:
    """'fp16' when every weight component holds only *.fp16.safetensors files, else None."""
    any_fp16 = False
    for comp in components:
        files = [p.name for p in (tree / comp).glob("*.safetensors")]
        fp16 = [f for f in files if f.endswith(".fp16.safetensors")]
        plain = [f for f in files if not f.endswith(".fp16.safetensors")]
        if plain:
            return None
        any_fp16 = any_fp16 or bool(fp16)
    return "fp16" if any_fp16 else None


def validate_tuning(spec: FamilySpec, tuning: Tuning) -> None:
    if not 1 <= tuning.clip_skip <= 4:
        raise UsageError("--clip-skip must be between 1 and 4")
    if tuning.prediction_type not in (None, "epsilon", "v_prediction"):
        raise UsageError("--prediction-type must be epsilon or v_prediction")
    if spec.family in ("sd3", "flux2") and not tuning.is_default:
        raise UsageError("--vae, --clip-skip and --prediction-type apply to SD 1.x / 2.x models only")


def make_plan(spec: FamilySpec, tree: Path | None, target: str, size: int | None, precision: str | None,
              tuning: Tuning) -> ExportPlan:
    validate_tuning(spec, tuning)
    warnings = check_variant(spec, tree, target)
    size = size or default_size(spec, tree, target)
    if size not in spec.sizes:
        raise UsageError(f"--size {size} is not available for {spec.family}; choose from {list(spec.sizes)}")
    precision = precision or spec.default_precision
    if precision not in ("fp16", "4bit"):
        raise UsageError("--precision must be fp16 or 4bit")
    if precision == "4bit" and spec.family in ("sd1", "sd2"):
        warnings.append("4-bit SD 1.x / 2.x output quality is unmeasured")

    sample_size = None
    if spec.family == "flux2":
        components = (["transformer_512", "text_encoder", "vae_decoder_half"] if size == 512
                      else ["transformer", "text_encoder", "vae_decoder"])
    else:
        components = (["text_encoder", "text_encoder_2", "transformer", "vae_decoder"] if spec.family == "sd3"
                      else ["text_encoder", "unet", "vae_decoder"])
        native = native_size(spec, tree)
        if size != native:
            sample_size = size // 8
            if spec.family == "sd2" and native == 768 and size == 512:
                warnings.append("a 768 px (often v-prediction) SD 2.x checkpoint degrades when traced at 512 px")
    return ExportPlan(
        spec=spec, target=target, size=size, precision=precision, components=components,
        multifunction=False, sample_size=sample_size,
        compression="none" if precision == "fp16" else "4bit",
        variant=weight_variant(tree, WEIGHT_COMPONENTS[spec.family]) if tree is not None else None,
        vae=tuning.vae, clip_skip=tuning.clip_skip, prediction_type=tuning.prediction_type,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Single-file checkpoints
# ---------------------------------------------------------------------------


class _ShapeView(Mapping):
    """A read-only mapping of tensor name -> object with ``.shape`` that never loads weights."""

    def __init__(self, path: Path) -> None:
        from safetensors import safe_open

        self._fh = safe_open(str(path), framework="pt")
        self._keys = set(self._fh.keys())

    def __getitem__(self, key: str) -> Any:
        if key not in self._keys:
            raise KeyError(key)
        return SimpleNamespace(shape=tuple(self._fh.get_slice(key).get_shape()))

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._keys


def single_file_model_type(path: Path) -> str:
    from diffusers.loaders.single_file_utils import infer_diffusers_model_type

    return str(infer_diffusers_model_type(_ShapeView(path)))


def family_for_single_file_type(model_type: str) -> str:
    if model_type.startswith("xl_") or model_type.startswith("playground"):
        raise UnsupportedModelError(SDXL_MESSAGE)
    if model_type.startswith("inpainting") or "inpaint" in model_type:
        raise UnsupportedModelError(INPAINT_MESSAGE)
    if model_type == "v1":
        return "sd1"
    if model_type == "v2":
        return "sd2"
    if model_type in ("sd3", "sd35_medium", "sd35_large"):
        return "sd3"
    if model_type.startswith("flux"):
        raise UnsupportedModelError("FLUX single-file checkpoints are not supported; convert a FLUX.2 "
                                    "Klein diffusers folder or Hub id instead")
    raise UnsupportedModelError(generic_unsupported(f"single-file checkpoint type '{model_type}'"))


def detect_single_file_family(path: Path) -> str:
    """Family of a single .safetensors checkpoint, via diffusers' infer_diffusers_model_type."""
    if path.suffix.lower() in (".ckpt", ".pt", ".pth", ".bin"):
        raise UnsupportedModelError(f"{path.suffix} checkpoints are pickle files and are not loaded; "
                                    "use a .safetensors checkpoint")
    from diffusers.loaders.single_file_utils import infer_diffusers_model_type

    view = _ShapeView(path)
    family = family_for_single_file_type(str(infer_diffusers_model_type(view)))
    if family in ("sd1", "sd2"):
        # "v1" is diffusers' fallback for anything it does not recognise (LoRAs, embeddings, VAEs),
        # so require a full UNet before trusting it.
        key = "model.diffusion_model.input_blocks.0.0.weight"
        if key not in view:
            raise UnsupportedModelError(generic_unsupported("a checkpoint without a full SD UNet "
                                                            "(a LoRA, embedding or VAE?)"))
        if view[key].shape[1] == 9:
            raise UnsupportedModelError(INPAINT_MESSAGE)
    return family
