# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""LoRA merging: ``--lora SOURCE[:SCALE]`` fused into the model before export.

The parent process parses each ``--lora``, resolves it (a local file, a Hugging Face file or a
Civitai version), checks its family and downloads it. The export worker then merges every LoRA,
in order, right after the pipeline is loaded and before clip skip, quantization and tracing, so
the exported graph simply contains the merged weights.

Text-encoder LoRAs and transformers 5 (measured, diffusers 0.37.1, transformers 5.12.1, peft
0.21.2):

* ``CLIPTextModel`` (SD 1.x / 2.x and SDXL text encoder 1) is flat in transformers 5: its module
  names start ``encoder.layers.N``. diffusers' text-encoder LoRA loader converts Kohya keys to
  ``text_model.encoder.layers.N`` and looks their ranks up among the model's module names, finds
  none and raises ``IndexError`` -- which aborts the whole ``load_lora_weights`` call, UNet part
  included. This class therefore uses ``_merge_text_encoder_lora`` (path "manual").
* ``CLIPTextModelWithProjection`` (SDXL text encoder 2, SD3's two CLIP encoders) still nests
  ``text_model.``; diffusers attaches its LoRA and the fused delta equals
  ``scale * alpha / rank * up @ down`` exactly. This class uses diffusers (path "diffusers").

Each encoder class has exactly one path (``TEXT_ENCODER_PATHS``); there is no runtime fallback.
Every LoRA state dict is split into denoiser keys and per-encoder text-encoder keys first, and
diffusers only ever sees the parts whose path is "diffusers".
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import UnsupportedModelError, UsageError

LOG = logging.getLogger(__name__)

MAX_LORAS = 8
SCALE_RANGE = (-4.0, 4.0)  # 0 is refused; negative scales are legitimate ("slider" LoRAs)
PICKLE_SUFFIXES = (".ckpt", ".pt", ".pth", ".bin")
HUB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
HUB_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*/.+\.safetensors$")

# Step 0 outcome, per text-encoder class (see the module docstring).
TEXT_ENCODER_PATHS = {"CLIPTextModel": "manual", "CLIPTextModelWithProjection": "diffusers"}
TEXT_ENCODERS = ("text_encoder", "text_encoder_2")

FAMILY_LABELS = {"sd1": "SD 1.x", "sd2": "SD 2.x", "sdxl": "SDXL", "sd3": "SD 3.x", "flux2": "FLUX.2 Klein"}
LYCORIS_MESSAGE = "LyCORIS / DoRA adapters are not supported; use a plain LoRA"
_LYCORIS_MARKERS = (".hada_w", ".lokr_w", ".oft_", "lora_mid", ".dora_scale", ".ia3")
KLEIN_DOUBLE_BLOCKS, KLEIN_SINGLE_BLOCKS = 5, 20


@dataclass(frozen=True)
class LoraSource:
    kind: str                 # "file" | "hf" | "civitai"
    ref: str                  # file basename, "org/repo[/file.safetensors]", or "<model>@<version>"
    revision: str | None      # Hub commit for "hf", else None

    def to_json(self) -> dict:
        return {"kind": self.kind, "ref": self.ref, "revision": self.revision}


@dataclass
class LoraSpec:
    raw: str                  # what the user typed (never written to the pack)
    scale: float
    source: LoraSource
    path: Path | None = None  # local .safetensors once resolved (never written to the pack)
    name: str = ""            # file basename
    sha256: str = ""          # lowercase hex, computed locally (verified against Civitai when known)
    trained_words: list[str] = field(default_factory=list)
    permissions: Any = None   # civitai.CivitaiPermissions | None
    family: str | None = None # detected from keys/shapes, or from Civitai baseModel before download
    civitai: Any = None       # civitai.CivitaiVersion | None (parent only)
    verified: bool = True     # False only for a Civitai file without a published SHA-256
    hf_filename: str | None = None

    def worker_form(self) -> dict:
        return {"path": str(self.path), "scale": self.scale, "name": self.name}

    def pack_json(self) -> dict:
        out = {"name": self.name, "sha256": self.sha256, "scale": self.scale,
               "source": self.source.to_json(), "trained_words": list(self.trained_words),
               "permissions": self.permissions.to_json() if self.permissions is not None else None}
        if not self.verified:
            out["verified"] = False
        return out

    def label(self) -> str:
        return self.name or self.raw


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def is_complete_source(s: str) -> bool:
    """A Civitai reference with an id, an existing local path, or a Hub id / Hub file path."""
    from . import civitai

    if civitai.is_civitai_ref(s):
        try:
            civitai.parse_ref(s)
            return True
        except UsageError:
            return False
    if Path(s).expanduser().exists():
        return True
    return bool(HUB_ID_RE.match(s) or HUB_FILE_RE.match(s))


def _parse_scale(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def check_scale(scale: float) -> float:
    if scale == 0:
        raise UsageError("--lora scale 0 merges nothing; leave the LoRA out instead")
    if not SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]:
        raise UsageError(f"--lora scale must be between {SCALE_RANGE[0]:g} and {SCALE_RANGE[1]:g}")
    return scale


def parse_lora_arg(arg: str) -> tuple[str, float]:
    """SOURCE[:SCALE]. Split at the LAST ':' only when the suffix parses as a float AND the prefix
    is itself a complete source; otherwise the whole string is the source and the scale is 1.0."""
    arg = arg.strip()
    if not arg:
        raise UsageError("--lora needs a source")
    head, sep, tail = arg.rpartition(":")
    if sep:
        scale = _parse_scale(tail)
        if scale is not None and head and is_complete_source(head):
            return head, check_scale(scale)
    return arg, 1.0


def parse_lora_args(args: Sequence[str] | None) -> list[tuple[str, float]]:
    args = list(args or [])
    if len(args) > MAX_LORAS:
        raise UsageError(f"at most {MAX_LORAS} --lora options are supported")
    return [parse_lora_arg(a) for a in args]


# ---------------------------------------------------------------------------
# Key inspection (safetensors header only)
# ---------------------------------------------------------------------------


def classify_lora_keys(keys: Iterable[str]) -> str:
    """'kohya' | 'diffusers' | 'peft'. LyCORIS / DoRA / IA3 adapters are refused."""
    keys = list(keys)
    for k in keys:
        if any(m in k for m in _LYCORIS_MARKERS):
            raise UnsupportedModelError(LYCORIS_MESSAGE)
    if any(k.startswith(("lora_unet_", "lora_te")) for k in keys):
        return "kohya"
    if any(".lora_A." in k or ".lora_B." in k for k in keys):
        return "peft"
    if any("lora" in k for k in keys):
        return "diffusers"
    raise UnsupportedModelError("the file holds no LoRA weights")


def header_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Tensor name -> shape from the safetensors header, without loading any weight."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as fh:
        return {k: tuple(fh.get_slice(k).get_shape()) for k in fh.keys()}


def _is_down(key: str) -> bool:
    return key.endswith((".lora_down.weight", ".lora_A.weight", ".lora.down.weight", ".down.weight"))


def _max_index(keys: Iterable[str], pattern: str) -> int:
    rx = re.compile(pattern)
    found = [int(m.group(1)) for k in keys for m in [rx.search(k)] if m]
    return max(found) if found else -1


def family_from_shapes(shapes: dict[str, tuple[int, ...]]) -> str | None:
    keys = list(shapes)
    # 1. FLUX (double / single stream blocks)
    if any(m in k for k in keys for m in ("single_transformer_blocks", "double_blocks", "single_blocks")):
        double = max(_max_index(keys, r"double_blocks[._](\d+)[._]"),
                     _max_index(keys, r"(?<!single_)transformer_blocks[._](\d+)[._]"))
        single = max(_max_index(keys, r"single_transformer_blocks[._](\d+)[._]"),
                     _max_index(keys, r"single_blocks[._](\d+)[._]"))
        if double >= KLEIN_DOUBLE_BLOCKS or single >= KLEIN_SINGLE_BLOCKS:
            raise UnsupportedModelError("not a FLUX.2 Klein 4B LoRA (its block count is that of a "
                                        "larger FLUX model)")
        return "flux2"
    # 2. SD3 MMDiT
    if any("joint_blocks" in k for k in keys) or any(
            "transformer_blocks" in k and ("context" in k or "add_q_proj" in k or "add_k_proj" in k)
            for k in keys):
        return "sd3"
    # 3. UNet cross-attention width
    for k, shape in shapes.items():
        if "attn2" in k and "to_k" in k and _is_down(k) and len(shape) >= 2:
            dim = shape[1]
            if dim == 768:
                return "sd1"
            if dim == 1024:
                return "sd2"
            if dim == 2048:
                return "sdxl"
    # 4. Text encoders only
    if any(k.startswith("lora_te3_") or k.startswith("text_encoder_3.") for k in keys):
        return "sd3"
    if any(k.startswith(("lora_te1_", "lora_te2_", "text_encoder_2.")) for k in keys):
        return "sdxl"
    for k, shape in shapes.items():
        if (k.startswith("lora_te_") or k.startswith("text_encoder.")) and _is_down(k) and len(shape) >= 2:
            if shape[1] == 768:
                return "sd1"
            if shape[1] == 1024:
                return "sd2"
    return None


def detect_lora_family(path: Path) -> str | None:
    """The LoRA's family from its safetensors header: denoiser keys decide, text-encoder keys are a
    tie-break only. None when it cannot be told (left to the post-fuse fingerprint check)."""
    return family_from_shapes(header_shapes(path))


def has_text_encoder_keys(keys: Iterable[str]) -> bool:
    return any(k.startswith(("lora_te", "text_encoder")) for k in keys)


def check_family(lora: LoraSpec, family: str) -> None:
    """Refuse a known family mismatch, and the LoRA shapes a family cannot take."""
    if lora.family and lora.family != family:
        raise UnsupportedModelError(f"{lora.label()} is an {FAMILY_LABELS.get(lora.family, lora.family)} "
                                    f"LoRA; the model is {FAMILY_LABELS.get(family, family)}")
    if lora.path is not None and lora.path.is_file():
        keys = list(header_shapes(lora.path))
        if family == "flux2" and has_text_encoder_keys(keys):
            raise UnsupportedModelError(f"{lora.label()}: FLUX.2 Klein LoRAs must target the transformer only")
        if family == "sd3" and any(k.startswith(("lora_te3_", "text_encoder_3.")) for k in keys):
            LOG.warning("%s: its T5 (text_encoder_3) weights are dropped; T5 is not exported", lora.label())


def check_local_file(path: Path) -> None:
    if path.suffix.lower() in PICKLE_SUFFIXES:
        raise UnsupportedModelError(f"--lora: {path.suffix} files are pickle and are not loaded; use .safetensors")
    if not path.is_file():
        raise UsageError(f"--lora {path.name} does not exist")
    if path.suffix.lower() != ".safetensors":
        raise UsageError("--lora must be a .safetensors file")


def sha256_file(path: Path) -> tuple[int, str]:
    from .pack import hash_file

    return hash_file(path)


# ---------------------------------------------------------------------------
# Worker side: merge
# ---------------------------------------------------------------------------


def _fingerprint(module: Any) -> dict[str, tuple[float, float]]:
    import torch

    out = {}
    for name, mod in module.named_modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
            w = mod.weight.detach().float()
            out[name] = (w.sum().item(), w.pow(2).sum().item())
    return out


def _changed(before: dict, after: dict) -> int:
    return sum(1 for k, v in after.items() if before.get(k) != v)


def split_state(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Denoiser keys, per-encoder text-encoder keys (prefix kept), and dropped T5 keys."""
    parts: dict[str, dict[str, Any]] = {"denoiser": {}, "text_encoder": {}, "text_encoder_2": {}, "dropped": {}}
    for k, v in state.items():
        if k.startswith("lora_te3_") or k.startswith("text_encoder_3."):
            parts["dropped"][k] = v
        elif k.startswith(("lora_te_", "lora_te1_", "text_encoder.")):
            parts["text_encoder"][k] = v
        elif k.startswith(("lora_te2_", "text_encoder_2.")):
            parts["text_encoder_2"][k] = v
        else:
            parts["denoiser"][k] = v
    return parts


_SUFFIXES = (
    (".lora_down.weight", "down"), (".lora_up.weight", "up"),
    (".lora_A.weight", "down"), (".lora_B.weight", "up"),
    (".lora.down.weight", "down"), (".lora.up.weight", "up"),
    (".lora_linear_layer.down.weight", "down"), (".lora_linear_layer.up.weight", "up"),
    (".alpha", "alpha"),
)
_TE_PREFIXES = ("lora_te1_", "lora_te2_", "lora_te_", "text_encoder_2.", "text_encoder.")


def _te_stem(key: str) -> tuple[str, str] | None:
    """(underscore stem, part) for a text-encoder LoRA key, or None for a key that is not one."""
    for suffix, part in _SUFFIXES:
        if key.endswith(suffix):
            body = key[: -len(suffix)]
            break
    else:
        return None
    for prefix in _TE_PREFIXES:
        if body.startswith(prefix):
            body = body[len(prefix):]
            break
    return body.replace(".", "_"), part


def _linear_index(text_encoder: Any) -> dict[str, Any]:
    import torch

    index: dict[str, Any] = {}
    for name, mod in text_encoder.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        flat = name.replace(".", "_")
        index[flat] = mod
        # Flat (transformers 5 CLIPTextModel) and nested layouts both match either key spelling.
        if flat.startswith("text_model_"):
            index.setdefault(flat[len("text_model_"):], mod)
        else:
            index.setdefault("text_model_" + flat, mod)
    return index


def _merge_text_encoder_lora(text_encoder: Any, state: dict[str, Any], scale: float) -> int:
    """Merge LoRA weights into a text encoder by hand: W += scale * alpha / rank * up @ down.
    Keys are matched through an index of ``named_modules()`` with '.' replaced by '_' (underscores
    are never turned back into dots: self_attn, q_proj, out_proj and fc1 contain them). Returns the
    number of modules changed; zero matches raise."""
    import torch

    groups: dict[str, dict[str, Any]] = {}
    for k, v in state.items():
        parsed = _te_stem(k)
        if parsed is None:
            continue
        stem, part = parsed
        groups.setdefault(stem, {})[part] = v
    index = _linear_index(text_encoder)
    matched, unmatched = 0, 0
    with torch.no_grad():
        for stem, g in sorted(groups.items()):
            if "down" not in g or "up" not in g:
                continue
            mod = index.get(stem)
            if mod is None:
                unmatched += 1
                LOG.debug("text-encoder LoRA key %s matched no layer", stem)
                continue
            down, up = g["down"].float(), g["up"].float()
            rank = down.shape[0]
            alpha = float(g["alpha"].item()) if "alpha" in g else float(rank)
            delta = (up.reshape(up.shape[0], -1) @ down.reshape(rank, -1)) * (scale * alpha / rank)
            w = mod.weight
            mod.weight.copy_((w.float() + delta.reshape(w.shape)).to(w.dtype))
            matched += 1
    if unmatched:
        LOG.warning("%d text-encoder LoRA layer(s) matched no module and were skipped", unmatched)
    if groups and matched == 0:
        raise UnsupportedModelError("the LoRA matched no layer of the text encoder; is it for this base model?")
    return matched


def text_encoder_path(text_encoder: Any) -> str:
    cls = type(text_encoder).__name__
    path = TEXT_ENCODER_PATHS.get(cls)
    if path is None:
        raise UnsupportedModelError(f"LoRA weights for a {cls} text encoder are not supported")
    return path


def apply_loras(pipe: Any, loras: Sequence[dict], family: str) -> list[dict]:
    """Merge each {'path','scale','name'} in order: split -> load_lora_weights (diffusers parts)
    -> fuse_lora -> unload_lora_weights -> manual text-encoder merges. Returns per-LoRA
    {'name','changed': {component: modules changed}}; raises when a component the LoRA has keys
    for changed in 0 modules."""
    if not loras:
        return []
    from safetensors.torch import load_file

    denoiser_name = "unet" if getattr(pipe, "unet", None) is not None else "transformer"
    results = []
    for i, lora in enumerate(loras):
        name, scale = str(lora["name"]), float(lora["scale"])
        state = load_file(str(lora["path"]))
        classify_lora_keys(state)
        parts = split_state(state)
        if parts["dropped"]:
            LOG.warning("%s: %d T5 (text_encoder_3) tensors dropped; T5 is not exported", name, len(parts["dropped"]))
        if family == "flux2" and (parts["text_encoder"] or parts["text_encoder_2"]):
            raise UnsupportedModelError(f"{name}: FLUX.2 Klein LoRAs must target the transformer only")
        targets: dict[str, Any] = {}
        if parts["denoiser"]:
            targets[denoiser_name] = getattr(pipe, denoiser_name)
        diffusers_part = dict(parts["denoiser"])
        manual: dict[str, dict] = {}
        for te in TEXT_ENCODERS:
            if not parts[te]:
                continue
            module = getattr(pipe, te, None)
            if module is None:
                LOG.warning("%s: %d %s tensors dropped; the model has no %s", name, len(parts[te]), te, te)
                continue
            targets[te] = module
            if text_encoder_path(module) == "diffusers":
                diffusers_part.update(parts[te])
            else:
                manual[te] = parts[te]
        if not targets:
            raise UnsupportedModelError(f"{name}: the LoRA has no weights for any exported component")
        before = {c: _fingerprint(m) for c, m in targets.items()}
        if diffusers_part:
            adapter = f"caipack{i}"
            pipe.load_lora_weights(diffusers_part, adapter_name=adapter)
            pipe.fuse_lora(lora_scale=scale, adapter_names=[adapter])
            pipe.unload_lora_weights()
        for te, te_state in manual.items():
            _merge_text_encoder_lora(targets[te], te_state, scale)
        changed = {c: _changed(before[c], _fingerprint(m)) for c, m in targets.items()}
        for c, n in changed.items():
            if n == 0:
                raise UnsupportedModelError(f"{name}: the LoRA matched no layer of the {c}; is it for this base model?")
        LOG.info("LoRA %s merged at scale %g: %s", name, scale,
                 ", ".join(f"{c} {n} modules" for c, n in changed.items()))
        results.append({"name": name, "changed": changed})
    return results
