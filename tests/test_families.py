# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from coreai_diffusion_converter import families as F
from coreai_diffusion_converter.errors import UnsupportedModelError, UsageError
from conftest import CONFIGS, write_weights

DEFAULT = F.Tuning()


def test_exact_class_to_family(family_tree):
    assert F.detect_family(family_tree("sd1")).family == "sd1"
    assert F.detect_family(family_tree("sd2")).family == "sd2"
    assert F.detect_family(family_tree("sd3")).family == "sd3"
    assert F.detect_family(family_tree("flux2")).family == "flux2"


@pytest.mark.parametrize("cls,message", [
    ("StableDiffusionXLImg2ImgPipeline", "SDXL refiner models are not supported"),
    ("StableDiffusionXLInpaintPipeline", "inpainting checkpoints"),
    ("StableDiffusionXLControlNetPipeline", "ControlNet"),
    ("StableDiffusionXLInstructPix2PixPipeline", "image-to-image"),
    ("StableDiffusionInpaintPipeline", "inpainting checkpoints"),
    ("Flux2Pipeline", "Flux2Pipeline is not supported"),
    ("WanPipeline", "WanPipeline is not supported"),
    ("StableDiffusionImg2ImgPipeline", "StableDiffusionImg2ImgPipeline is not supported"),
])
def test_refused_classes(family_tree, cls, message):
    tree = family_tree("sd1", model_index__json={"_class_name": cls})
    with pytest.raises(UnsupportedModelError, match=message) as err:
        F.detect_family(tree)
    assert err.value.exit_code == 3


def test_nine_channel_unet_is_inpainting(family_tree):
    tree = family_tree("sd1", unet__config__json={"in_channels": 9})
    with pytest.raises(UnsupportedModelError, match="inpainting"):
        F.detect_family(tree)


def test_klein_4b_geometry_matches_reference_fixture():
    cfg = json.loads((CONFIGS / "flux2_klein_4b.json").read_text())
    assert {"num_layers": cfg["num_layers"], "num_single_layers": cfg["num_single_layers"],
            "inner_dim": cfg["num_attention_heads"] * cfg["attention_head_dim"],
            "joint_attention_dim": cfg["joint_attention_dim"]} == F.KLEIN_4B_GEOMETRY


def test_flux2_geometry(family_tree):
    ok = family_tree("flux2")
    assert F.check_variant(F.FAMILIES["flux2"], ok, "ios") == []
    other = family_tree("flux2", transformer__config__json={"num_layers": 8, "num_single_layers": 24,
                                                            "joint_attention_dim": 12288})
    with pytest.raises(UnsupportedModelError):
        F.check_variant(F.FAMILIES["flux2"], other, "ios")
    assert "untested" in F.check_variant(F.FAMILIES["flux2"], other, "macos")[0]


def test_sd3_large_refused_for_ios(family_tree):
    tree = family_tree("sd3", transformer__config__json={"num_layers": 38})
    with pytest.raises(UnsupportedModelError):
        F.make_plan(F.FAMILIES["sd3"], tree, "ios", None, None, DEFAULT)
    plan = F.make_plan(F.FAMILIES["sd3"], tree, "macos", None, None, DEFAULT)
    assert any("untested" in w for w in plan.warnings)


def test_default_sizes_and_sample_size(family_tree):
    sd1 = family_tree("sd1")
    p = F.make_plan(F.FAMILIES["sd1"], sd1, "ios", None, None, DEFAULT)
    assert (p.size, p.sample_size, p.precision, p.compression) == (512, None, "fp16", "none")
    sd2 = family_tree("sd2")
    p = F.make_plan(F.FAMILIES["sd2"], sd2, "macos", None, None, DEFAULT)
    assert (p.size, p.sample_size) == (768, None)
    p = F.make_plan(F.FAMILIES["sd2"], sd2, "ios", None, None, DEFAULT)
    assert (p.size, p.sample_size) == (512, 64)
    assert any("degrades" in w for w in p.warnings)
    sd3 = family_tree("sd3")
    p = F.make_plan(F.FAMILIES["sd3"], sd3, "ios", None, None, DEFAULT)
    assert (p.size, p.sample_size, p.compression) == (512, 64, "4bit")
    assert p.components == ["text_encoder", "text_encoder_2", "transformer", "vae_decoder"]
    p = F.make_plan(F.FAMILIES["sd3"], sd3, "macos", None, None, DEFAULT)
    assert (p.size, p.sample_size) == (1024, None)


def test_flux2_components_by_size(family_tree):
    tree = family_tree("flux2")
    p = F.make_plan(F.FAMILIES["flux2"], tree, "ios", None, None, DEFAULT)
    assert p.components == ["transformer_512", "text_encoder", "vae_decoder_half"]
    assert p.multifunction is False and p.sample_size is None
    p = F.make_plan(F.FAMILIES["flux2"], tree, "macos", None, None, DEFAULT)
    assert p.components == ["transformer", "text_encoder", "vae_decoder"]


def test_size_refusals(family_tree):
    with pytest.raises(UsageError):
        F.make_plan(F.FAMILIES["sd1"], family_tree("sd1"), "ios", 768, None, DEFAULT)
    with pytest.raises(UsageError):
        F.make_plan(F.FAMILIES["flux2"], family_tree("flux2"), "ios", 768, None, DEFAULT)


def test_4bit_sd_warns(family_tree):
    p = F.make_plan(F.FAMILIES["sd1"], family_tree("sd1"), "ios", None, "4bit", DEFAULT)
    assert p.compression == "4bit" and any("unmeasured" in w for w in p.warnings)


def test_tuning_flags_refused_on_sd3_and_flux2(family_tree):
    for fam in ("sd3", "flux2"):
        for tuning in (F.Tuning(vae="x"), F.Tuning(clip_skip=2), F.Tuning(prediction_type="epsilon")):
            with pytest.raises(UsageError) as err:
                F.make_plan(F.FAMILIES[fam], family_tree(fam), "macos", None, None, tuning)
            assert err.value.exit_code == 2


def test_clip_skip_range(family_tree):
    with pytest.raises(UsageError):
        F.make_plan(F.FAMILIES["sd1"], family_tree("sd1"), "ios", None, None, F.Tuning(clip_skip=5))


def test_weight_variant(family_tree):
    tree = family_tree("sd1")
    write_weights(tree, ("text_encoder", "unet", "vae"))
    assert F.weight_variant(tree, F.WEIGHT_COMPONENTS["sd1"]) == "fp16"
    write_weights(tree, ("unet",), fp16=False, plain=True)
    assert F.weight_variant(tree, F.WEIGHT_COMPONENTS["sd1"]) is None


@pytest.mark.parametrize("model_type,family", [("v1", "sd1"), ("v2", "sd2"), ("xl_base", "sdxl"), ("sd3", "sd3"),
                                                ("sd35_medium", "sd3"), ("sd35_large", "sd3")])
def test_single_file_mapping(model_type, family):
    assert F.family_for_single_file_type(model_type) == family


@pytest.mark.parametrize("model_type,message", [
    ("xl_refiner", "SDXL refiner"), ("xl_inpaint", "inpainting"), ("playground-v2-5", "Playground v2.5"),
    ("inpainting", "inpainting"),
    ("inpainting_v2", "inpainting"), ("flux-2-dev", "FLUX single-file"), ("flux-dev", "FLUX single-file"),
    ("controlnet", "not supported"), ("wan-t2v-14B", "not supported"),
])
def test_single_file_refusals(model_type, message):
    with pytest.raises(UnsupportedModelError, match=message):
        F.family_for_single_file_type(model_type)


def test_pickle_single_file_refused(tmp_path):
    p = tmp_path / "model.ckpt"
    p.write_bytes(b"x")
    with pytest.raises(UnsupportedModelError, match="pickle"):
        F.detect_single_file_family(p)


def test_single_file_detection_reads_shapes_only(tmp_path):
    import torch
    from safetensors.torch import save_file

    # A minimal v1-style checkpoint: diffusers falls back to "v1"; the UNet input conv is present.
    p = tmp_path / "tiny.safetensors"
    save_file({"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(8, 4, 3, 3)}, str(p))
    assert F.detect_single_file_family(p) == "sd1"
    q = tmp_path / "lora.safetensors"
    save_file({"lora_unet_down.weight": torch.zeros(2, 2)}, str(q))
    with pytest.raises(UnsupportedModelError, match="LoRA"):
        F.detect_single_file_family(q)
    r = tmp_path / "inpaint.safetensors"
    save_file({"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(8, 9, 3, 3)}, str(r))
    with pytest.raises(UnsupportedModelError, match="inpainting"):
        F.detect_single_file_family(r)


# --- SDXL ---------------------------------------------------------------------------------------


def test_sdxl_folder_is_sdxl(family_tree):
    spec = F.detect_family(family_tree("sdxl"))
    assert (spec.family, spec.pipeline, spec.sizes, spec.default_precision) == ("sdxl", "sdxl", (1024,), "4bit")
    assert (spec.default_steps, spec.guidance, spec.scheduler) == (25, 5.0, "dpmpp")


def test_sdxl_refiner_and_variants_refused(family_tree):
    with pytest.raises(UnsupportedModelError, match="refiner"):
        F.detect_family(family_tree("sdxl_refiner"))
    # A refiner relabelled as a base pipeline is still caught (no text_encoder, 2560 projection).
    relabelled = family_tree("sdxl_refiner", model_index__json={"_class_name": "StableDiffusionXLPipeline"})
    with pytest.raises(UnsupportedModelError, match="refiner"):
        F.detect_family(relabelled)
    with pytest.raises(UnsupportedModelError, match="inpainting"):
        F.detect_family(family_tree("sdxl", unet__config__json={"in_channels": 9}))
    with pytest.raises(UnsupportedModelError, match="geometry"):
        F.detect_family(family_tree("sdxl", unet__config__json={"cross_attention_dim": 1024}))


def test_sdxl_ios_refused_with_and_without_a_tree(family_tree):
    spec = F.FAMILIES["sdxl"]
    for tree in (family_tree("sdxl"), None):
        with pytest.raises(UnsupportedModelError, match="Mac-only"):
            F.make_plan(spec, tree, "ios", None, None, DEFAULT)


def test_sdxl_plan(family_tree):
    spec = F.FAMILIES["sdxl"]
    tree = family_tree("sdxl")
    p = F.make_plan(spec, tree, "macos", None, None, DEFAULT)
    assert (p.size, p.sample_size, p.precision, p.compression) == (1024, None, "4bit", "4bit")
    assert p.components == ["text_encoder", "text_encoder_2", "unet", "vae_decoder"]
    assert F.make_plan(spec, None, "macos", None, None, DEFAULT).size == 1024
    for size in (512, 768):
        with pytest.raises(UsageError, match="not available"):
            F.make_plan(spec, tree, "macos", size, None, DEFAULT)
    odd = family_tree("sdxl", unet__config__json={"sample_size": 96})
    p = F.make_plan(spec, odd, "macos", None, None, DEFAULT)
    assert p.sample_size == 128 and any("traced at 1024" in w for w in p.warnings)


def test_sdxl_tuning(family_tree):
    spec = F.FAMILIES["sdxl"]
    tree = family_tree("sdxl")
    with pytest.raises(UsageError, match="does not apply to SDXL"):
        F.make_plan(spec, tree, "macos", None, None, F.Tuning(clip_skip=2))
    p = F.make_plan(spec, tree, "macos", None, None, F.Tuning(vae="x", prediction_type="v_prediction"))
    assert (p.vae, p.prediction_type) == ("x", "v_prediction")


def test_sdxl_single_file_needs_a_four_channel_unet(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    monkeypatch.setattr(F, "family_for_single_file_type", lambda t: "sdxl")
    p = tmp_path / "xl.safetensors"
    save_file({"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(8, 4, 3, 3)}, str(p))
    assert F.detect_single_file_family(p) == "sdxl"
    q = tmp_path / "xl-inpaint.safetensors"
    save_file({"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(8, 9, 3, 3)}, str(q))
    with pytest.raises(UnsupportedModelError, match="inpainting"):
        F.detect_single_file_family(q)
