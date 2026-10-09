# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from coreai_diffusion_converter import lora
from coreai_diffusion_converter.errors import UnsupportedModelError, UsageError
import tiny_models as T


@pytest.mark.parametrize("arg,expected", [
    ("civitai:118427", ("civitai:118427", 1.0)),
    ("civitai:3", ("civitai:3", 1.0)),
    ("civitai:118427:0.8", ("civitai:118427", 0.8)),
    ("civitai:1@2:0.7", ("civitai:1@2", 0.7)),
    ("civitai:1@2:-0.5", ("civitai:1@2", -0.5)),
    ("https://civitai.com/models/1?modelVersionId=2:0.5", ("https://civitai.com/models/1?modelVersionId=2", 0.5)),
    ("https://civitai.com/models/1?modelVersionId=2", ("https://civitai.com/models/1?modelVersionId=2", 1.0)),
    ("https://civitai.com/models/120096", ("https://civitai.com/models/120096", 1.0)),
    ("org/repo:0.3", ("org/repo", 0.3)),
    ("org/repo/style.safetensors:1.5", ("org/repo/style.safetensors", 1.5)),
])
def test_parse_lora_arg(arg, expected):
    assert lora.parse_lora_arg(arg) == expected


def test_parse_local_path_with_scale(tmp_path, monkeypatch):
    p = tmp_path / "style.safetensors"
    p.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    assert lora.parse_lora_arg("./style.safetensors:0.6") == ("./style.safetensors", 0.6)
    assert lora.parse_lora_arg(f"{p}:2") == (str(p), 2.0)


@pytest.mark.parametrize("arg", ["civitai:1@2:0", "org/repo:5", "org/repo:-5", "org/repo:0.0"])
def test_bad_scales_refused(arg):
    with pytest.raises(UsageError):
        lora.parse_lora_arg(arg)


def test_too_many_loras():
    assert len(lora.parse_lora_args([f"civitai:{i}" for i in range(8)])) == 8
    with pytest.raises(UsageError, match="at most 8"):
        lora.parse_lora_args([f"civitai:{i}" for i in range(9)])


def test_classify_lora_keys():
    assert lora.classify_lora_keys(["lora_unet_x.lora_down.weight", "lora_te1_y.alpha"]) == "kohya"
    assert lora.classify_lora_keys(["unet.down_blocks.0.attn.to_q.lora_A.weight"]) == "peft"
    assert lora.classify_lora_keys(["unet.down.to_q.lora.down.weight"]) == "diffusers"
    for marker in ("lora_unet_x.hada_w1_a", "lora_unet_x.lokr_w1", "lora_unet_x.oft_blocks", "lora_unet_x.lora_mid.weight",
                   "lora_unet_x.dora_scale", "lora_unet_x.ia3.weight"):
        with pytest.raises(UnsupportedModelError, match="LyCORIS"):
            lora.classify_lora_keys([marker, "lora_unet_y.lora_down.weight"])
    with pytest.raises(UnsupportedModelError, match="no LoRA"):
        lora.classify_lora_keys(["model.diffusion_model.input_blocks.0.0.weight"])


def _shapes(**named):
    return {k.replace("__", "."): v for k, v in named.items()}


def write(tmp_path, name, shapes):
    return T.save({k: torch.zeros(*s) for k, s in shapes.items()}, tmp_path / name)


@pytest.mark.parametrize("dim,family", [(768, "sd1"), (1024, "sd2"), (2048, "sdxl")])
def test_family_from_unet_cross_attention(tmp_path, dim, family):
    p = write(tmp_path, "a.safetensors", {
        "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_down.weight": (4, dim),
        "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_up.weight": (640, 4)})
    assert lora.detect_lora_family(p) == family


def test_sd3_kohya_te_keys_are_not_sdxl():
    shapes = {"lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": (4, 768),
              "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": (4, 1280),
              "lora_te3_encoder_block_0_layer_0_SelfAttention_q.lora_down.weight": (4, 4096),
              "lora_unet_joint_blocks_0_context_block_attn_qkv.lora_down.weight": (4, 1536)}
    assert lora.family_from_shapes(shapes) == "sd3"
    assert lora.family_from_shapes({k: v for k, v in shapes.items() if "unet" not in k}) == "sd3"


def test_flux2_transformer_keys_are_not_sd3():
    shapes = {"transformer.single_transformer_blocks.19.attn.to_q.lora_A.weight": (4, 3072),
              "transformer.transformer_blocks.4.attn.add_q_proj.lora_A.weight": (4, 3072)}
    assert lora.family_from_shapes(shapes) == "flux2"


def test_flux1_sized_lora_refused():
    with pytest.raises(UnsupportedModelError, match="Klein 4B"):
        lora.family_from_shapes({"transformer.single_transformer_blocks.37.attn.to_q.lora_A.weight": (4, 3072)})
    with pytest.raises(UnsupportedModelError, match="Klein 4B"):
        lora.family_from_shapes({"lora_unet_double_blocks_18_img_attn_qkv.lora_down.weight": (4, 3072)})


def test_text_encoder_only_tie_breaks():
    assert lora.family_from_shapes({"lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": (4, 768)}) == "sdxl"
    assert lora.family_from_shapes({"lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": (4, 768)}) == "sd1"
    assert lora.family_from_shapes({"lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": (4, 1024)}) == "sd2"
    assert lora.family_from_shapes({"text_encoder.text_model.encoder.layers.0.mlp.fc1.lora_A.weight": (4, 768)}) == "sd1"
    assert lora.family_from_shapes({"something.lora_down.weight": (4, 3)}) is None


def test_check_family_messages(tmp_path):
    spec = lora.LoraSpec(raw="x", scale=1.0, source=lora.LoraSource("file", "x.safetensors", None),
                         name="PerfectEyesXL.safetensors", family="sdxl")
    with pytest.raises(UnsupportedModelError, match="PerfectEyesXL.safetensors is an SDXL LoRA; the model is SD 1.x"):
        lora.check_family(spec, "sd1")
    lora.check_family(spec, "sdxl")
    te = write(tmp_path, "te.safetensors", {"lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": (4, 32)})
    flux = lora.LoraSpec(raw="x", scale=1.0, source=lora.LoraSource("file", "te.safetensors", None),
                         name="te.safetensors", path=__import__("pathlib").Path(te))
    with pytest.raises(UnsupportedModelError, match="transformer only"):
        lora.check_family(flux, "flux2")


def test_pickle_lora_refused_before_read(tmp_path):
    from coreai_diffusion_converter import sources

    p = tmp_path / "style.pt"
    p.write_bytes(b"not read")
    with pytest.raises(UnsupportedModelError, match="pickle"):
        sources.probe_lora(str(p), 1.0, client=None)
    with pytest.raises(UnsupportedModelError, match="pickle"):
        lora.check_local_file(p)


def test_local_lora_probe_records_basename_and_family(tmp_path):
    from coreai_diffusion_converter import sources

    nested = tmp_path / "deep" / "er"
    nested.mkdir(parents=True)
    p = write(nested, "eyes.safetensors", {
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight": (4, 2048)})
    spec = sources.probe_lora(p, 0.8, client=None)
    assert spec.name == "eyes.safetensors" and spec.source.ref == "eyes.safetensors" and spec.family == "sdxl"
    assert spec.worker_form() == {"path": str(p), "scale": 0.8, "name": "eyes.safetensors"}
    assert str(tmp_path) not in str(spec.pack_json())
