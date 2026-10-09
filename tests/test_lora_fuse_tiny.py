# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""LoRA fusing on tiny in-memory pipelines (CPU, offline, no tokenizers).

The first two tests are step 0 of the LoRA work: they decide, per text-encoder class, whether a
Kohya text-encoder LoRA goes through diffusers or through the converter's manual merge, and they
pin that decision (lora.TEXT_ENCODER_PATHS).
"""

import copy

import pytest
import torch

from coreai_diffusion_converter import lora
from coreai_diffusion_converter.errors import UnsupportedModelError
import tiny_models as T

ATOL = 1e-5


def _w(module):
    return module.weight.detach().clone()


def test_kohya_te_lora_attaches_to_flat_clip(tmp_path):
    """transformers 5 CLIPTextModel is flat; its Kohya TE LoRA is merged by hand, exactly."""
    pipe = T.sd1_pipe()
    te = pipe.text_encoder
    assert not hasattr(te, "text_model")  # the flat layout this decision depends on
    assert lora.TEXT_ENCODER_PATHS["CLIPTextModel"] == "manual" == lora.text_encoder_path(te)
    state = {**T.kohya("lora_te_text_model_encoder_layers_0_self_attn_q_proj", 32, 32),
             **T.kohya("lora_te_text_model_encoder_layers_1_mlp_fc1", 32, 37, seed=1)}
    q0, fc0 = _w(te.encoder.layers[0].self_attn.q_proj), _w(te.encoder.layers[1].mlp.fc1)
    out = lora.apply_loras(pipe, [{"path": T.save(state, tmp_path / "te.safetensors"), "scale": 0.7, "name": "te"}],
                           "sd1")
    assert out[0]["changed"] == {"text_encoder": 2}
    assert torch.allclose(_w(te.encoder.layers[0].self_attn.q_proj) - q0,
                          T.expected_delta(state, "lora_te_text_model_encoder_layers_0_self_attn_q_proj", 0.7),
                          atol=ATOL)
    assert torch.allclose(_w(te.encoder.layers[1].mlp.fc1) - fc0,
                          T.expected_delta(state, "lora_te_text_model_encoder_layers_1_mlp_fc1", 0.7), atol=ATOL)


def test_diffusers_cannot_attach_to_flat_clip():
    """Why the flat class is on the manual path: diffusers looks the ranks up under
    text_model.* module names, finds none, and fails (which would also abort the UNet part). If
    this starts passing, step 0 should be re-run and the class moved to the diffusers path."""
    pipe = T.sd1_pipe()
    state = T.kohya("lora_te_text_model_encoder_layers_0_self_attn_q_proj", 32, 32)
    before = _w(pipe.text_encoder.encoder.layers[0].self_attn.q_proj)
    try:
        pipe.load_lora_weights(dict(state), adapter_name="probe")
        pipe.fuse_lora()
    except (IndexError, ValueError, KeyError):
        return
    assert torch.equal(before, _w(pipe.text_encoder.encoder.layers[0].self_attn.q_proj))


def test_kohya_te2_lora_attaches_to_clip_with_projection(tmp_path):
    """CLIPTextModelWithProjection still nests text_model.; diffusers attaches its LoRA exactly."""
    pipe = T.sdxl_pipe()
    te2 = pipe.text_encoder_2
    assert hasattr(te2, "text_model")
    assert lora.TEXT_ENCODER_PATHS["CLIPTextModelWithProjection"] == "diffusers" == lora.text_encoder_path(te2)
    key = "lora_te2_text_model_encoder_layers_0_self_attn_out_proj"
    state = T.kohya(key, 32, 32)
    w0 = _w(te2.text_model.encoder.layers[0].self_attn.out_proj)
    snapshot = {k: v.clone() for k, v in te2.state_dict().items()}
    out = lora.apply_loras(pipe, [{"path": T.save(state, tmp_path / "te2.safetensors"), "scale": 1.3, "name": "te2"}],
                           "sdxl")
    assert out[0]["changed"] == {"text_encoder_2": 1}
    assert torch.allclose(_w(te2.text_model.encoder.layers[0].self_attn.out_proj) - w0,
                          T.expected_delta(state, key, 1.3), atol=ATOL)
    changed = [k for k, v in te2.state_dict().items() if not torch.equal(v, snapshot[k])]
    assert changed == ["text_model.encoder.layers.0.self_attn.out_proj.weight"]
    assert not any("lora" in n for n, _ in te2.named_modules())


def test_flat_te_keys_never_block_the_unet(tmp_path):
    """TE1 (manual) keys beside UNet and TE2 keys: everything fuses, nothing aborts."""
    pipe = T.sdxl_pipe()
    unet_key = "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k"
    state = {**T.kohya(unet_key, 64, 64, seed=3),
             **T.kohya("lora_te1_text_model_encoder_layers_0_self_attn_q_proj", 32, 32, seed=4),
             **T.kohya("lora_te2_text_model_encoder_layers_1_mlp_fc2", 37, 32, seed=5)}
    k0 = _w(pipe.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn2.to_k)
    out = lora.apply_loras(pipe, [{"path": T.save(state, tmp_path / "x.safetensors"), "scale": 1.0, "name": "x"}],
                           "sdxl")
    assert out[0]["changed"] == {"unet": 1, "text_encoder": 1, "text_encoder_2": 1}
    assert torch.allclose(_w(pipe.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn2.to_k) - k0,
                          T.expected_delta(state, unet_key), atol=ATOL)
    assert not any("lora" in n for n, _ in pipe.unet.named_modules())


@pytest.mark.parametrize("nested", [False, True])
def test_manual_merge_maps_keys_on_both_layouts(nested):
    te = T.clip(with_projection=nested)
    enc = te.text_model.encoder if nested else te.encoder
    state = {**T.kohya("lora_te1_text_model_encoder_layers_0_self_attn_q_proj", 32, 32),
             **T.kohya("lora_te1_text_model_encoder_layers_1_self_attn_out_proj", 32, 32, seed=1),
             **T.kohya("lora_te1_text_model_encoder_layers_0_mlp_fc1", 32, 37, seed=2)}
    g = torch.Generator().manual_seed(9)
    down, up = torch.randn(T.RANK, 37, generator=g), torch.randn(32, T.RANK, generator=g)
    state["text_encoder.text_model.encoder.layers.1.mlp.fc2.lora_A.weight"] = down  # diffusers / PEFT spelling
    state["text_encoder.text_model.encoder.layers.1.mlp.fc2.lora_B.weight"] = up
    before = {n: _w(m) for n, m in (("q", enc.layers[0].self_attn.q_proj), ("o", enc.layers[1].self_attn.out_proj),
                                    ("fc1", enc.layers[0].mlp.fc1), ("fc2", enc.layers[1].mlp.fc2))}
    assert lora._merge_text_encoder_lora(te, state, 0.5) == 4
    assert torch.allclose(_w(enc.layers[0].self_attn.q_proj) - before["q"],
                          T.expected_delta(state, "lora_te1_text_model_encoder_layers_0_self_attn_q_proj", 0.5), atol=ATOL)
    assert torch.allclose(_w(enc.layers[1].self_attn.out_proj) - before["o"],
                          T.expected_delta(state, "lora_te1_text_model_encoder_layers_1_self_attn_out_proj", 0.5),
                          atol=ATOL)
    assert torch.allclose(_w(enc.layers[0].mlp.fc1) - before["fc1"],
                          T.expected_delta(state, "lora_te1_text_model_encoder_layers_0_mlp_fc1", 0.5), atol=ATOL)
    assert torch.allclose(_w(enc.layers[1].mlp.fc2) - before["fc2"], 0.5 * (up @ down), atol=ATOL)  # alpha = rank


def test_sd1_unet_and_te_fused_and_lora_layers_gone(tmp_path):
    pipe = T.sd1_pipe()
    unet_key = "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v"
    state = {**T.kohya(unet_key, 32, 64), **T.kohya("lora_te_text_model_encoder_layers_0_self_attn_k_proj", 32, 32)}
    out = lora.apply_loras(pipe, [{"path": T.save(state, tmp_path / "a.safetensors"), "scale": 1.0, "name": "a"}],
                           "sd1")
    assert out[0]["changed"]["unet"] >= 1 and out[0]["changed"]["text_encoder"] == 1
    for module in (pipe.unet, pipe.text_encoder):
        assert not any("lora" in n for n, _ in module.named_modules())


def test_sdxl_all_three_components_change(tmp_path):
    pipe = T.sdxl_pipe()
    state = {**T.kohya("lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_k", 64, 64),
             **T.kohya("lora_te1_text_model_encoder_layers_1_self_attn_v_proj", 32, 32, seed=1),
             **T.kohya("lora_te2_text_model_encoder_layers_0_self_attn_q_proj", 32, 32, seed=2),
             # Kohya always trains out_proj too; diffusers infers the state-dict format from it (a
             # q/k/v-only text-encoder dict fails its inference).
             **T.kohya("lora_te2_text_model_encoder_layers_0_self_attn_out_proj", 32, 32, seed=6)}
    out = lora.apply_loras(pipe, [{"path": T.save(state, tmp_path / "a.safetensors"), "scale": 0.8, "name": "a"}],
                           "sdxl")
    assert set(out[0]["changed"]) == {"unet", "text_encoder", "text_encoder_2"}
    assert all(n >= 1 for n in out[0]["changed"].values())


def test_lora_matching_nothing_is_refused(tmp_path):
    pipe = T.sd1_pipe()
    state = T.kohya("lora_te_text_model_encoder_layers_7_self_attn_q_proj", 32, 32)  # no layer 7
    with pytest.raises(UnsupportedModelError, match="matched no layer"):
        lora.apply_loras(pipe, [{"path": T.save(state, tmp_path / "a.safetensors"), "scale": 1.0, "name": "a"}],
                         "sd1")


def test_fuse_happens_before_clip_skip(tmp_path):
    """make_loader merges first, then clip skip 2 truncates: equal to 'fuse full TE, then truncate'."""
    from diffusers import PNDMScheduler

    from coreai_diffusion_converter import _export_worker as W
    from coreai_diffusion_converter.tuning import apply_clip_skip

    key = "lora_te_text_model_encoder_layers_1_mlp_fc2"  # the last layer: removed by clip skip 2
    key0 = "lora_te_text_model_encoder_layers_0_self_attn_q_proj"
    state = {**T.kohya(key, 37, 32), **T.kohya(key0, 32, 32, seed=1)}
    path = T.save(state, tmp_path / "a.safetensors")
    pipe = T.sd1_pipe()
    reference = copy.deepcopy(pipe.text_encoder)
    lora._merge_text_encoder_lora(reference, state, 1.0)
    apply_clip_skip(reference, 2)

    def original(tree, **kw):
        pipe.scheduler = PNDMScheduler()
        return pipe

    result: dict = {}
    load = W.make_loader(original, pack_id="p", tree="t", variant=None, sample_size=None, family="sd1",
                         tuning={"clip_skip": 2}, result=result,
                         loras=[{"path": path, "scale": 1.0, "name": "a"}])
    load("p")
    ids = torch.tensor([[0, 5, 6, 7, 2]])
    with torch.no_grad():
        assert torch.allclose(pipe.text_encoder(ids).last_hidden_state, reference(ids).last_hidden_state, atol=ATOL)
    assert result["loras"][0]["changed"] == {"text_encoder": 2}


def test_two_loras_in_order_equal_sequential_merges(tmp_path):
    pipe = T.sd1_pipe()
    k = "lora_te_text_model_encoder_layers_0_self_attn_q_proj"
    a, b = T.kohya(k, 32, 32, seed=1), T.kohya(k, 32, 32, seed=2)
    w0 = _w(pipe.text_encoder.encoder.layers[0].self_attn.q_proj)
    lora.apply_loras(pipe, [{"path": T.save(a, tmp_path / "a.safetensors"), "scale": 0.5, "name": "a"},
                            {"path": T.save(b, tmp_path / "b.safetensors"), "scale": -1.0, "name": "b"}], "sd1")
    expected = w0 + T.expected_delta(a, k, 0.5) + T.expected_delta(b, k, -1.0)
    assert torch.allclose(_w(pipe.text_encoder.encoder.layers[0].self_attn.q_proj), expected, atol=ATOL)
