# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from coreai_diffusion_converter import tuning
from coreai_diffusion_converter.errors import UsageError


def tiny_clip(layers=4):
    from transformers import CLIPTextConfig, CLIPTextModel

    cfg = CLIPTextConfig(num_hidden_layers=layers, hidden_size=32, intermediate_size=37, num_attention_heads=4,
                         vocab_size=1000, projection_dim=32, bos_token_id=0, eos_token_id=2, pad_token_id=1)
    return CLIPTextModel(cfg).eval()


def test_clip_skip_truncates_layers_and_config():
    te = tiny_clip(4)
    assert tuning.apply_clip_skip(te, 2) == 3
    assert len(tuning.clip_encoder(te).layers) == 3 and te.config.num_hidden_layers == 3
    te = tiny_clip(4)
    assert tuning.apply_clip_skip(te, 1) == 4 and len(tuning.clip_encoder(te).layers) == 4


def test_clip_skip_output_equals_penultimate_with_final_norm():
    import torch

    te = tiny_clip(4)
    ids = torch.tensor([[0, 5, 6, 7, 2]])
    with torch.no_grad():
        full = te(ids, output_hidden_states=True)
        expected = te.final_layer_norm(full.hidden_states[-2])
        tuning.apply_clip_skip(te, 2)
        got = te(ids).last_hidden_state
    assert torch.allclose(got, expected, atol=1e-6)


def test_clip_skip_range():
    with pytest.raises(UsageError):
        tuning.apply_clip_skip(tiny_clip(4), 5)
    with pytest.raises(UsageError):
        tuning.apply_clip_skip(tiny_clip(2), 4)


def test_vae_mismatch_is_a_usage_error():
    pipe_vae = SimpleNamespace(block_out_channels=[128, 256, 512, 512])
    with pytest.raises(UsageError, match="4 latent channels") as err:
        tuning.check_vae(SimpleNamespace(latent_channels=16, block_out_channels=[128, 256, 512, 512]), pipe_vae)
    assert err.value.exit_code == 2
    with pytest.raises(UsageError, match="block_out_channels"):
        tuning.check_vae(SimpleNamespace(latent_channels=4, block_out_channels=[64, 128]), pipe_vae)
    tuning.check_vae(SimpleNamespace(latent_channels=4, block_out_channels=[128, 256, 512, 512]), pipe_vae)


def test_vae_from_tiny_folder_mismatch(tmp_path):
    from diffusers import AutoencoderKL

    AutoencoderKL(latent_channels=8, block_out_channels=(32,), norm_num_groups=8).save_pretrained(tmp_path / "vae")
    import torch

    vae = tuning.load_vae(str(tmp_path), torch.float32)
    with pytest.raises(UsageError):
        tuning.check_vae(vae.config, SimpleNamespace(block_out_channels=[32]))


def test_pickle_vae_refused(tmp_path):
    p = tmp_path / "vae.pt"
    p.write_bytes(b"x")
    with pytest.raises(UsageError, match="pickle"):
        tuning.load_vae(str(p), None)


def test_prediction_type_sets_scheduler_config():
    from diffusers import PNDMScheduler

    s = PNDMScheduler()
    assert tuning.effective_prediction_type(s) == "epsilon"
    tuning.apply_prediction_type(s, "v_prediction")
    assert s.config.prediction_type == "v_prediction"
    with pytest.raises(UsageError):
        tuning.apply_prediction_type(s, "flow")


def test_apply_tuning_reports_effective_type():
    from diffusers import PNDMScheduler

    pipe = SimpleNamespace(text_encoder=tiny_clip(3), scheduler=PNDMScheduler(), vae=None)
    assert tuning.apply_tuning(pipe, vae=None, clip_skip=2, prediction_type="v_prediction") == \
        {"prediction_type": "v_prediction"}
    assert len(tuning.clip_encoder(pipe.text_encoder).layers) == 2


def test_single_file_vae_takes_its_config_from_the_tree(tmp_path, monkeypatch):
    # The worker runs offline: a single-file VAE must not ask the Hub for a default config.
    import diffusers

    (tmp_path / "tree" / "vae").mkdir(parents=True)
    (tmp_path / "tree" / "vae" / "config.json").write_text("{}")
    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(diffusers.AutoencoderKL, "from_single_file",
                        classmethod(lambda cls, path, **kw: calls.append(kw) or "vae"))
    assert tuning.load_vae(str(vae), None, tmp_path / "tree") == "vae"
    assert calls[0]["config"] == str(tmp_path / "tree") and calls[0]["subfolder"] == "vae"
