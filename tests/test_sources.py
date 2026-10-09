# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace

import pytest

from coreai_diffusion_converter import sources
from coreai_diffusion_converter.errors import UnsupportedModelError, UsageError

SD_LISTING = [
    "README.md", "LICENSE.md", "model_index.json", "v1-5-pruned.ckpt", "v1-5-pruned.safetensors",
    "feature_extractor/preprocessor_config.json",
    "safety_checker/config.json", "safety_checker/model.fp16.safetensors", "safety_checker/model.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/model.fp16.safetensors", "text_encoder/model.safetensors",
    "text_encoder/pytorch_model.bin", "text_encoder/flax_model.msgpack",
    "tokenizer/merges.txt", "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.safetensors",
    "unet/diffusion_pytorch_model.bin", "unet/diffusion_flax_model.msgpack",
    "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.safetensors",
    "vae_encoder/model.onnx",
]


def test_download_filter_prefers_fp16_and_skips_unneeded():
    files = sources.select_download_files(SD_LISTING, "sd1")
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in files
    assert "unet/diffusion_pytorch_model.safetensors" not in files
    assert not any(f.startswith(("feature_extractor/", "safety_checker/")) for f in files)
    assert not any(f.endswith((".bin", ".ckpt", ".msgpack", ".onnx")) for f in files)
    assert "v1-5-pruned.safetensors" not in files  # top-level single files are not needed
    assert {"model_index.json", "README.md", "LICENSE.md", "tokenizer/vocab.json",
            "scheduler/scheduler_config.json"} <= set(files)


def test_download_filter_falls_back_to_plain_weights_when_fp16_is_incomplete():
    listing = [f for f in SD_LISTING if f != "vae/diffusion_pytorch_model.fp16.safetensors"]
    files = sources.select_download_files(listing, "sd1")
    assert "unet/diffusion_pytorch_model.safetensors" in files
    assert not any(f.endswith(".fp16.safetensors") for f in files)


def test_sd3_filter_skips_t5():
    listing = ["model_index.json", "LICENSE.md", "text_encoder/model.safetensors", "text_encoder/config.json",
               "text_encoder_2/model.safetensors", "text_encoder_2/config.json",
               "text_encoder_3/model-00001-of-00002.safetensors", "tokenizer_3/spiece.model",
               "tokenizer/vocab.json", "tokenizer_2/vocab.json",
               "transformer/diffusion_pytorch_model.safetensors", "transformer/config.json",
               "vae/diffusion_pytorch_model.safetensors", "vae/config.json"]
    files = sources.select_download_files(listing, "sd3")
    assert not any(f.startswith(("text_encoder_3/", "tokenizer_3/")) for f in files)
    assert "text_encoder_2/model.safetensors" in files


def test_bin_only_repo_refused_early():
    listing = ["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.bin",
               "text_encoder/pytorch_model.bin", "vae/diffusion_pytorch_model.bin"]
    with pytest.raises(UnsupportedModelError, match="no .safetensors weights") as err:
        sources.select_download_files(listing, "sd1")
    assert err.value.exit_code == 3


def test_classify(tmp_path):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "model_index.json").write_text("{}")
    assert sources.classify(str(folder)) == "folder"
    st = tmp_path / "a.safetensors"
    st.write_bytes(b"x")
    assert sources.classify(str(st)) == "single_file"
    ck = tmp_path / "a.ckpt"
    ck.write_bytes(b"x")
    with pytest.raises(UnsupportedModelError, match="pickle"):
        sources.classify(str(ck))
    assert sources.classify("org/name") == "hf"
    with pytest.raises(UsageError):
        sources.classify("not a thing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(UsageError):
        sources.classify(str(empty))


def test_folder_ref_is_never_absolute(family_tree):
    tree = family_tree("sd1")
    probe = sources.probe_source(str(tree))
    assert probe.ref == tree.name and not Path(probe.ref).is_absolute()
    resolved = sources.resolve_source(probe, base=None, work_dir=tree.parent / "w", pack_id="x")
    assert resolved.ref == tree.name and resolved.kind == "folder"


def test_hub_probe_and_gated_message(monkeypatch, family_tree):
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    tree = family_tree("sd1")
    info = SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(rfilename=f) for f in SD_LISTING],
                           card_data={"license": "creativeml-openrail-m"})

    class Api:
        def model_info(self, repo_id, revision=None):
            return info

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tree))
    probe = sources.probe_source("org/model")
    assert probe.kind == "hf" and probe.ref == "org/model" and probe.revision == "a" * 40
    assert probe.spec.family == "sd1" and probe.has_licence
    assert probe.card_license == ("creativeml-openrail-m", None)
    assert not any("feature_extractor" in f for f in probe.download_files)

    class Gated:
        def model_info(self, repo_id, revision=None):
            raise GatedRepoError("gated", response=SimpleNamespace(status_code=401, headers={}, request=None))

    monkeypatch.setattr(huggingface_hub, "HfApi", Gated)
    with pytest.raises(UnsupportedModelError, match="hf auth login"):
        sources.probe_source("org/gated")


def test_single_file_tree_passes_feature_extractor_none(monkeypatch, tmp_path):
    import diffusers

    calls = {}

    class FakePipe:
        scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type="epsilon"))

        def save_pretrained(self, path, safe_serialization=True):
            calls["saved"] = path
            Path(path, "model_index.json").write_text("{}")

    def from_single_file(path, **kw):
        calls["kw"] = kw
        return FakePipe()

    monkeypatch.setattr(diffusers.StableDiffusionPipeline, "from_single_file", staticmethod(from_single_file))
    monkeypatch.setattr(sources, "load_single_file_text_encoder", lambda *a: "text-encoder")
    ckpt = tmp_path / "m.safetensors"
    ckpt.write_bytes(b"x")
    probe = sources.SourceProbe(kind="single_file", ref="m.safetensors",
                                spec=sources.FAMILIES["sd1"], config_tree=None, path=ckpt)
    resolved = sources.resolve_source(probe, base=None, work_dir=tmp_path / "w", pack_id="m")
    assert calls["kw"]["feature_extractor"] is None and calls["kw"]["safety_checker"] is None
    assert calls["kw"]["text_encoder"] == "text-encoder"
    assert calls["kw"]["config"] == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert resolved.tree == tmp_path / "w" / "tree" / "m" and resolved.ref == "m.safetensors"
    assert resolved.licence_dir is None


def test_vae_hub_files_prefer_the_vae_subfolder():
    files = sources.select_vae_files(SD_LISTING)
    assert files == ["vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
                     "vae/diffusion_pytorch_model.safetensors"]
    assert sources.select_vae_files(["config.json", "diffusion_pytorch_model.safetensors", "a/b.safetensors"]) == \
        ["config.json", "diffusion_pytorch_model.safetensors"]
    with pytest.raises(UsageError):
        sources.select_vae_files(["vae/config.json", "vae/diffusion_pytorch_model.bin"])


def test_fetch_vae_keeps_local_paths_and_refuses_garbage(tmp_path):
    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"x")
    assert sources.fetch_vae(None) is None
    assert sources.fetch_vae(str(vae)) == str(vae)
    with pytest.raises(UsageError):
        sources.fetch_vae("not a hub id")


# --- Civitai and SDXL ---------------------------------------------------------------------------

ANIMAGINE_LISTING = [
    ".gitattributes", "README.md", "animagine-xl-4.0-opt.safetensors", "animagine-xl-4.0.safetensors",
    "model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json", "text_encoder/model.safetensors",
    "text_encoder_2/config.json", "text_encoder_2/model.safetensors", "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "tokenizer_2/merges.txt", "tokenizer_2/special_tokens_map.json", "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors", "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]


def test_classify_civitai():
    assert sources.classify("civitai:1188071@1408658") == "civitai"
    assert sources.classify("https://civitai.com/models/1188071") == "civitai"


def test_sdxl_download_filter_on_the_animagine_listing():
    files = sources.select_download_files(ANIMAGINE_LISTING, "sdxl")
    assert "animagine-xl-4.0.safetensors" not in files and "animagine-xl-4.0-opt.safetensors" not in files
    assert {"vae/config.json", "vae/diffusion_pytorch_model.safetensors", "text_encoder_2/model.safetensors",
            "tokenizer_2/vocab.json", "unet/diffusion_pytorch_model.safetensors"} <= set(files)


def _sdxl_ckpt(tmp_path, *extra, text_encoders=True):
    import torch
    from safetensors.torch import save_file

    state = {"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(2, 4, 3, 3)}
    if text_encoders:
        state["conditioner.embedders.0.transformer.text_model.final_layer_norm.weight"] = torch.zeros(2)
    for k in extra:
        state[k] = torch.zeros(1)
    p = tmp_path / "xl.safetensors"
    save_file(state, str(p))
    return p


class _FakeXL:
    def __init__(self, calls):
        from diffusers import EulerDiscreteScheduler

        self.calls = calls
        self.scheduler = EulerDiscreteScheduler(prediction_type="epsilon")

    def save_pretrained(self, path, safe_serialization=True):
        self.scheduler.save_pretrained(Path(path) / "scheduler")
        Path(path, "model_index.json").write_text("{}")


def _patch_xl(monkeypatch, calls):
    import diffusers

    def from_single_file(path, **kw):
        calls["single"] = kw
        return _FakeXL(calls)

    def from_pretrained(base, **kw):
        calls["pretrained"] = (base, kw)
        return _FakeXL(calls)

    monkeypatch.setattr(diffusers.StableDiffusionXLPipeline, "from_single_file", staticmethod(from_single_file))
    monkeypatch.setattr(diffusers.StableDiffusionXLPipeline, "from_pretrained", staticmethod(from_pretrained))
    monkeypatch.setattr(diffusers.UNet2DConditionModel, "from_single_file",
                        staticmethod(lambda path, **kw: calls.setdefault("unet", kw) and "unet"))
    monkeypatch.setattr(sources, "load_single_file_sdxl_text_encoders", lambda *a: ("te1", "te2"))


def _probe(path):
    return sources.SourceProbe(kind="single_file", ref=path.name, spec=sources.FAMILIES["sdxl"], config_tree=None,
                               path=path)


def _saved_prediction(tree):
    import json

    return json.loads((tree / "scheduler" / "scheduler_config.json").read_text())["prediction_type"]


def test_sdxl_single_file_v_pred_key_sets_v_prediction(tmp_path, monkeypatch):
    calls = {}
    _patch_xl(monkeypatch, calls)
    path = _sdxl_ckpt(tmp_path, "v_pred")
    resolved = sources.resolve_source(_probe(path), base=None, work_dir=tmp_path / "w", pack_id="xl")
    assert resolved.prediction_type == "v_prediction" and _saved_prediction(resolved.tree) == "v_prediction"
    assert calls["single"]["config"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert calls["single"]["text_encoder"] == "te1" and calls["single"]["text_encoder_2"] == "te2"
    assert resolved.notes == ()


def test_sdxl_explicit_prediction_type_wins(tmp_path, monkeypatch):
    _patch_xl(monkeypatch, {})
    path = _sdxl_ckpt(tmp_path, "v_pred")
    resolved = sources.resolve_source(_probe(path), base=None, work_dir=tmp_path / "w", pack_id="xl",
                                      prediction_type="epsilon")
    assert resolved.prediction_type == "epsilon" and _saved_prediction(resolved.tree) == "epsilon"


def test_sdxl_ztsnr_key_warns_and_notes(tmp_path, monkeypatch, caplog):
    _patch_xl(monkeypatch, {})
    path = _sdxl_ckpt(tmp_path, "v_pred", "ztsnr")
    resolved = sources.resolve_source(_probe(path), base=None, work_dir=tmp_path / "w", pack_id="xl")
    assert "zero-terminal-SNR" in caplog.text and any("Zero-terminal-SNR" in n for n in resolved.notes)


def test_unet_only_sdxl_file_needs_base(tmp_path, monkeypatch):
    calls = {}
    _patch_xl(monkeypatch, calls)
    path = _sdxl_ckpt(tmp_path, text_encoders=False)
    with pytest.raises(UsageError, match="--base"):
        sources.resolve_source(_probe(path), base=None, work_dir=tmp_path / "w", pack_id="xl")
    resolved = sources.resolve_source(_probe(path), base="org/sdxl-base", work_dir=tmp_path / "w2", pack_id="xl")
    assert calls["pretrained"][0] == "org/sdxl-base" and calls["pretrained"][1]["unet"] == "unet"
    assert calls["unet"]["subfolder"] == "unet" and resolved.prediction_type == "epsilon"


def test_slow_tokenizer_files_are_written_from_tokenizer_json(tmp_path):
    """transformers 5 saves only tokenizer.json; the app's BPE tokenizer needs vocab.json,
    merges.txt and (for SDXL's pad token) special_tokens_map.json."""
    import json

    tok = tmp_path / "tokenizer_2"
    tok.mkdir()
    (tok / "tokenizer.json").write_text(json.dumps({"model": {
        "type": "BPE", "vocab": {"a": 0, "b": 1, "ab</w>": 2}, "merges": [["a", "b</w>"], "a b"]}}))
    (tok / "tokenizer_config.json").write_text(json.dumps({"pad_token": "!", "bos_token": "<s>", "x": 1}))
    other = tmp_path / "tokenizer"
    other.mkdir()
    (other / "vocab.json").write_text("{}")
    (other / "merges.txt").write_text("#version: 0.2\n")
    assert sources.ensure_slow_tokenizer_files(tmp_path) == ["tokenizer_2"]
    assert json.loads((tok / "vocab.json").read_text()) == {"a": 0, "b": 1, "ab</w>": 2}
    assert (tok / "merges.txt").read_text() == "#version: 0.2\na b</w>\na b\n"
    assert json.loads((tok / "special_tokens_map.json").read_text()) == {"bos_token": "<s>", "pad_token": "!"}
