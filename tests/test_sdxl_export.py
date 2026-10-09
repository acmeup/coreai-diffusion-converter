# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""The SDXL export driver on tiny offline components (no Core AI compilation)."""

import asyncio
import copy
import json
import zipfile
from types import SimpleNamespace

import pytest
import torch

from coreai_diffusion_converter import _sdxl_export as X
from coreai_diffusion_converter import cli, exporter
from coreai_diffusion_converter.errors import ExportError
from conftest import write_weights
import tiny_models as T

P = pytest.importorskip("coreai_models.diffusion.pipeline")


def tiny_tokenizer(folder, pad="<|endoftext|>"):
    """A CLIP tokenizer with a character vocabulary and no merges, written to ``folder``."""
    from transformers import CLIPTokenizer

    folder.mkdir(parents=True, exist_ok=True)
    vocab = {"<|startoftext|>": 0, "!": 1, "<|endoftext|>": 2}
    for ch in "abcdefghijklmnopqrstuvwxyz":
        vocab[ch] = len(vocab)
        vocab[f"{ch}</w>"] = len(vocab)
    (folder / "vocab.json").write_text(json.dumps(vocab))
    (folder / "merges.txt").write_text("#version: 0.2\n")
    return CLIPTokenizer(str(folder / "vocab.json"), str(folder / "merges.txt"), pad_token=pad,
                         model_max_length=77)


def test_text_encoder_wrappers_equal_encode_prompt(tmp_path):
    pipe = T.sdxl_pipe()
    tok1 = tiny_tokenizer(tmp_path / "t1")
    tok2 = tiny_tokenizer(tmp_path / "t2", pad="!")
    pipe.tokenizer, pipe.tokenizer_2 = tok1, tok2
    with torch.no_grad():
        embeds, _, pooled, _ = pipe.encode_prompt("a girl", device="cpu", num_images_per_prompt=1,
                                                  do_classifier_free_guidance=False)
        ids1 = tok1("a girl", padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        ids2 = tok2("a girl", padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        h1 = X.SDXLTextEncoderWrapper(pipe.text_encoder)(ids1)
        h2, p2 = X.SDXLTextEncoder2Wrapper(pipe.text_encoder_2)(ids2)
    assert torch.allclose(torch.cat([h1, h2], dim=-1), embeds, atol=1e-6)
    assert torch.allclose(p2, pooled, atol=1e-6)


def test_unet_wrapper_equals_added_cond_kwargs_call():
    pipe = T.sdxl_pipe()
    reference = copy.deepcopy(pipe.unet)
    sample, t, ehs, te, tid = X.dummy_inputs(pipe)["unet"]
    sample, t, ehs, te, tid = (x.float() for x in (sample, t, ehs, te, tid))
    with torch.no_grad():
        got = X.SDXLUNetWrapper(pipe.unet)(sample, t, ehs, te, tid)
        want = reference(sample, t, ehs, added_cond_kwargs={"text_embeds": te, "time_ids": tid}).sample
    assert torch.allclose(got, want, atol=1e-5)


def test_dummy_shapes():
    d = X.dummy_inputs(T.sdxl_pipe())
    assert d["text_encoder"][0].shape == (1, 77) and d["text_encoder"][0].dtype == torch.long
    sample, t, ehs, te, tid = d["unet"]
    assert sample.shape == (2, 4, 8, 8) and t.shape == (2,) and ehs.shape == (2, 77, 64)
    assert te.shape == (2, 32) and tid.shape == (2, 6) and tid[0].tolist() == [64.0, 64.0, 0.0, 0.0, 64.0, 64.0]
    assert d["vae_decoder"][0].dtype == torch.float32 and d["vae_decoder"][0].shape == (1, 4, 8, 8)
    assert X.time_ids(1024) == [1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]


@pytest.fixture
def tiny_tree(tmp_path):
    tree = tmp_path / "tree"
    T.sdxl_pipe().save_pretrained(str(tree), safe_serialization=True)
    return tree


def test_load_pipeline_keeps_the_vae_in_float32(tiny_tree):
    pipe, info = X.load_pipeline(str(tiny_tree), variant=None, sample_size=None, loras=(), tuning={})
    assert next(pipe.vae.parameters()).dtype == torch.float32
    assert next(pipe.unet.parameters()).dtype == torch.float16
    assert info == {"prediction_type": "epsilon"}


def test_load_pipeline_applies_prediction_type_and_sample_size(tiny_tree):
    pipe, info = X.load_pipeline(str(tiny_tree), variant=None, sample_size=16, loras=(),
                                 tuning={"prediction_type": "v_prediction"})
    assert info["prediction_type"] == "v_prediction" and pipe.unet.config.sample_size == 16


class FakeProgram:
    saved: list = []

    def __init__(self, name):
        self.name = name

    def save_asset(self, path, metadata):
        path.mkdir(parents=True)
        (path / "main.mlirb").write_bytes(b"w")
        FakeProgram.saved.append((path.name, metadata))


def _fake_P(monkeypatch, tokenizers=("tokenizer", "tokenizer_2")):
    calls = {"quantized": [], "exported": []}

    def export_stateless(wrapper, dummy, input_names, output_names, **kw):
        calls["exported"].append((type(wrapper).__name__, input_names, output_names,
                                  [getattr(x, "dtype", None) for x in dummy]))
        return FakeProgram(type(wrapper).__name__)

    async def quantize(program, cfg):
        calls["quantized"].append(program.name)
        return program

    def save_tokenizer(model_id, out, pipe, overwrite):
        for sub in tokenizers:
            (out / sub).mkdir(parents=True, exist_ok=True)
            for f in ("vocab.json", "merges.txt"):
                (out / sub / f).write_text("{}")

    def write_metadata(pipe, model_id, pipeline_type, out, compression, results, **kw):
        assert pipeline_type == "sd"
        (out / "metadata.json").write_text(json.dumps({"diffusion": {
            "type": "stable-diffusion", "image_size": 1024, "prediction_type": "epsilon"}}))

    monkeypatch.setattr(P, "export_stateless", export_stateless)
    monkeypatch.setattr(P, "apply_mlir_quantization", quantize)
    monkeypatch.setattr(P, "_save_tokenizer", save_tokenizer)
    monkeypatch.setattr(P, "_write_metadata_json", write_metadata)
    monkeypatch.setattr(P, "build_aimodel_metadata", lambda pack_id, component=None: SimpleNamespace(
        license="Licence X", component=component))
    return calls


def _plan(tmp_path, compression="4bit"):
    return {"pack_id": "xl", "tree": str(tmp_path / "tree"), "out_root": str(tmp_path / "out"),
            "components": ["text_encoder", "text_encoder_2", "unet", "vae_decoder"], "compression": compression,
            "variant": None, "sample_size": None, "loras": [], "tuning": {}}


def test_export_quantizes_text_encoders_and_unet_only(tmp_path, monkeypatch):
    calls = _fake_P(monkeypatch)
    pipe = T.sdxl_pipe()
    monkeypatch.setattr(X, "load_pipeline", lambda *a, **k: (pipe, {"prediction_type": "epsilon"}))
    FakeProgram.saved = []
    result: dict = {}
    asyncio.run(X.export_sdxl(_plan(tmp_path), result))
    assert calls["quantized"] == ["SDXLTextEncoderWrapper", "SDXLTextEncoder2Wrapper", "SDXLUNetWrapper"]
    assert [c[1] for c in calls["exported"]] == [("input_ids",), ("input_ids",), X.UNET_INPUTS, ("z",)]
    assert calls["exported"][1][2] == ("hidden_embeds", "pooled_outputs")
    assert calls["exported"][3][3] == [torch.float32]
    assert [s[0] for s in FakeProgram.saved] == ["TextEncoder.aimodel", "TextEncoder2.aimodel", "Unet.aimodel",
                                                 "VAEDecoder.aimodel"]
    assert all(m.license == "Licence X" for _, m in FakeProgram.saved)  # through the patched attribute
    md = json.loads((tmp_path / "out" / "xl" / "metadata.json").read_text())
    assert md["diffusion"]["type"] == "stable-diffusion-xl"
    assert md["diffusion"]["force_zeros_for_empty_prompt"] is True
    assert result == {"prediction_type": "epsilon", "vae_precision": "float32"}


def test_fp16_export_quantizes_nothing(tmp_path, monkeypatch):
    calls = _fake_P(monkeypatch)
    monkeypatch.setattr(X, "load_pipeline", lambda *a, **k: (T.sdxl_pipe(), {}))
    asyncio.run(X.export_sdxl(_plan(tmp_path, compression="none"), {}))
    assert calls["quantized"] == []


def test_missing_tokenizer_2_is_an_export_error(tmp_path, monkeypatch):
    _fake_P(monkeypatch, tokenizers=("tokenizer",))
    monkeypatch.setattr(X, "load_pipeline", lambda *a, **k: (T.sdxl_pipe(), {}))
    with pytest.raises(ExportError, match="tokenizer was not exported"):
        asyncio.run(X.export_sdxl(_plan(tmp_path), {}))


def test_force_zeros_false_is_kept(tmp_path):
    pipe = T.sdxl_pipe()
    pipe.register_to_config(force_zeros_for_empty_prompt=False)
    (tmp_path / "metadata.json").write_text(json.dumps({"diffusion": {"type": "stable-diffusion"}}))
    md = X.rewrite_metadata(tmp_path, pipe)
    assert md["diffusion"] == {"type": "stable-diffusion-xl", "force_zeros_for_empty_prompt": False}


# --- the CLI records the SDXL prediction type and checks it -------------------------------------


@pytest.fixture
def sdxl_folder(family_tree):
    tree = family_tree("sdxl")
    write_weights(tree, ("text_encoder", "text_encoder_2", "unet", "vae"))
    (tree / "LICENSE").write_text("terms")
    return tree


def _fake_export(prediction_type, metadata_prediction):
    def run(tree, plan, *, pack_id, work_dir, licence_name, verbose=False):
        assert plan.spec.family == "sdxl"
        bundle = work_dir / "export" / pack_id
        for a in ("TextEncoder", "TextEncoder2", "Unet", "VAEDecoder"):
            (bundle / f"{a}.aimodel").mkdir(parents=True)
            (bundle / f"{a}.aimodel" / "main.mlirb").write_bytes(b"w")
        for sub in ("tokenizer", "tokenizer_2"):
            (bundle / sub).mkdir()
            (bundle / sub / "vocab.json").write_text("{}")
        (bundle / "metadata.json").write_text(json.dumps({
            "name": pack_id, "diffusion": {"type": "stable-diffusion-xl", "image_size": 1024,
                                           "prediction_type": metadata_prediction},
            "source": {"hf_model_id": pack_id}, "compilation": {"date": "2026-10-08T08:00:00Z"}}))
        return bundle, {"prediction_type": prediction_type, "vae_precision": "float32"}

    return run


def test_cli_records_and_checks_the_sdxl_prediction_type(sdxl_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "run_export", _fake_export("v_prediction", "v_prediction"))
    out = tmp_path / "out"
    assert cli.main(["convert", str(sdxl_folder), "--target", "macos", "--name", "XL",
                     "--output-dir", str(out)]) == 0
    with zipfile.ZipFile(out / "XL.macos.caipack") as zf:
        p = json.loads(zf.read("pack.json"))
        changes = zf.read("CHANGES.md").decode()
    assert p["family"] == "sdxl" and p["pipeline"] == "sdxl" and p["supported_sizes"] == [1024]
    assert p["conversion"]["prediction_type"] == "v_prediction" and p["conversion"]["vae_precision"] == "float32"
    assert "TextEncoder2.aimodel" in p["assets"] and "tokenizer_2/vocab.json" in p["assets"]
    assert "Prediction type: v_prediction" in changes and "VAE decoder: float32" in changes

    monkeypatch.setattr(exporter, "run_export", _fake_export("v_prediction", "epsilon"))
    assert cli.main(["convert", str(sdxl_folder), "--target", "macos", "--name", "XL2",
                     "--output-dir", str(out)]) == 4
