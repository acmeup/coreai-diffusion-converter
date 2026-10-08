# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Drift guard against the pinned apple/coreai-models commit."""

import dataclasses
import inspect
import subprocess

import pytest

from coreai_diffusion_converter import _export_worker as W
from coreai_diffusion_converter import exporter
from coreai_diffusion_converter.families import FAMILIES, ExportPlan

P = pytest.importorskip("coreai_models.diffusion.pipeline")


def test_pinned_attributes_exist():
    for name in ("export_diffusion", "DiffusionExportConfig", "get_pipeline_type", "snapshot_download",
                 "build_aimodel_metadata", "_load_hf_pipeline"):
        assert hasattr(P, name), name
    fields = {f.name for f in dataclasses.fields(P.DiffusionExportConfig)}
    assert {"hf_model_id", "output_dir", "components", "compute_precision", "compression", "overwrite",
            "multifunction"} <= fields


def test_output_dir_naming_rule():
    src = inspect.getsource(P._async_export_diffusion)
    assert 'model_subdir = config.hf_model_id.split("/")[-1]' in src
    assert "Path(config.output_dir) / model_subdir" in src


def test_components_exist_for_every_family():
    from coreai_models.diffusion.components import get_valid_components

    assert {"text_encoder", "unet", "vae_decoder"} <= set(get_valid_components("sd"))
    assert {"text_encoder", "text_encoder_2", "transformer", "vae_decoder"} <= set(get_valid_components("sd3"))
    assert {"transformer_512", "text_encoder", "vae_decoder_half", "transformer", "vae_decoder"} <= \
        set(get_valid_components("flux2", multifunction=False))


def test_text_encoder_wrapper_returns_last_hidden_state():
    from coreai_models.diffusion.components import TextEncoderWrapper

    assert "last_hidden_state" in inspect.getsource(TextEncoderWrapper.forward)


def test_sd_loader_passes_safety_checker_none():
    assert "safety_checker=None" in inspect.getsource(P._load_hf_pipeline)


def test_worker_gets_offline_env(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, env=None, check=False):
        seen["cmd"], seen["env"] = cmd, env
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)
    plan = ExportPlan(spec=FAMILIES["sd1"], target="ios", size=512, precision="fp16",
                      components=["text_encoder", "unet", "vae_decoder"], multifunction=False, sample_size=None,
                      compression="none", variant=None, vae=None, clip_skip=1, prediction_type=None)
    with pytest.raises(Exception):  # no bundle is produced by the fake run
        exporter.run_export(tmp_path / "tree", plan, pack_id="p", work_dir=tmp_path, licence_name="L")
    assert seen["env"]["HF_HUB_OFFLINE"] == "1"
    assert seen["cmd"][1:3] == ["-m", "coreai_diffusion_converter._export_worker"]


def test_patch_context_managers_restore_originals():
    import diffusers

    originals = {n: getattr(diffusers, n).__dict__.get("from_pretrained") for n in W.PIPELINE_CLASSES}
    bound = {n: getattr(diffusers, n).from_pretrained for n in W.PIPELINE_CLASSES}
    get_type, snap, meta = P.get_pipeline_type, P.snapshot_download, P.build_aimodel_metadata
    with W.patch_attr(P, "get_pipeline_type", lambda _id: "sd"), \
         W.patch_attr(P, "snapshot_download", lambda *a, **k: "x"), \
         W.patch_from_pretrained(pack_id="p", tree="t", variant=None, sample_size=None, family="sd1",
                                 tuning={}, result={}):
        assert P.get_pipeline_type("anything") == "sd"
        assert getattr(diffusers.StableDiffusionPipeline, "from_pretrained") is not bound["StableDiffusionPipeline"]
    assert (P.get_pipeline_type, P.snapshot_download, P.build_aimodel_metadata) == (get_type, snap, meta)
    for n in W.PIPELINE_CLASSES:
        assert getattr(diffusers, n).__dict__.get("from_pretrained") is originals[n]


def test_loader_swaps_pack_id_for_tree_and_applies_sample_size():
    from types import SimpleNamespace

    calls = {}

    class Denoiser:
        config = SimpleNamespace(sample_size=128)

        def register_to_config(self, **kw):
            calls["registered"] = kw

    def original(model_id, **kw):
        calls["id"], calls["kw"] = model_id, kw
        return SimpleNamespace(unet=None, transformer=Denoiser())

    load = W.make_loader(original, pack_id="p", tree="/tree", variant="fp16", sample_size=64, family="sd3",
                         tuning={}, result={})
    load("p", torch_dtype="f16")
    assert calls["id"] == "/tree" and calls["kw"]["variant"] == "fp16"
    assert "feature_extractor" not in calls["kw"] and calls["registered"] == {"sample_size": 64}
    load_sd = W.make_loader(original, pack_id="p", tree="/tree", variant=None, sample_size=None, family="sd1",
                            tuning={"clip_skip": 1}, result={})
    try:
        load_sd("p", safety_checker=None)
    except AttributeError:
        pass  # the fake pipe has no text encoder; only the kwargs matter here
    assert calls["kw"]["feature_extractor"] is None and calls["kw"]["safety_checker"] is None


def test_wrap_metadata_sets_licence():
    from types import SimpleNamespace

    def original(hf_model_id, component=None):
        return SimpleNamespace(license="", author="")

    md = W.wrap_metadata(original, "CreativeML OpenRAIL-M")("p", component="Unet")
    assert md.license == "CreativeML OpenRAIL-M" and md.author == ""
