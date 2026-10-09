# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from coreai_diffusion_converter import cli, exporter
from conftest import PACKS, write_weights


def never_export(*a, **k):
    raise AssertionError("the exporter must not run")


@pytest.fixture
def sd1_folder(family_tree):
    tree = family_tree("sd1")
    write_weights(tree, ("text_encoder", "unet", "vae"))
    (tree / "LICENSE").write_text("terms")
    return tree


def test_dry_run_never_exports(sd1_folder, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(exporter, "run_export", never_export)
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--dry-run",
                     "--output-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "family:          sd1" in out and "tree-sd1.ios.caipack" in out
    assert not (tmp_path / ".caipack-work").exists()


def test_bad_name_fails_before_export(sd1_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "run_export", never_export)
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--name", "x" * 81,
                     "--output-dir", str(tmp_path)]) == 2
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--steps", "0",
                     "--output-dir", str(tmp_path)]) == 2
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--steps", "51",
                     "--output-dir", str(tmp_path)]) == 2
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--guidance", "31",
                     "--output-dir", str(tmp_path)]) == 2


def test_usage_errors(sd1_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "run_export", never_export)
    with pytest.raises(SystemExit) as err:
        cli.main(["convert", str(sd1_folder)])
    assert err.value.code == 2
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--size", "768",
                     "--output-dir", str(tmp_path)]) == 2
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--clip-skip", "5",
                     "--output-dir", str(tmp_path)]) == 2


def test_missing_licence_exit_2(family_tree, tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "run_export", never_export)
    tree = family_tree("sd1")
    assert cli.main(["convert", str(tree), "--target", "ios", "--output-dir", str(tmp_path)]) == 2
    assert cli.main(["convert", str(tree), "--target", "ios", "--output-dir", str(tmp_path),
                     "--allow-missing-license", "--dry-run"]) == 0
    # A dry run reports the missing licence as a warning instead of stopping.
    assert cli.main(["convert", str(tree), "--target", "ios", "--output-dir", str(tmp_path), "--dry-run"]) == 0


def test_unsupported_exit_3(family_tree, tmp_path):
    tree = family_tree("sd1", model_index__json={"_class_name": "StableDiffusionXLPipeline"})
    assert cli.main(["convert", str(tree), "--target", "ios", "--output-dir", str(tmp_path)]) == 3
    ckpt = tmp_path / "m.ckpt"
    ckpt.write_bytes(b"x")
    assert cli.main(["convert", str(ckpt), "--target", "ios", "--output-dir", str(tmp_path)]) == 3


def test_overwrite_guard(sd1_folder, tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "run_export", never_export)
    (tmp_path / "My Pack.ios.caipack").write_bytes(b"old")
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--name", "My Pack",
                     "--output-dir", str(tmp_path)]) == 2
    assert (tmp_path / "My Pack.ios.caipack").read_bytes() == b"old"


def test_full_convert_with_a_fake_export(sd1_folder, tmp_path, monkeypatch):
    """The whole convert flow with the export replaced by a tiny bundle; checks naming, privacy,
    pack.json and self-validation."""

    def fake_export(tree, plan, *, pack_id, work_dir, licence_name, verbose=False):
        bundle = work_dir / "export" / pack_id
        for a in ("TextEncoder", "Unet", "VAEDecoder"):
            (bundle / f"{a}.aimodel").mkdir(parents=True)
            (bundle / f"{a}.aimodel" / "main.mlirb").write_bytes(b"w")
            (bundle / f"{a}.aimodel" / "main.hash").write_text("h")
            (bundle / f"{a}.aimodel" / "metadata.json").write_text("{}")
        (bundle / "tokenizer").mkdir()
        (bundle / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({"name_or_path": str(tree)}))
        (bundle / "metadata.json").write_text(json.dumps({
            "name": pack_id, "diffusion": {"type": "stable-diffusion", "image_size": 512, "prediction_type": "epsilon"},
            "source": {"hf_model_id": pack_id}, "compilation": {"date": "2026-10-08T08:00:00-04:00"}}))
        return bundle, {"prediction_type": "epsilon"}

    monkeypatch.setattr(exporter, "run_export", fake_export)
    out_dir = tmp_path / "out"
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--name", "Unit Model",
                     "--output-dir", str(out_dir)]) == 0
    pack = out_dir / "Unit Model.ios.caipack"
    assert pack.is_file() and not (out_dir / ".caipack-work").exists()
    assert cli.main(["validate", str(pack)]) == 0
    assert cli.main(["inspect", str(pack)]) == 0
    import zipfile

    with zipfile.ZipFile(pack) as zf:
        p = json.loads(zf.read("pack.json"))
        md = json.loads(zf.read("metadata.json"))
        tok = json.loads(zf.read("tokenizer/tokenizer_config.json"))
    assert p["id"] == "unit-model" and p["source"] == {"kind": "folder", "ref": sd1_folder.name, "revision": None}
    assert md["source"]["hf_model_id"] == sd1_folder.name and md["compilation"]["date"].endswith("Z")
    assert "name_or_path" not in tok


def test_validate_exit_codes(tmp_path):
    assert cli.main(["validate", str(PACKS / "zip64_tiny.caipack")]) == 0
    bad = tmp_path / "bad.caipack"
    bad.write_text("x")
    assert cli.main(["validate", str(bad)]) == 5


# --- LoRAs, Civitai and the worker plan (M4) ----------------------------------------------------


def _v1_file(tmp_path):
    import torch
    from safetensors.torch import save_file

    p = tmp_path / "single.safetensors"
    save_file({"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(8, 4, 3, 3)}, str(p))
    return p


def _sd1_lora(folder, name="style.safetensors"):
    import tiny_models as T

    folder.mkdir(parents=True, exist_ok=True)
    return T.save(T.kohya("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k", 768, 320),
                  folder / name)


def _fake_worker(monkeypatch, changed):
    """Replace only the subprocess: run_export itself writes plan.json for real."""
    from coreai_diffusion_converter import exporter

    seen = {}

    def run(cmd, env=None, check=False):
        import subprocess
        from pathlib import Path

        plan = json.loads(Path(cmd[3]).read_text())
        seen["plan"] = plan
        bundle = Path(plan["out_root"]) / plan["pack_id"]
        for a in ("TextEncoder", "Unet", "VAEDecoder"):
            (bundle / f"{a}.aimodel").mkdir(parents=True)
            (bundle / f"{a}.aimodel" / "main.mlirb").write_bytes(b"w")
        (bundle / "tokenizer").mkdir()
        (bundle / "tokenizer" / "vocab.json").write_text("{}")
        (bundle / "metadata.json").write_text(json.dumps({
            "name": plan["pack_id"], "diffusion": {"type": "stable-diffusion", "image_size": 512,
                                                   "prediction_type": "epsilon"},
            "source": {"hf_model_id": plan["pack_id"]}, "compilation": {"date": "2026-10-08T08:00:00Z"}}))
        Path(plan["result_path"]).write_text(json.dumps({
            "prediction_type": "epsilon",
            "loras": [{"name": lo["name"], "changed": {"unet": changed}} for lo in plan["loras"]]}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(exporter.subprocess, "run", run)
    return seen


def _fake_resolve(monkeypatch, sd1_folder, kind, order):
    from coreai_diffusion_converter import sources

    def resolve(probe, **kw):
        order.append("checkpoint")
        return sources.ResolvedSource(tree=sd1_folder, kind=kind, ref=probe.ref, revision=None, licence_dir=None,
                                      card_license=(None, None), prediction_type="epsilon")

    monkeypatch.setattr(sources, "resolve_source", resolve)


def test_single_file_plan_json_carries_the_lora(sd1_folder, tmp_path, monkeypatch, capsys):
    order = []
    _fake_resolve(monkeypatch, sd1_folder, "single_file", order)
    seen = _fake_worker(monkeypatch, changed=12)
    lora_path = _sd1_lora(tmp_path / "nested" / "deeper")
    out = tmp_path / "out"
    rc = cli.main(["convert", str(_v1_file(tmp_path)), "--target", "ios", "--name", "With Lora",
                   "--lora", f"{lora_path}:0.75", "--license-file", str(sd1_folder / "LICENSE"),
                   "--output-dir", str(out), "--keep-work", "--work-dir", str(tmp_path / "w")])
    assert rc == 0
    assert seen["plan"]["loras"] == [{"path": str(lora_path), "scale": 0.75, "name": "style.safetensors"}]
    import zipfile

    with zipfile.ZipFile(out / "With Lora.ios.caipack") as zf:
        p = json.loads(zf.read("pack.json"))
        changes = zf.read("CHANGES.md").decode()
    lo = p["conversion"]["loras"][0]
    assert lo["name"] == "style.safetensors" and lo["scale"] == 0.75 and len(lo["sha256"]) == 64
    assert lo["source"] == {"kind": "file", "ref": "style.safetensors", "revision": None}
    for text in (json.dumps(p), changes):
        assert str(tmp_path) not in text
    assert "LoRA merged: style.safetensors (sha256 " in changes and "fused into the model before export" in changes


def test_zero_changed_modules_abort_before_the_pack(sd1_folder, tmp_path, monkeypatch):
    _fake_resolve(monkeypatch, sd1_folder, "single_file", [])
    _fake_worker(monkeypatch, changed=0)
    out = tmp_path / "out"
    rc = cli.main(["convert", str(_v1_file(tmp_path)), "--target", "ios", "--name", "Zero",
                   "--lora", str(_sd1_lora(tmp_path / "l")), "--license-file", str(sd1_folder / "LICENSE"),
                   "--output-dir", str(out)])
    assert rc == 4 and not (out / "Zero.ios.caipack").exists()


def test_civitai_order_name_notice_and_trigger_words(sd1_folder, tmp_path, monkeypatch, capsys, caplog):
    import logging

    import httpx

    from coreai_diffusion_converter import civitai, sources
    import tiny_models as T
    from test_civitai import FakeCivitai

    real = civitai.CivitaiClient
    monkeypatch.setattr(civitai, "CivitaiClient", lambda token, **kw: real(
        token, transport=httpx.MockTransport(FakeCivitai()), sleep=lambda s: None))
    order = []
    _fake_resolve(monkeypatch, sd1_folder, "civitai", order)
    seen = _fake_worker(monkeypatch, changed=3)
    sdxl_lora = T.save(T.kohya("lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_k", 2048, 1280),
                       tmp_path / "92996-PerfectEyesXL.safetensors")

    def fetch_lora(spec, family, **kw):
        order.append(f"lora {spec.source.ref}")
        spec.path = __import__("pathlib").Path(sdxl_lora)
        spec.sha256 = "a" * 64
        return spec

    monkeypatch.setattr(sources, "fetch_lora", fetch_lora)
    monkeypatch.setattr(sources, "fetch_vae", lambda ref, **kw: order.append("vae") or ref)
    # The pack's family is sdxl here, so the fake worker's sd1 bundle would fail rule 11: check
    # everything up to the export instead.
    import coreai_diffusion_converter.cli as C

    monkeypatch.setattr(C, "_finish_bundle", lambda *a, **k: (_ for _ in ()).throw(SystemExit("stop")))
    caplog.set_level(logging.WARNING)
    with pytest.raises(SystemExit):
        cli.main(["convert", "civitai:1188071@1408658", "--target", "macos",
                  "--lora", "civitai:118427@128461:0.8", "--lora", "civitai:120096@135931",
                  "--license-file", str(sd1_folder / "LICENSE"), "--output-dir", str(tmp_path / "out"),
                  "--cache-dir", str(tmp_path / "cache")])
    assert order == ["lora 118427@128461", "lora 120096@135931", "vae", "checkpoint"]
    assert seen["plan"]["pack_id"] == "animagine-xl-4-0-v4-opt"  # the name came from the probe
    assert [lo["name"] for lo in seen["plan"]["loras"]] == ["PerfectEyesXL.safetensors", "pixel-art-xl-v1.1.safetensors"]
    assert "pixel-art-xl-v1.1.safetensors: the creator does not allow derivatives" in caplog.text


def test_trigger_words_and_notice_in_a_full_civitai_lora_run(sd1_folder, tmp_path, monkeypatch, capsys):
    """An SD 1.x folder + a Civitai LoRA end to end (download faked): NOTICE carries the Civitai
    permissions, pack.json the LoRA record, and the trigger words follow 'validation passed'."""
    import httpx

    from coreai_diffusion_converter import civitai, sources
    from test_civitai import FakeCivitai

    real = civitai.CivitaiClient
    monkeypatch.setattr(civitai, "CivitaiClient", lambda token, **kw: real(
        token, transport=httpx.MockTransport(FakeCivitai()), sleep=lambda s: None))
    _fake_worker(monkeypatch, changed=5)
    lora_file = _sd1_lora(tmp_path / "cache")

    def fetch_lora(spec, family, **kw):
        spec.path = __import__("pathlib").Path(lora_file)
        spec.sha256 = "b" * 64
        return spec

    monkeypatch.setattr(sources, "fetch_lora", fetch_lora)
    out = tmp_path / "out"
    monkeypatch.setenv("CIVITAI_API_TOKEN", "tok-SECRET-0123456789")
    rc = cli.main(["convert", str(sd1_folder), "--target", "ios", "--name", "Styled",
                   "--lora", "civitai:900106@900006:0.6", "--output-dir", str(out),
                   "--cache-dir", str(tmp_path / "cache")])
    assert rc == 0
    stdout = capsys.readouterr().out
    assert stdout.index("validation passed") < stdout.index("trigger words (PerfectEyesXL.safetensors): green eyes")
    import zipfile

    with zipfile.ZipFile(out / "Styled.ios.caipack") as zf:
        p = json.loads(zf.read("pack.json"))
        notice = zf.read("NOTICE").decode()
        texts = [zf.read(n).decode("utf-8", "replace") for n in zf.namelist() if not n.endswith(".mlirb")]
    assert p["license"]["notice_file"] == "NOTICE"
    assert "Civitai permissions for Example LORA 900106 sd15 lora (https://civitai.com/models/900106?modelVersionId=900006), by example-creator:" in notice
    lo = p["conversion"]["loras"][0]
    assert lo["source"] == {"kind": "civitai", "ref": "900106@900006", "revision": None}
    assert lo["permissions"]["allow_derivatives"] is True and lo["trained_words"][0] == "green eyes"
    assert all("tok-SECRET" not in t and str(tmp_path) not in t for t in texts)
