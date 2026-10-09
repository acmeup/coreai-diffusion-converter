# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Civitai client, file choice, refusals and downloads, all against httpx.MockTransport."""

import dataclasses
import fcntl
import hashlib
import json
import logging
import re
from collections import namedtuple
from pathlib import Path

import httpx
import pytest

from coreai_diffusion_converter import cli, civitai, exporter, sources
from coreai_diffusion_converter.errors import DownloadError, UnsupportedModelError, UsageError
from conftest import FIXTURES, write_weights

CIV = FIXTURES / "civitai"
TOKEN = "tok-SECRET-0123456789"
PRESIGNED_QUERY = "Authorization=PRESIGNED-SECRET-abcdef"
Call = namedtuple("Call", "method url headers")


def load(name):
    return json.loads((CIV / name).read_text())


class FakeCivitai:
    """Serves the fixtures under /api/v1, answers downloads with a 307 to a presigned b2 URL, and
    serves ``payload`` there with Range support."""

    def __init__(self, payload=b"", *, location_host="b2.civitai.com", b2=None, api_status=None):
        self.payload = payload
        self.calls: list[Call] = []
        self.location_host = location_host
        self.b2 = b2                    # optional override: callable(request) -> Response
        self.api_status = api_status or {}  # path -> list of statuses to return first

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(Call(request.method, str(request.url), dict(request.headers)))
        url = request.url
        if url.host == "civitai.com" and url.path.startswith("/api/v1/"):
            queue = self.api_status.get(url.path)
            if queue:
                return httpx.Response(queue.pop(0))
            m = re.match(r"^/api/v1/(models|model-versions)/(\d+)$", url.path)
            name = f"{'model' if m.group(1) == 'models' else 'version'}_{m.group(2)}.json"
            if not (CIV / name).is_file():
                return httpx.Response(404)
            return httpx.Response(200, json=load(name))
        if url.host == "civitai.com" and url.path.startswith("/api/download/models/"):
            queue = self.api_status.get(url.path)
            if queue:
                return httpx.Response(queue.pop(0))
            return httpx.Response(307, headers={"location": f"https://{self.location_host}/file/x.safetensors?"
                                                            f"{PRESIGNED_QUERY}"})
        if self.b2 is not None:
            return self.b2(request)
        rng = request.headers.get("range")
        total = len(self.payload)
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= total:
                return httpx.Response(416)
            return httpx.Response(206, content=self.payload[start:], headers={
                "content-type": "application/octet-stream", "content-range": f"bytes {start}-{total - 1}/{total}"})
        return httpx.Response(200, content=self.payload, headers={"content-type": "application/octet-stream",
                                                                  "content-length": str(total)})

    def paths(self):
        return [httpx.URL(c.url).path for c in self.calls]


def client(fake, token=TOKEN):
    return civitai.CivitaiClient(token, transport=httpx.MockTransport(fake), sleep=lambda s: None)


def small(version, payload, sha=None):
    """The version with its file shrunk to ``payload`` (fixtures list multi-GB files)."""
    digest = hashlib.sha256(payload).hexdigest() if sha is None else sha
    return dataclasses.replace(version, file=dataclasses.replace(version.file, size_bytes=len(payload), sha256=digest))


# --- references ---------------------------------------------------------------------------------


@pytest.mark.parametrize("ref,expected", [
    ("civitai:1188071", (1188071, None)),
    ("civitai:1188071@1408658", (1188071, 1408658)),
    ("https://civitai.com/models/1188071", (1188071, None)),
    ("https://civitai.com/models/1188071/animagine-xl", (1188071, None)),
    ("https://www.civitai.com/models/118427/perfect-eyes-xl?modelVersionId=128461", (118427, 128461)),
    ("https://civitai.com/api/download/models/128461", (None, 128461)),
])
def test_reference_forms(ref, expected):
    assert civitai.is_civitai_ref(ref)
    r = civitai.parse_ref(ref)
    assert (r.model_id, r.version_id) == expected


@pytest.mark.parametrize("ref", ["civitai:", "civitai:abc", "civitai:1@x", "https://civitai.com/images/5",
                                 "https://civitai.com/models/1?modelVersionId=x", "https://civitai.com/user/x"])
def test_junk_references_refused(ref):
    with pytest.raises(UsageError):
        civitai.parse_ref(ref)


def test_non_civitai_strings_are_not_refs():
    assert not civitai.is_civitai_ref("org/repo")
    assert not civitai.is_civitai_ref("https://huggingface.co/org/repo")
    assert not civitai.is_civitai_ref("./model.safetensors")


# --- metadata -----------------------------------------------------------------------------------


def test_version_and_model_are_merged():
    fake = FakeCivitai()
    v = client(fake).resolve(civitai.CivitaiRef(118427, 128461), "lora")
    assert fake.paths() == ["/api/v1/model-versions/128461", "/api/v1/models/118427"]
    assert v.trained_words == ("green eyes", "blue eyes", "brown eyes", "perfecteyes")
    assert v.permissions.allow_derivatives and v.creator == "Deizor" and v.model_type == "LORA"
    assert v.file.name == "PerfectEyesXL.safetensors" and v.file.sha256 == v.file.sha256.lower()
    assert v.family == "sdxl" and v.ref == "118427@128461"


def test_version_only_reference_fetches_the_version_once():
    fake = FakeCivitai()
    v = client(fake).resolve(civitai.parse_ref("https://civitai.com/api/download/models/1408658"), "model")
    assert fake.paths().count("/api/v1/model-versions/1408658") == 1
    assert v.model_id == 1188071 and v.file.name == "animagineXL40_v4Opt.safetensors"
    assert v.file.sha256 == "6327eca98bfb6538dd7a4edce22484a1bbc57a8cff6b11d075d40da1afb847ac"
    assert v.display_name == "Animagine XL 4.0 v4 Opt"


def test_default_version_is_the_creators_first_published(capsys):
    fake = FakeCivitai()
    v = client(fake).resolve(civitai.CivitaiRef(118427, None), "lora")
    assert v.version_id == 128461  # listed first although 842340 is newer
    assert "creator's default version 128461" in capsys.readouterr().out
    v = client(FakeCivitai()).resolve(civitai.CivitaiRef(900110, None), "lora")
    assert v.version_id == 900011  # the first entry is a draft


def test_unpublished_and_generation_only_versions_refused():
    with pytest.raises(UnsupportedModelError, match="not published"):
        client(FakeCivitai()).resolve(civitai.CivitaiRef(900110, 900010), "lora")
    with pytest.raises(UnsupportedModelError, match="Generation access only"):
        client(FakeCivitai()).resolve(civitai.CivitaiRef(900107, 900007), "model")


def test_sparse_metadata_reads_without_key_errors():
    v = client(FakeCivitai()).resolve(civitai.CivitaiRef(120096, 135931), "lora")
    assert v.file.fp is None and v.file.size_kind is None
    assert not v.permissions.allow_derivatives and v.permissions.allow_commercial_use == ()


def test_missing_sha256_refused_unless_allowed(tmp_path):
    payload = b"lora-bytes" * 10
    fake = FakeCivitai(payload)
    c = client(fake)
    v = c.resolve(civitai.CivitaiRef(900108, 900008), "lora")
    assert v.file.sha256 == ""
    v = small(v, payload, sha="")
    with pytest.raises(UnsupportedModelError, match="allow-unverified-download"):
        c.download(v, tmp_path)
    assert not any("/api/download/" in c_.url for c_ in fake.calls)
    path = c.download(v, tmp_path, allow_unverified=True)
    assert path.read_bytes() == payload


def test_unverified_lora_is_recorded(tmp_path, monkeypatch):
    payload = b"x" * 64
    fake = FakeCivitai(payload)
    c = client(fake)
    spec = sources.probe_lora("civitai:900108@900008", 1.0, client=c)
    assert spec.verified is False
    spec.civitai = small(spec.civitai, payload, sha="")
    monkeypatch.setattr(sources.loramod, "classify_lora_keys", lambda keys: "kohya")
    monkeypatch.setattr(sources.loramod, "header_shapes", lambda p: {})
    sources.fetch_lora(spec, "sdxl", client=c, cache_dir=tmp_path, allow_unverified=True)
    assert spec.pack_json()["verified"] is False and spec.sha256 == hashlib.sha256(payload).hexdigest()


def test_locon_passes_metadata_then_real_lycoris_is_refused(tmp_path):
    from coreai_diffusion_converter import lora
    import tiny_models as T
    import torch

    real = tmp_path / "src.safetensors"
    T.save({"lora_unet_x.lora_down.weight": torch.zeros(4, 8), "lora_unet_x.lora_up.weight": torch.zeros(8, 4),
            "lora_unet_x.lora_mid.weight": torch.zeros(4, 4, 3, 3)}, real)
    payload = real.read_bytes()
    c = client(FakeCivitai(payload))
    spec = sources.probe_lora("civitai:900105@900005", 1.0, client=c)
    assert spec.civitai.model_type == "LoCon"
    spec.civitai = small(spec.civitai, payload)
    with pytest.raises(UnsupportedModelError, match=re.escape(lora.LYCORIS_MESSAGE)):
        sources.fetch_lora(spec, "sdxl", client=c, cache_dir=tmp_path)


def test_file_choice():
    mixed = load("version_900012.json")["files"]
    assert civitai.choose_file(mixed, "model").file_id == 2  # fp16 pruned beats primary fp32
    no_pruned = [f for f in mixed if f["id"] != 2]
    assert civitai.choose_file(no_pruned, "model").file_id == 6  # then fp16 full
    only_fp32 = [f for f in mixed if f["id"] == 1]
    assert civitai.choose_file(only_fp32, "model").file_id == 1
    for name in ("version_900001.json", "version_900002.json"):  # pickle only, int8 only
        with pytest.raises(UnsupportedModelError, match="no .safetensors file"):
            civitai.choose_file(load(name)["files"], "model")
    training = [dict(mixed[4])]
    with pytest.raises(UnsupportedModelError):
        civitai.choose_file(training, "model")


@pytest.mark.parametrize("model_type,slot,ok,message", [
    ("Checkpoint", "model", True, ""), ("Checkpoint", "lora", False, "pass it as SOURCE"),
    ("Checkpoint", "vae", False, "pass it as SOURCE"),
    ("LORA", "model", False, "pass it with --lora"), ("LORA", "lora", True, ""), ("LORA", "vae", False, "not a VAE"),
    ("VAE", "model", False, "pass it with --vae"), ("VAE", "lora", False, "not a LoRA"), ("VAE", "vae", True, ""),
    ("LoCon", "model", False, "pass it with --lora"), ("LoCon", "lora", True, ""), ("LoCon", "vae", False, "not a VAE"),
    ("DoRA", "lora", False, "DoRA"), ("TextualInversion", "lora", False, "TextualInversion models are not supported"),
    ("Hypernetwork", "model", False, "not supported"), ("Controlnet", "model", False, "not supported"),
])
def test_slot_matrix(model_type, slot, ok, message):
    if ok:
        civitai.check_slot(model_type, "Standard", slot)
    else:
        with pytest.raises(UnsupportedModelError, match=message):
            civitai.check_slot(model_type, "Standard", slot)


@pytest.mark.parametrize("bmt,message", [("Inpainting", "inpainting"), ("Refiner", "refiner"),
                                         ("Pix2Pix", "image-to-image")])
def test_base_model_type_refusals(bmt, message):
    with pytest.raises(UnsupportedModelError, match=message):
        civitai.check_slot("Checkpoint", bmt, "model")


def test_inpainting_fixture_refused_from_metadata():
    fake = FakeCivitai()
    with pytest.raises(UnsupportedModelError, match="inpainting"):
        client(fake).resolve(civitai.CivitaiRef(900103, 900003), "model")
    assert not any("/download/" in c.url for c in fake.calls)


@pytest.mark.parametrize("base,family", [
    ("SD 1.4", "sd1"), ("SD 1.5", "sd1"), ("SD 2.0", "sd2"), ("SD 2.0 768", "sd2"), ("SD 2.1", "sd2"),
    ("SD 2.1 768", "sd2"), ("SDXL 1.0", "sdxl"), ("Pony", "sdxl"), ("Illustrious", "sdxl"), ("NoobAI", "sdxl"),
    ("SD 3", "sd3"), ("SD 3.5", "sd3"), ("SD 3.5 Medium", "sd3"), ("SD 3.5 Large", "sd3"),
    ("Flux.2 Klein 4B", "flux2"), ("Flux.2 Klein 4B-base", "flux2"),
])
def test_base_model_family(base, family):
    assert civitai.family_for_base_model(base) == family


@pytest.mark.parametrize("base,message", [
    ("SDXL Lightning", "distilled"), ("SD 1.5 LCM", "distilled"), ("SDXL Turbo", "distilled"),
    ("SD 3.5 Large Turbo", "distilled"), ("Flux.2 Klein 9B", "only FLUX.2 Klein 4B"),
    ("Flux.2 Klein 9B-base", "only FLUX.2 Klein 4B"), ("Pony V7", "not supported"), ("Flux.1 D", "not supported"),
    ("SDXL 0.9", "not supported"), ("Playground v2", "not supported"), ("Something New", "not supported"),
])
def test_base_model_refusals(base, message):
    with pytest.raises(UnsupportedModelError, match=message):
        civitai.family_for_base_model(base)


def test_klein_9b_refused_before_any_download():
    fake = FakeCivitai()
    with pytest.raises(UnsupportedModelError, match="9B"):
        sources.probe_civitai("civitai:900104@900004", client(fake), "macos")
    assert not any("/download/" in c.url for c in fake.calls)


def test_sdxl_and_sd35_large_on_ios_refused_from_metadata(monkeypatch):
    with pytest.raises(UnsupportedModelError, match="Mac-only"):
        sources.probe_civitai("civitai:1188071@1408658", client(FakeCivitai()), "ios")
    original = civitai.CivitaiClient.resolve

    def large(self, ref, slot):
        v = original(self, ref, slot)
        return dataclasses.replace(v, base_model="SD 3.5 Large")

    monkeypatch.setattr(civitai.CivitaiClient, "resolve", large)
    with pytest.raises(UnsupportedModelError, match="Large"):
        sources.probe_civitai("civitai:1188071@1408658", client(FakeCivitai()), "ios")


# --- requests, status codes, secrets ------------------------------------------------------------


def test_token_goes_to_civitai_only(tmp_path):
    payload = b"m" * 100
    fake = FakeCivitai(payload)
    c = client(fake)
    v = small(c.resolve(civitai.CivitaiRef(118427, 128461), "lora"), payload)
    c.download(v, tmp_path)
    for call in fake.calls:
        host = httpx.URL(call.url).host
        if host == "civitai.com":
            assert call.headers.get("authorization") == f"Bearer {TOKEN}"
        else:
            assert host == "b2.civitai.com" and "authorization" not in call.headers
    assert any("fileId=92996" in c_.url for c_ in fake.calls)
    assert all(c_.headers["user-agent"].startswith("coreai-diffusion-converter/") for c_ in fake.calls)


def test_no_token_sends_no_authorization():
    fake = FakeCivitai()
    client(fake, token=None).resolve(civitai.CivitaiRef(118427, 128461), "lora")
    assert all("authorization" not in c.headers for c in fake.calls)


def test_redirect_to_civitai_r2_storage_is_accepted(tmp_path):
    payload = b"r2" * 50
    fake = FakeCivitai(payload, location_host="civitai-delivery-worker-prod.0123abcd.r2.cloudflarestorage.com")
    c = client(fake)
    v = small(c.resolve(civitai.CivitaiRef(118427, 128461), "lora"), payload)
    assert c.download(v, tmp_path).read_bytes() == payload
    r2 = [x for x in fake.calls if "cloudflarestorage" in x.url]
    assert r2 and all("authorization" not in x.headers for x in r2)


def test_redirect_to_another_host_refused(tmp_path):
    payload = b"m" * 10
    fake = FakeCivitai(payload, location_host="evil.example.com")
    c = client(fake)
    v = small(c.resolve(civitai.CivitaiRef(118427, 128461), "lora"), payload)
    with pytest.raises(DownloadError, match="unexpected download host evil.example.com"):
        c.download(v, tmp_path)
    assert not any("evil" in c_.url for c_ in fake.calls)


@pytest.mark.parametrize("status,token,message", [
    (401, None, "needs a Civitai API token"), (401, TOKEN, "rejected the API token"),
    (403, TOKEN, "early access or not available to your account"), (403, None, "needs a Civitai API token"),
    (404, TOKEN, "not found"),
])
def test_status_messages(status, token, message):
    fake = FakeCivitai(api_status={"/api/v1/model-versions/128461": [status]})
    with pytest.raises(UsageError, match=message):
        client(fake, token).resolve(civitai.CivitaiRef(118427, 128461), "lora")


def test_429_is_retried():
    slept = []
    fake = FakeCivitai(api_status={"/api/v1/model-versions/128461": [429, 503]})
    c = civitai.CivitaiClient(TOKEN, transport=httpx.MockTransport(fake), sleep=slept.append)
    assert c.resolve(civitai.CivitaiRef(118427, 128461), "lora").version_id == 128461
    assert slept == [2.0, 4.0]


# --- downloads ----------------------------------------------------------------------------------


def _v(c, payload):
    return small(c.resolve(civitai.CivitaiRef(118427, 128461), "lora"), payload)


def test_cache_layout_and_reuse_without_requests(tmp_path):
    payload = b"abc" * 100
    fake = FakeCivitai(payload)
    c = client(fake)
    v = _v(c, payload)
    path = c.download(v, tmp_path)
    assert path == tmp_path / "civitai" / "128461" / "92996-PerfectEyesXL.safetensors"
    assert not list(path.parent.glob("*.partial*"))
    n = len(fake.calls)
    assert c.download(v, tmp_path) == path
    assert len(fake.calls) == n


def test_resume_with_range_206(tmp_path, capsys):
    payload = bytes(range(256)) * 40
    fake = FakeCivitai(payload)
    c = client(fake)
    v = _v(c, payload)
    final = c.cached_path(v, tmp_path)
    final.parent.mkdir(parents=True)
    final.with_name(final.name + ".partial").write_bytes(payload[:1000])
    assert c.download(v, tmp_path).read_bytes() == payload
    assert fake.calls[-1].headers["range"] == "bytes=1000-"
    assert "resuming at 1000 bytes" in capsys.readouterr().out


def test_resume_answered_with_200_restarts(tmp_path):
    payload = b"z" * 3000

    def b2(request):
        return httpx.Response(200, content=payload, headers={"content-type": "application/octet-stream",
                                                             "content-length": str(len(payload))})

    fake = FakeCivitai(payload, b2=b2)
    c = client(fake)
    v = _v(c, payload)
    final = c.cached_path(v, tmp_path)
    final.parent.mkdir(parents=True)
    final.with_name(final.name + ".partial").write_bytes(b"garbage")
    assert c.download(v, tmp_path).read_bytes() == payload


def test_resume_416_means_complete(tmp_path):
    payload = b"q" * 500
    c = client(FakeCivitai(payload))
    v = _v(c, payload)
    final = c.cached_path(v, tmp_path)
    final.parent.mkdir(parents=True)
    final.with_name(final.name + ".partial").write_bytes(payload)
    assert c.download(v, tmp_path).read_bytes() == payload


def test_sha_mismatch_discards_the_partial(tmp_path):
    payload = b"p" * 200
    c = client(FakeCivitai(payload))
    v = small(_v(c, payload), payload, sha="0" * 64)
    with pytest.raises(DownloadError, match="SHA-256 does not match") as err:
        c.download(v, tmp_path)
    assert err.value.exit_code == 6
    assert not list((tmp_path / "civitai").rglob("*.partial"))


@pytest.mark.parametrize("response", [
    httpx.Response(200, content=b"<html>error</html>", headers={"content-type": "text/html"}),
    httpx.Response(200, content=b'{"error": 1}', headers={"content-type": "application/json"}),
    httpx.Response(206, content=b"x" * 10, headers={"content-type": "application/octet-stream",
                                                    "content-range": "bytes 0-9/99999999"}),
    httpx.Response(200, content=b"x" * 10, headers={"content-type": "application/octet-stream",
                                                    "content-length": "10"}),
])
def test_wrong_answer_refused_before_writing(tmp_path, response):
    payload = b"r" * 5000
    c = client(FakeCivitai(payload, b2=lambda request: response))
    v = _v(c, payload)
    with pytest.raises(DownloadError, match="did not return the model file"):
        c.download(v, tmp_path)
    partial = c.cached_path(v, tmp_path).with_name("92996-PerfectEyesXL.safetensors.partial")
    assert not partial.exists() or partial.stat().st_size == 0


def test_held_lock_refuses_a_second_run(tmp_path):
    payload = b"l" * 100
    c = client(FakeCivitai(payload))
    v = _v(c, payload)
    final = c.cached_path(v, tmp_path)
    final.parent.mkdir(parents=True)
    with open(final.with_name(final.name + ".partial.lock"), "a+b") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        with pytest.raises(UsageError, match="being downloaded by another caipack run"):
            c.download(v, tmp_path)


def test_cache_dir_is_created_before_the_free_space_check(tmp_path, monkeypatch):
    payload = b"f" * 100
    c = client(FakeCivitai(payload))
    v = _v(c, payload)
    seen = []
    Usage = namedtuple("Usage", "total used free")

    def disk_usage(path):
        seen.append(Path(path).is_dir())
        return Usage(1, 1, 10)

    monkeypatch.setattr(civitai.shutil, "disk_usage", disk_usage)
    with pytest.raises(UsageError, match="not enough free disk space"):
        c.download(v, tmp_path / "fresh")
    assert seen == [True]


def test_transport_errors_become_redacted_download_errors(tmp_path):
    payload = b"t" * 100

    def b2(request):
        raise httpx.ReadTimeout("timed out reading " + str(request.url), request=request)

    c = client(FakeCivitai(payload, b2=b2))
    v = _v(c, payload)
    with pytest.raises(DownloadError) as err:
        c.download(v, tmp_path)
    text = str(err.value)
    assert "ReadTimeout" in text and "PRESIGNED" not in text and "Authorization" not in text
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_worker_env_drops_the_token(monkeypatch):
    monkeypatch.setenv("CIVITAI_API_TOKEN", TOKEN)
    env = exporter.worker_env()
    assert "CIVITAI_API_TOKEN" not in env and env["HF_HUB_OFFLINE"] == "1"


def test_redact_url():
    assert civitai.redact_url(f"https://b2.civitai.com/file/x?{PRESIGNED_QUERY}#frag") == "https://b2.civitai.com/file/x"


# --- through the CLI ----------------------------------------------------------------------------


@pytest.fixture
def sd1_folder(family_tree):
    tree = family_tree("sd1")
    write_weights(tree, ("text_encoder", "unet", "vae"))
    (tree / "LICENSE").write_text("terms")
    return tree


def _patch_client(monkeypatch, fake):
    real = civitai.CivitaiClient

    def make(token, **kw):
        return real(token, transport=httpx.MockTransport(fake), sleep=lambda s: None)

    monkeypatch.setattr(civitai, "CivitaiClient", make)


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
def test_secrets_never_reach_logs_through_the_cli(sd1_folder, tmp_path, monkeypatch, caplog, capsys, error):
    def b2(request):
        raise error(f"failed for {request.url}", request=request)

    fake = FakeCivitai(b"", b2=b2)
    _patch_client(monkeypatch, fake)
    monkeypatch.setenv("CIVITAI_API_TOKEN", TOKEN)
    monkeypatch.setattr(exporter, "run_export", lambda *a, **k: pytest.fail("no export"))
    caplog.set_level(logging.DEBUG)
    rc = cli.main(["convert", str(sd1_folder), "--target", "ios", "--lora", "civitai:900106@900006:0.8",
                   "--cache-dir", str(tmp_path / "cache"), "--output-dir", str(tmp_path / "out"), "-v"])
    assert rc == 6
    out = capsys.readouterr()
    everything = out.out + out.err + caplog.text + "".join(str(r.args) for r in caplog.records)
    assert TOKEN not in everything and "PRESIGNED" not in everything and "Authorization=" not in everything
    assert "https://b2.civitai.com/file/x.safetensors" in out.err  # the redacted URL is kept


def test_dry_run_makes_metadata_requests_only(sd1_folder, tmp_path, monkeypatch, capsys):
    fake = FakeCivitai()
    _patch_client(monkeypatch, fake)
    monkeypatch.setenv("CIVITAI_API_TOKEN", TOKEN)
    rc = cli.main(["convert", "civitai:1188071@1408658", "--target", "macos", "--lora", "civitai:120096",
                   "--license-file", str(sd1_folder / "LICENSE"), "--cache-dir", str(tmp_path / "c"),
                   "--output-dir", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert all(httpx.URL(c.url).path.startswith("/api/v1/") for c in fake.calls)
    assert "Animagine XL 4.0 v4 Opt" in out and "SDXL 1.0 -> sdxl" in out and "animagineXL40_v4Opt.safetensors" in out
    assert "Sharing merges or other derivatives is not allowed." in out
    assert "the creator does not allow derivatives" in out
    assert TOKEN not in out and not (tmp_path / "c").exists()


def test_lora_as_model_refused(tmp_path, monkeypatch):
    _patch_client(monkeypatch, FakeCivitai())
    assert cli.main(["convert", "civitai:118427", "--target", "macos", "--dry-run",
                     "--output-dir", str(tmp_path)]) == 3


def test_civitai_lora_family_mismatch_refused_early(sd1_folder, tmp_path, monkeypatch):
    fake = FakeCivitai()
    _patch_client(monkeypatch, fake)
    rc = cli.main(["convert", str(sd1_folder), "--target", "ios", "--lora", "civitai:118427@128461",
                   "--output-dir", str(tmp_path)])
    assert rc == 3
    assert not any("/download/" in c.url for c in fake.calls)


def test_pony_lora_on_sdxl_checkpoint_warns(tmp_path, monkeypatch, caplog, sd1_folder):
    _patch_client(monkeypatch, FakeCivitai())
    caplog.set_level(logging.WARNING)
    assert cli.main(["convert", "civitai:1188071@1408658", "--target", "macos", "--lora", "civitai:900109@900009",
                     "--license-file", str(sd1_folder / "LICENSE"), "--output-dir", str(tmp_path), "--dry-run"]) == 0
    assert "was trained on Pony, the checkpoint is SDXL 1.0; results may differ" in caplog.text


def test_civitai_vae_family_checked(tmp_path, monkeypatch, sd1_folder):
    _patch_client(monkeypatch, FakeCivitai())
    # 900106 is listed as an SD 1.5 *LoRA*: as --vae the slot rule refuses it from metadata.
    assert cli.main(["convert", str(sd1_folder), "--target", "ios", "--vae", "civitai:900106@900006",
                     "--output-dir", str(tmp_path), "--dry-run"]) == 3
