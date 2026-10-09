# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Civitai: resolve a model or version through the public REST API, refuse what cannot be
converted from metadata alone, and download one .safetensors file with resume and SHA-256
verification.

Secrets: the API token is held by ``CivitaiClient`` only. It is sent as a bearer token to
civitai.com and nowhere else, never logged, and never written to any file. The download host
answers with a presigned URL whose query string is itself a credential, so no URL reaches a log
line or an error message without ``redact_url``; httpx's own request logging is silenced.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

import httpx

from . import __version__
from .errors import DownloadError, UnsupportedModelError, UsageError

LOG = logging.getLogger(__name__)
# httpx logs every request with its full URL at INFO; the download URL carries a credential.
for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

API = "https://civitai.com/api/v1"
DOWNLOAD_HOSTS = ("civitai.com", "www.civitai.com")
# Where Civitai's download endpoint redirects: its own b2.civitai.com and, for large checkpoints
# (measured 2026-10-08), Cloudflare R2 buckets under r2.cloudflarestorage.com. The redirect target
# never receives the API token, and every file is checked against Civitai's SHA-256.
FILE_HOST_SUFFIXES = ("civitai.com", "r2.cloudflarestorage.com")
TOKEN_ENV = "CIVITAI_API_TOKEN"
CACHE_ENV = "CAIPACK_CACHE_DIR"
DEFAULT_CACHE = Path("~/Library/Caches/coreai-diffusion-converter")
RETRY_STATUS = (429, 502, 503, 504)
BACKOFF = (2.0, 4.0, 8.0)
CHUNK = 8 << 20
FREE_MARGIN = 1 << 30
INPAINT_MESSAGE = "inpainting checkpoints (9-channel UNet) are not supported"
DISTILLED_MESSAGE = "distilled models need a scheduler the app does not implement"

_PREFIX_RE = re.compile(r"^civitai:(\d+)(?:@(\d+))?$")
_MODEL_PATH_RE = re.compile(r"^/models/(\d+)(?:/[^/]*)?/?$")
_DOWNLOAD_PATH_RE = re.compile(r"^/api/download/models/(\d+)/?$")
_FILE_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]")

# Civitai baseModel -> family (strings from GET /api/v1/enums).
BASE_MODEL_FAMILY = {
    "SD 1.4": "sd1", "SD 1.5": "sd1",
    "SD 2.0": "sd2", "SD 2.0 768": "sd2", "SD 2.1": "sd2", "SD 2.1 768": "sd2",
    "SDXL 1.0": "sdxl", "Pony": "sdxl", "Illustrious": "sdxl", "NoobAI": "sdxl",
    "SD 3": "sd3", "SD 3.5": "sd3", "SD 3.5 Medium": "sd3", "SD 3.5 Large": "sd3",
    "Flux.2 Klein 4B": "flux2", "Flux.2 Klein 4B-base": "flux2",
}
DISTILLED_BASE_MODELS = ("SD 1.5 LCM", "SD 1.5 Hyper", "SDXL 1.0 LCM", "SDXL Lightning", "SDXL Hyper",
                         "SDXL Turbo", "SDXL Distilled", "SD 3.5 Large Turbo")
NEVER_BASE_MODELS = ("Flux.2 Klein 9B", "Flux.2 Klein 9B-base")
# Default licence names, only for base models whose licence is unambiguous.
DEFAULT_LICENCE_NAMES = {
    "SD 1.4": "CreativeML OpenRAIL-M", "SD 1.5": "CreativeML OpenRAIL-M",
    "SD 2.0": "CreativeML Open RAIL++-M", "SD 2.0 768": "CreativeML Open RAIL++-M",
    "SD 2.1": "CreativeML Open RAIL++-M", "SD 2.1 768": "CreativeML Open RAIL++-M",
    "SDXL 1.0": "CreativeML Open RAIL++-M",
    "SD 3": "Stability AI Community License", "SD 3.5": "Stability AI Community License",
    "SD 3.5 Medium": "Stability AI Community License", "SD 3.5 Large": "Stability AI Community License",
}
COMMERCIAL_USE = {"Image": "selling images you generate", "RentCivit": "running it on Civitai's generator",
                  "Rent": "running it on other generation services", "Sell": "selling the model",
                  "SellMerge": "selling merges"}


def redact_url(url: str) -> str:
    """The URL without its query string and fragment (a presigned query is a credential)."""
    try:
        parts = urlsplit(str(url))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except ValueError:
        return "<url>"


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CivitaiRef:
    model_id: int | None
    version_id: int | None

    def label(self) -> str:
        if self.model_id is not None and self.version_id is not None:
            return f"{self.model_id}@{self.version_id}"
        return str(self.model_id if self.model_id is not None else f"version {self.version_id}")


def _is_civitai_host(host: str) -> bool:
    host = host.lower()
    return host == "civitai.com" or host.endswith(".civitai.com")


def _is_file_host(host: str) -> bool:
    host = host.lower()
    return any(host == s or host.endswith("." + s) for s in FILE_HOST_SUFFIXES)


def is_civitai_ref(s: str) -> bool:
    s = s.strip()
    if s.lower().startswith("civitai:"):
        return True
    if s.lower().startswith(("http://", "https://")):
        try:
            return _is_civitai_host(urlsplit(s).hostname or "")
        except ValueError:
            return False
    return False


def parse_ref(s: str) -> CivitaiRef:
    s = s.strip()
    m = _PREFIX_RE.match(s)
    if m:
        return CivitaiRef(int(m.group(1)), int(m.group(2)) if m.group(2) else None)
    if s.lower().startswith("civitai:"):
        raise UsageError(f"{s!r} is not a Civitai reference; use civitai:<model id>[@<version id>]")
    try:
        parts = urlsplit(s)
    except ValueError as err:
        raise UsageError("not a Civitai URL") from err
    if parts.scheme not in ("http", "https") or (parts.hostname or "").lower() not in DOWNLOAD_HOSTS:
        raise UsageError("not a Civitai model URL; use https://civitai.com/models/<id>")
    m = _MODEL_PATH_RE.match(parts.path)
    if m:
        version = parse_qs(parts.query).get("modelVersionId", [None])[0]
        if version is not None and not version.isdigit():
            raise UsageError("modelVersionId must be an integer")
        return CivitaiRef(int(m.group(1)), int(version) if version else None)
    m = _DOWNLOAD_PATH_RE.match(parts.path)
    if m:
        return CivitaiRef(None, int(m.group(1)))
    raise UsageError("not a Civitai model URL; use https://civitai.com/models/<id>[?modelVersionId=<id>]")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CivitaiPermissions:
    allow_no_credit: bool
    allow_commercial_use: tuple[str, ...]
    allow_derivatives: bool
    allow_different_license: bool
    creator: str = ""

    @classmethod
    def from_model(cls, model: dict) -> CivitaiPermissions:
        use = model.get("allowCommercialUse")
        if isinstance(use, str):
            use = [use] if use and use != "None" else []
        return cls(allow_no_credit=bool(model.get("allowNoCredit", True)),
                   allow_commercial_use=tuple(str(u) for u in (use or [])),
                   allow_derivatives=bool(model.get("allowDerivatives", True)),
                   allow_different_license=bool(model.get("allowDifferentLicense", True)),
                   creator=str((model.get("creator") or {}).get("username") or ""))

    def summary_lines(self) -> list[str]:
        lines = []
        if not self.allow_no_credit:
            lines.append(f"Credit the creator ({self.creator or 'see the model page'}) when you share "
                         "the model or its outputs.")
        if self.allow_commercial_use:
            uses = [COMMERCIAL_USE.get(u, u) for u in self.allow_commercial_use]
            lines.append("Commercial use allowed: " + ", ".join(uses) + ".")
        else:
            lines.append("No commercial use is allowed.")
        if not self.allow_derivatives:
            lines.append("Sharing merges or other derivatives is not allowed.")
        if not self.allow_different_license:
            lines.append("Derivatives must keep these same permissions.")
        return lines

    def to_json(self) -> dict:
        return {"allow_no_credit": self.allow_no_credit,
                "allow_commercial_use": list(self.allow_commercial_use),
                "allow_derivatives": self.allow_derivatives,
                "allow_different_license": self.allow_different_license}


@dataclass(frozen=True)
class CivitaiFile:
    file_id: int
    name: str
    size_bytes: int
    sha256: str          # lowercase, "" when Civitai publishes none
    fp: str | None
    size_kind: str | None
    download_url: str


@dataclass(frozen=True)
class CivitaiVersion:
    model_id: int
    version_id: int
    model_name: str
    version_name: str
    model_type: str
    base_model: str
    base_model_type: str
    trained_words: tuple[str, ...]
    permissions: CivitaiPermissions
    creator: str
    file: CivitaiFile

    @property
    def ref(self) -> str:
        return f"{self.model_id}@{self.version_id}"

    @property
    def page_url(self) -> str:
        return f"https://civitai.com/models/{self.model_id}?modelVersionId={self.version_id}"

    @property
    def display_name(self) -> str:
        return f"{self.model_name} {self.version_name}".strip()

    @property
    def family(self) -> str:
        return family_for_base_model(self.base_model)


def family_for_base_model(base: str) -> str:
    if base in BASE_MODEL_FAMILY:
        return BASE_MODEL_FAMILY[base]
    if base in DISTILLED_BASE_MODELS:
        raise UnsupportedModelError(f"Civitai base model '{base}': {DISTILLED_MESSAGE}")
    if base in NEVER_BASE_MODELS:
        raise UnsupportedModelError(f"Civitai base model '{base}' is not supported; only FLUX.2 Klein 4B is")
    from .families import generic_unsupported

    raise UnsupportedModelError(generic_unsupported(f"Civitai base model '{base}'"))


_FP_RANK = {"fp16": 0, "bf16": 1, "fp32": 2, None: 3}
_SIZE_RANK = {"pruned": 0, "full": 1}


def _file_order(f: dict) -> tuple:
    md = f.get("metadata") or {}
    fp, size = md.get("fp"), md.get("size")
    if fp == "fp16":
        rank = 0 if size == "pruned" else 1
    elif fp == "bf16":
        rank = 2
    elif fp == "fp32":
        rank = 3 if size == "pruned" else 4
    else:
        rank = 5
    return (rank, 0 if f.get("primary") else 1, float(f.get("sizeKB") or 0))


def choose_file(files: list[dict], slot: str, model_type: str = "") -> CivitaiFile:
    """The best .safetensors file of a version; never a pickle, quantized or training file."""
    types = {"Model", "Pruned Model"}
    if slot == "vae" or model_type == "VAE":
        types.add("VAE")
    candidates = []
    for f in files or []:
        name = str(f.get("name") or "")
        md = f.get("metadata") or {}
        if name.lower().endswith((".ckpt", ".pt", ".pth", ".bin")) or md.get("format") == "PickleTensor":
            continue
        if md.get("format") != "SafeTensor" or not name.lower().endswith(".safetensors"):
            continue
        if f.get("type") not in types:
            continue
        if md.get("fp") not in _FP_RANK:
            continue
        candidates.append(f)
    if not candidates:
        raise UnsupportedModelError("this Civitai version has no .safetensors file the converter can use; "
                                    "pickle (.ckpt/.pt) files are never downloaded")
    best = sorted(candidates, key=_file_order)[0]
    md = best.get("metadata") or {}
    sha = str((best.get("hashes") or {}).get("SHA256") or "").lower()
    return CivitaiFile(file_id=int(best.get("id") or 0), name=str(best.get("name")),
                       size_bytes=int(round(float(best.get("sizeKB") or 0) * 1024)), sha256=sha,
                       fp=md.get("fp"), size_kind=md.get("size"),
                       download_url=str(best.get("downloadUrl") or ""))


SLOT_FLAG = {"model": "SOURCE", "lora": "--lora", "vae": "--vae"}


def check_slot(model_type: str, base_model_type: str, slot: str) -> None:
    """Refuse a Civitai model used in the wrong place, or of a kind the converter cannot use."""
    if model_type == "Checkpoint":
        if slot != "model":
            raise UnsupportedModelError("this is a checkpoint; pass it as SOURCE")
    elif model_type == "LORA":
        if slot == "model":
            raise UnsupportedModelError("this is a LoRA; pass it with --lora")
        if slot == "vae":
            raise UnsupportedModelError("this is a LoRA, not a VAE")
    elif model_type == "VAE":
        if slot == "model":
            raise UnsupportedModelError("this is a VAE; pass it with --vae")
        if slot == "lora":
            raise UnsupportedModelError("this is a VAE, not a LoRA")
    elif model_type == "LoCon":
        if slot == "model":
            raise UnsupportedModelError("this is a LoCon LoRA; pass it with --lora")
        if slot == "vae":
            raise UnsupportedModelError("this is a LoCon LoRA, not a VAE")
    elif model_type == "DoRA":
        raise UnsupportedModelError("DoRA adapters are not supported; use a plain LoRA")
    else:
        raise UnsupportedModelError(f"{model_type or 'this kind of'} models are not supported")
    if base_model_type and base_model_type != "Standard":
        if base_model_type == "Inpainting":
            raise UnsupportedModelError(INPAINT_MESSAGE)
        if base_model_type == "Refiner":
            raise UnsupportedModelError("SDXL refiner models are not supported")
        if base_model_type == "Pix2Pix":
            raise UnsupportedModelError("image-to-image models are not supported")
        raise UnsupportedModelError(f"Civitai base model type '{base_model_type}' is not supported")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def default_cache_dir(arg: str | Path | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    env = os.environ.get(CACHE_ENV)
    return Path(env).expanduser().resolve() if env else DEFAULT_CACHE.expanduser()


def _safe_name(name: str) -> str:
    return _FILE_NAME_RE.sub("-", name).strip(" .") or "file"


class CivitaiClient:
    def __init__(self, token: str | None, *, transport: httpx.BaseTransport | None = None,
                 timeout: float = 30.0, sleep: Any = time.sleep) -> None:
        self._token = token or None
        self._sleep = sleep
        self._client = httpx.Client(transport=transport, timeout=timeout, follow_redirects=False,
                                    headers={"User-Agent": f"coreai-diffusion-converter/{__version__}"})

    @property
    def has_token(self) -> bool:
        return self._token is not None

    def close(self) -> None:
        self._client.close()

    # -- requests ------------------------------------------------------------------------------

    def _send(self, method: str, url: str, *, auth: bool, what: str, headers: dict | None = None,
              stream: bool = False) -> httpx.Response:
        hdrs = dict(headers or {})
        if auth and self._token:
            hdrs["Authorization"] = f"Bearer {self._token}"
        for attempt in range(len(BACKOFF) + 1):
            try:
                req = self._client.build_request(method, url, headers=hdrs)
                resp = self._client.send(req, stream=stream)
            except httpx.HTTPError as err:
                raise DownloadError(f"{what} failed: {type(err).__name__} for {redact_url(url)}") from None
            if resp.status_code in RETRY_STATUS and attempt < len(BACKOFF):
                resp.close()
                LOG.warning("Civitai answered %d; retrying in %g s", resp.status_code, BACKOFF[attempt])
                self._sleep(BACKOFF[attempt])
                continue
            return resp
        return resp  # pragma: no cover - the loop always returns

    def _status_error(self, status: int, what: str) -> Exception:
        if status == 401:
            if self._token:
                return UsageError("Civitai rejected the API token")
            return UsageError("Civitai refused the request: this model needs a Civitai API token (set "
                              "CIVITAI_API_TOKEN or pass --civitai-token) or is in early access")
        if status == 403:
            if self._token:
                return UsageError("Civitai refused the request: this model is in early access or not "
                                  "available to your account")
            return UsageError("Civitai refused the request: this model needs a Civitai API token (set "
                              "CIVITAI_API_TOKEN or pass --civitai-token) or is in early access")
        if status == 404:
            return UsageError(f"Civitai model or version {what} not found")
        return DownloadError(f"Civitai answered HTTP {status} for {what}")

    def _get_json(self, path: str, what: str) -> dict:
        url = f"{API}{path}"
        resp = self._send("GET", url, auth=True, what=f"Civitai request {redact_url(url)}")
        if resp.status_code != 200:
            raise self._status_error(resp.status_code, what)
        try:
            data = resp.json()
        except ValueError:
            raise DownloadError(f"Civitai returned an unreadable answer for {what}") from None
        if not isinstance(data, dict):
            raise DownloadError(f"Civitai returned an unexpected answer for {what}")
        return data

    # -- metadata ------------------------------------------------------------------------------

    def resolve(self, ref: CivitaiRef, slot: str) -> CivitaiVersion:
        if ref.version_id is not None:
            version = self._get_json(f"/model-versions/{ref.version_id}", str(ref.version_id))
            model_id = int(version.get("modelId") or ref.model_id or 0)
            if ref.model_id is not None and model_id != ref.model_id:
                raise UsageError(f"Civitai version {ref.version_id} belongs to model {model_id}, not {ref.model_id}")
            model = self._get_json(f"/models/{model_id}", str(model_id))
        else:
            model = self._get_json(f"/models/{ref.model_id}", str(ref.model_id))
            chosen = None
            for v in model.get("modelVersions") or []:
                if (v.get("status") or "Published") == "Published" and (v.get("availability") or "Public") == "Public":
                    chosen = v
                    break
            if chosen is None:
                raise UnsupportedModelError("this Civitai model has no published public version")
            print(f"using the creator's default version {chosen.get('id')} {chosen.get('name') or ''}; "
                  "pin one with @<version>".replace("  ", " "))
            version = self._get_json(f"/model-versions/{chosen.get('id')}", str(chosen.get("id")))
            model_id = int(model.get("id") or ref.model_id or 0)
        status = version.get("status") or "Published"
        availability = version.get("availability") or "Public"
        if status != "Published" or availability != "Public":
            raise UnsupportedModelError(f"this Civitai version is not published for download ({status}, {availability})")
        usage = version.get("usageControl") or "Download"
        if usage != "Download":
            raise UnsupportedModelError(f"this Civitai version cannot be downloaded ({usage} access only)")
        model_type = str(model.get("type") or (version.get("model") or {}).get("type") or "")
        base_model_type = str(version.get("baseModelType") or "Standard")
        check_slot(model_type, base_model_type, slot)
        file = choose_file(version.get("files") or [], slot, model_type)
        perms = CivitaiPermissions.from_model(model)
        return CivitaiVersion(
            model_id=model_id, version_id=int(version.get("id") or ref.version_id or 0),
            model_name=str(model.get("name") or (version.get("model") or {}).get("name") or ""),
            version_name=str(version.get("name") or ""), model_type=model_type,
            base_model=str(version.get("baseModel") or ""), base_model_type=base_model_type,
            trained_words=tuple(str(w) for w in (version.get("trainedWords") or []) if str(w).strip()),
            permissions=perms, creator=perms.creator, file=file)

    # -- download ------------------------------------------------------------------------------

    def cached_path(self, version: CivitaiVersion, cache_dir: Path) -> Path:
        return cache_dir / "civitai" / str(version.version_id) / f"{version.file.file_id}-{_safe_name(version.file.name)}"

    def download(self, version: CivitaiVersion, cache_dir: Path, *, allow_unverified: bool = False) -> Path:
        f = version.file
        if not f.sha256 and not allow_unverified:
            raise UnsupportedModelError(f"Civitai publishes no SHA-256 for {f.name}; pass "
                                        "--allow-unverified-download to accept it unverified")
        final = self.cached_path(version, cache_dir)
        if final.is_file():
            if f.sha256:
                _, digest = _hash_path(final)
                if digest == f.sha256:
                    LOG.info("%s: using the cached download", f.name)
                    return final
                LOG.warning("%s: the cached file does not match Civitai's hash; downloading again", f.name)
                final.unlink()
            else:
                return final
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name(final.name + ".partial")
        have = partial.stat().st_size if partial.is_file() else 0
        free = shutil.disk_usage(final.parent).free
        if free < f.size_bytes - have + FREE_MARGIN:
            raise UsageError(f"not enough free disk space for {f.name} ({f.size_bytes / 1e9:.1f} GB) in the cache "
                             "directory; free some space or pass --cache-dir")
        lock_path = final.with_name(final.name + ".partial.lock")
        lock = open(lock_path, "a+b")  # noqa: SIM115 - held until the rename
        try:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise UsageError(f"{f.name} is being downloaded by another caipack run; wait for it to finish") from None
            digest = self._fetch(version, partial)
            if f.sha256 and digest != f.sha256:
                partial.unlink(missing_ok=True)
                raise DownloadError(f"{f.name}: SHA-256 does not match Civitai's published hash; the download was discarded")
            if not f.sha256:
                LOG.warning("%s: no published SHA-256; accepted unverified (sha256 %s)", f.name, digest[:12])
            os.replace(partial, final)
        finally:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            finally:
                lock.close()
                lock_path.unlink(missing_ok=True)
        return final

    def _fetch(self, version: CivitaiVersion, partial: Path) -> str:
        """Download into ``partial`` (resuming it) and return the whole file's SHA-256."""
        f = version.file
        api_url = f"https://civitai.com/api/download/models/{version.version_id}?fileId={f.file_id}"
        what = f"download of {f.name}"
        resp = self._send("GET", api_url, auth=True, what=what)
        try:
            if resp.status_code not in (301, 302, 303, 307, 308):
                raise self._status_error(resp.status_code, str(version.version_id)) if resp.status_code != 200 else \
                    DownloadError("Civitai did not redirect to the file; try again later")
            location = resp.headers.get("location", "")
        finally:
            resp.close()
        try:
            loc = urlsplit(location)
        except ValueError:
            raise DownloadError("unexpected download location") from None
        if loc.scheme != "https" or not _is_file_host(loc.hostname or ""):
            raise DownloadError(f"unexpected download host {loc.hostname or '<none>'}")
        have = partial.stat().st_size if partial.is_file() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        if have:
            print(f"{f.name}: resuming at {have} bytes")
        # No Authorization header: the presigned URL authorises itself, and a bearer token must
        # never leave the API host.
        resp = self._send("GET", location, auth=False, what=what, headers=headers, stream=True)
        try:
            ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if resp.status_code == 416:
                return _hash_path(partial)[1]  # the partial is already complete
            if resp.status_code not in (200, 206) or ctype in ("text/html", "application/json"):
                raise DownloadError(f"Civitai did not return the model file (got {ctype or 'no content type'}, "
                                    f"HTTP {resp.status_code}); try again later")
            h = hashlib.sha256()
            if resp.status_code == 206:
                total = _content_range_total(resp.headers.get("content-range", ""))
                mode = "ab"
            else:
                total = int(resp.headers.get("content-length") or -1)
                have, mode = 0, "wb"
            if total < 0 or abs(total - f.size_bytes) > 1024:
                raise DownloadError(f"Civitai did not return the model file (got {ctype or 'no content type'}, "
                                    f"{total} bytes); try again later")
            if mode == "ab":
                with partial.open("rb") as existing:
                    for chunk in iter(lambda: existing.read(CHUNK), b""):
                        h.update(chunk)
            done, next_report = have, 0.0
            with partial.open(mode) as out:
                for chunk in resp.iter_bytes(CHUNK):
                    out.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    frac = done / total if total else 1.0
                    if frac >= next_report:
                        LOG.info("%s: %.0f%% (%.2f of %.2f GB)", f.name, frac * 100, done / 1e9, total / 1e9)
                        next_report = frac + 0.05
            if done != total:
                raise DownloadError(f"{what} ended early ({done} of {total} bytes); run again to resume")
            return h.hexdigest()
        except httpx.HTTPError as err:
            raise DownloadError(f"{what} failed: {type(err).__name__} for {redact_url(location)}") from None
        finally:
            resp.close()


def _content_range_total(value: str) -> int:
    m = re.match(r"^bytes \d+-\d+/(\d+)$", value.strip())
    return int(m.group(1)) if m else -1


def _hash_path(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def notice_section(version: CivitaiVersion) -> str:
    lines = [f"Civitai permissions for {version.display_name} ({version.page_url}), by {version.creator or 'unknown'}:"]
    lines += [f"- {line}" for line in version.permissions.summary_lines()]
    return "\n".join(lines) + "\n"
