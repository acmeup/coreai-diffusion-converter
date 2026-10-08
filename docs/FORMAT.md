# The .caipack format (format_version 1)

A `.caipack` is one file that carries everything an app needs to install an Apple Core AI
text-to-image model: the exported Core AI bundle, its tokenizer and sidecar files, the model's
licence, and `pack.json`, which describes how to run the model. An importer extracts the archive
unchanged into one folder; `pack.json` stays inside that folder as the installed model's record.

The machine-readable twin of this document is [`schema/pack.schema.json`](../schema/pack.schema.json)
(types and enums only). The coded rules in `src/coreai_diffusion_converter/validate.py` are
authoritative, and the fixtures in `tests/fixtures/packs/` pin them for every implementation.

## Uniform Type Identifier

The reference importer registers:

| | |
|---|---|
| UTI | `com.acmeup.caipack` |
| Conforms to | `public.data`, `public.archive` (deliberately **not** `public.zip-archive`, so file managers do not offer to expand a multi-gigabyte pack) |
| Extension | `.caipack` |
| MIME type | `application/x-caipack` |

## Container

- A ZIP archive. Every entry is **STORED** (method 0, no compression). Weights do not compress,
  and STORED makes extraction a plain copy with incremental hashing.
- ZIP64 is used for entries of 2 GiB and more (and is always accepted).
- No directory entries, no symlinks, no extra fields other than ZIP64, no comments.
- Entries are sorted by path, with `pack.json` first.
- Every entry has the fixed timestamp 1980-01-01 00:00:00, `create_system` 3 (Unix) and mode
  `0100644`, so two conversions of the same input differ only in `created_at`.

## Root layout

The archive root is exactly the installed bundle:

```text
pack.json
LICENSE                 the model's licence (absent only when packed with --allow-missing-license)
NOTICE                  when the source has one, and always for the Stability AI Community License
CHANGES.md              the distributor's statement of changes
metadata.json           the exporter's pipeline descriptor
*.aimodel/main.mlirb, main.hash, metadata.json   every file of each Core AI package
tokenizer/..., tokenizer_2/...
vae_bn_mean.npy, vae_bn_var.npy   (FLUX.2)
```

## pack.json

UTF-8 JSON, snake_case keys. Unknown fields are ignored by readers (forward compatibility).

| Field | Type | Meaning |
|---|---|---|
| `format` | string | Always `"caipack"`. |
| `format_version` | integer | `1`. A reader refuses a larger version it does not know. |
| `id` | string | `^[a-z0-9][a-z0-9-]{0,47}$`. Stable identity of the model; re-importing the same id replaces it. |
| `name` | string | Display name, 1-80 characters. |
| `description` | string | At most 500 characters. |
| `family` | string | `sd1`, `sd2`, `sd3` or `flux2`. |
| `pipeline` | string | `stable_diffusion` (sd1, sd2), `sd3` or `flux2`. Must agree with `family`. |
| `target` | string | `ios` or `macos`: the device class the pack was converted for. Informational for importers. |
| `supported_sizes` | [integer] | Exactly one element, equal to `default_size`. |
| `default_size` | integer | The traced image edge in pixels. sd1: 512; sd2: 512 or 768; sd3 and flux2: 512 or 1024. |
| `default_steps`, `max_steps` | integer | `1 <= default_steps <= max_steps <= 100`. |
| `guidance_scale` | number | `0...30`. |
| `scheduler` | string | `dpmpp`, `pndm` or `flow_match_euler`. |
| `precision` | string | `fp16` or `4bit` (weight precision). |
| `compute_precision` | string | `float16`. |
| `lazy_model_loading` | boolean | Load components on demand. |
| `excluded_architectures` | [string] | Device architecture prefixes the model is known to fail on (for example `h13`). |
| `assets` | [string] | Paths an importer checks before calling the model ready: each is a listed file or a directory prefix of listed files. |
| `source` | object | `kind` (`hf`, `folder`, `single_file`), `ref` (Hub id, folder name or file name; never an absolute path), `revision` (Hub commit or null). |
| `conversion` | object | Informational: `clip_skip`, `vae`, `prediction_type`. |
| `license` | object | `name`; `file` (`"LICENSE"` or null); `notice_file` (`"NOTICE"` or null). |
| `attribution` | string | A line the model's licence requires to be displayed, or empty. |
| `converter` | object | `name`, `version`, `exporter` (the exporter repository and commit). |
| `created_at` | string | UTC timestamp, ISO 8601 with `Z`. |
| `files` | [object] | Every archive entry except `pack.json`: `path`, `size` (bytes), `sha256` (lowercase hex). |

## Validation rules and error codes

Readers apply these rules in this order and report the first failure by its code. The steps are:
decode, rules 1-8, rule 11 (when `metadata.json` is available), rule 9 (when the archive is
available), rule 10 (full validation).

**Decode.** A file that is not a ZIP archive, or has no `pack.json` entry, gives `not_a_pack`. `pack.json` must be a JSON object. A missing field takes a default that then fails its
rule (missing `format` gives `not_a_pack`, missing `format_version` gives
`format_version_invalid`). A field present with the wrong JSON type (for example a string where an
integer is expected, or a boolean `format_version`) gives `unreadable_pack_json`. A `pack.json`
larger than 4 MiB gives `files_invalid`.

1. `format == "caipack"`, else `not_a_pack`. `format_version` must be at least 1
   (`format_version_invalid`); greater than 1 gives `format_version_unsupported`.
2. `id` matches the id pattern (`invalid_id`). `name` is 1-80 characters (`invalid_name`).
   `description` is at most 500 characters (also reported as `invalid_name`).
3. `family` is known (`family_unsupported`); `pipeline` agrees with it (`pipeline_mismatch`);
   `target` is `ios` or `macos` (`target_invalid`).
4. One traced size (`sizes_invalid`): `supported_sizes == [default_size]` and the size is in the
   family's set. For `flux2`, 512 requires assets `Transformer_512.aimodel` and
   `VAEDecoder_half.aimodel`; 1024 requires `Transformer.aimodel` and `VAEDecoder.aimodel`.
5. `1 <= default_steps <= max_steps <= 100` (`steps_invalid`); `0 <= guidance_scale <= 30`
   (`guidance_invalid`); known `scheduler` (`scheduler_unknown`); known `precision`
   (`precision_invalid`).
6. Paths, for every `files[].path` and every `assets[]` entry: relative, `/`-separated, no empty
   component; each component only ASCII `[A-Za-z0-9._-]` and not starting with `.` (so no `..` and
   no hidden files); `pack.json` is never listed (`unsafe_path`). File paths are unique when
   compared case-insensitively (`duplicate_path`).
7. At most 20,000 files; every `size` an integer in `0...64 GiB`; every `sha256` 64 lowercase hex
   characters; the total, summed after every element passed the per-file cap, at most 64 GiB
   (`files_invalid`).
8. `metadata.json` is listed (`metadata_missing`). Every asset is a listed file or a directory
   prefix of one (`asset_missing`). A non-null `license.file` equals `"LICENSE"` and is listed, and
   a non-null `license.notice_file` equals `"NOTICE"` and is listed (`license_file_missing`).
9. Archive entries, compared as a list: a duplicate name, exact or case-folded
   (`duplicate_entry`); an entry not in `files` and not `pack.json` (`entry_not_listed`); a
   listed file or `pack.json` without an entry (`entry_missing`); a compressed entry
   (`entry_compressed`); a directory or symlink entry (`entry_not_regular_file`); an entry whose
   size does not fit a signed 64-bit integer or differs from `size` (`size_mismatch`). Each of
   these checks runs over the whole list before the next one.
10. Full validation re-hashes every entry (`checksum_mismatch`).
11. `metadata.json` agrees with `pack.json` (`metadata_mismatch`): `diffusion.type` is
    `stable-diffusion`, `stable-diffusion-3` or `flux2` for the pipeline; for sd1, sd2 and sd3
    `diffusion.image_size == default_size`; for sd1 and sd2 `diffusion.prediction_type` is
    `epsilon` or `v_prediction`.

## Versioning

The format is versioned by `format_version`. Adding optional fields does not change the version;
any change an existing reader would misinterpret does.
