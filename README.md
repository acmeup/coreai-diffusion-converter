# coreai-diffusion-converter

Convert Stable Diffusion 1.x / 2.x / 3.x, SDXL and FLUX.2 Klein models, including fine-tunes,
single-file `.safetensors` checkpoints and Civitai models, optionally with LoRAs merged in, into one
`.caipack` file for Apple Core AI on iOS 27 and macOS 27.

A `.caipack` holds the exported Core AI bundle, its tokenizer, the model's licence and a
`pack.json` describing how to run it (pipeline, image size, steps, guidance, scheduler). The
format is documented in [docs/FORMAT.md](docs/FORMAT.md).

## Requirements

- An Apple-silicon Mac on macOS 26 or later (the Core AI compiler wheels are macOS arm64 only).
- Python 3.11 and [uv](https://docs.astral.sh/uv/).
- Free disk space of about three times the model's size.

## Install

```bash
git clone https://github.com/acmeup/coreai-diffusion-converter.git
cd coreai-diffusion-converter
uv sync
```

## Usage

```bash
# A Hugging Face model, for iPhone and iPad
uv run caipack convert stable-diffusion-v1-5/stable-diffusion-v1-5 --target ios

# A local diffusers folder, for Mac
uv run caipack convert ./my-model --target macos --name "My Model"

# A single-file checkpoint with a replacement VAE and clip skip 2
uv run caipack convert ./model.safetensors --target ios --license-file ./LICENSE \
    --vae ./vae.safetensors --clip-skip 2

# An SDXL model from Civitai with a LoRA from Civitai merged in at 0.8 (Mac only)
export CIVITAI_API_TOKEN=...   # only needed for models that require a login
uv run caipack convert civitai:1188071@1408658 --lora civitai:118427@128461:0.8 \
    --target macos --license-file ./LICENSE.md

# Check a pack (re-hashes every file) or print its description
uv run caipack validate "My Model.ios.caipack"
uv run caipack inspect "My Model.ios.caipack"
```

`--dry-run` prints the resolved plan (family, components, size, precision, steps, guidance,
scheduler, tuning, licence, download filter and output name) without exporting.

Other options: `--size {512,768,1024}`, `--precision {fp16,4bit}`, `--steps N`, `--guidance G`,
`--revision REV`, `--base REPO_OR_DIR` (configs and text encoders for a single-file checkpoint that
lacks them), `--license-name NAME`, `--allow-missing-license`, `--lora SOURCE[:SCALE]`,
`--civitai-token TOKEN`, `--cache-dir DIR`, `--allow-unverified-download`, `--output-dir DIR`,
`--work-dir DIR`, `--keep-work`, `--overwrite`, `-v`.

Exit codes: 0 success, 2 usage, 3 unsupported model, 4 export failure, 5 validation failure,
6 download failure.

The export runs offline: the network is used only to resolve and download a Hugging Face or
Civitai model. Hugging Face downloads go to your own Hugging Face cache (`HF_HOME`); Civitai
downloads go to the converter's cache (below). The converter never deletes either.

## Supported families

| Family | Detected from | Sizes | Default precision |
|---|---|---|---|
| SD 1.x | `StableDiffusionPipeline`, 768-wide text conditioning | 512 | fp16 |
| SD 2.x | `StableDiffusionPipeline`, 1024-wide text conditioning | 512, 768 | fp16 |
| SDXL (SDXL 1.0, Pony, Illustrious, NoobAI, Animagine) | `StableDiffusionXLPipeline` and SDXL single files; `--target macos` only | 1024 | 4-bit UNet and text encoders, float32 VAE decoder |
| SD 3.x | `StableDiffusion3Pipeline` (exported without the T5 encoder) | 512, 1024 | 4-bit |
| FLUX.2 Klein | `Flux2KleinPipeline`, Klein 4B geometry | 512, 1024 | 4-bit |

SDXL at 4-bit is about 2.1 GB and renders clean, on-prompt images, but details and colours can differ
from the original model's output for the same seed. Use `--precision fp16` (about 7 GB) when the output
must match the original model.

Not supported: LyCORIS adapters (LoHa, LoKr, LoCon with mid weights, OFT, IA3) and DoRA,
textual-inversion embeddings, SDXL refiners, inpainting checkpoints (9-channel UNet), image-to-image
and video pipelines, distilled models (LCM, Lightning, Turbo, Hyper), FLUX.1, FLUX.2 dev and FLUX.2
Klein 9B. FLUX.2 single-file checkpoints are not supported; convert a FLUX.2 Klein diffusers folder
or Hub id instead. Pickle checkpoints (`.ckpt`, `.pt`, `.bin`) are never loaded.

Each pack is traced at one image size. `--target ios` defaults to 512; `--target macos` defaults
to the family's native size. SDXL renders only at 1024 px (smaller traces of SDXL fine-tunes were
judged unacceptable), so SDXL packs are made for `--target macos` only; `--target ios` is refused.

SDXL notes: the text encoders output the penultimate hidden state (what "clip skip 2" means in other
tools), so `--clip-skip` is refused for SDXL; `--vae` and `--prediction-type` apply. A single-file
SDXL checkpoint is loaded with the configuration of `stabilityai/stable-diffusion-xl-base-1.0`
unless `--base` names another; a checkpoint with a `v_pred` key (the NoobAI v-prediction convention)
is converted as `v_prediction`, and one with a `ztsnr` key converts with a warning (the app's
scheduler does not rescale betas for zero terminal SNR). A UNet-only SDXL file needs `--base`. The
VAE decoder is kept in float32 because the stock SDXL VAE overflows in float16.

## Options for SD 1.x / 2.x fine-tunes

- `--vae FILE_OR_REPO` replaces the model's VAE decoder before export, for fine-tunes distributed
  without a VAE or with a weak one. The VAE must be an SD 1.x / 2.x VAE (4 latent channels, same
  block layout).
- `--clip-skip N` (1-4) uses the text encoder's output N-1 layers before the last one, as some
  fine-tunes are trained to expect; 2 means the penultimate layer. The layers after it are removed
  before export, which gives the same embeddings as diffusers' `clip_skip=N-1`.
- `--prediction-type {epsilon,v_prediction}` sets the noise prediction type when a single-file
  checkpoint's own configuration is ambiguous.

A single-file SD 1.x / 2.x checkpoint is loaded with the configuration of
`stable-diffusion-v1-5/stable-diffusion-v1-5` or `sd2-community/stable-diffusion-2-1` (a 768 px,
v-prediction model) unless `--base` names another one. For an SD 2.x checkpoint trained at 512 px
with epsilon prediction, pass `--prediction-type epsilon`. The inferred prediction type is printed
during conversion. A `--vae` given as a Hub id is downloaded before the export starts.

Every option used is recorded in `pack.json` and in the pack's `CHANGES.md`.

## LoRAs

`--lora SOURCE[:SCALE]` merges a LoRA into the model before export. Repeat it to merge several (at
most 8); they are applied in the order given. `SCALE` is between -4 and 4 (default 1; 0 is refused;
negative scales are used by "slider" LoRAs). `SOURCE` is one of:

- a local `.safetensors` file;
- a Hugging Face file, `org/repo/path/file.safetensors`, or `org/repo` when the repository holds
  exactly one top-level `.safetensors` file;
- a Civitai reference (below).

Kohya, diffusers and PEFT LoRA formats are accepted, for SD 1.x, SD 2.x, SDXL, SD 3.x (CLIP text
encoders and the transformer; T5 weights are dropped) and FLUX.2 Klein 4B (transformer only). A
LoRA's family is read from its weights and must match the model's. The LoRA is fused into the
weights before clip skip, quantization and tracing, so the pack needs no LoRA support at run time;
every LoRA is checked to have changed at least one layer of each component it targets.

Each merged LoRA is recorded in `pack.json` (`conversion.loras`: file name, SHA-256, scale, source,
trigger words, Civitai permissions) and in `CHANGES.md`, and its trigger words are printed at the
end of the run. The pack keeps the base model's licence; each LoRA's own terms are your
responsibility.

## Civitai

A model, LoRA or VAE can be given as `civitai:<model id>`, `civitai:<model id>@<version id>`, a
model page URL (`https://civitai.com/models/<id>[?modelVersionId=<id>]`) or a download URL
(`https://civitai.com/api/download/models/<version id>`). Without a version, the creator's default
version (the first published one on the model page) is used and named; pin one with `@<version>`.

- A Civitai API token is needed only for models that require a login. Set `CIVITAI_API_TOKEN`
  (preferred) or pass `--civitai-token`; a token on the command line is visible in `ps` and in
  your shell history. The token is sent only to civitai.com, never to the download host, and is
  never written to any file or log.
- Only `.safetensors` files are downloaded (fp16 preferred, then bf16, then fp32; quantized files
  are skipped); pickle files never are. Every file is checked against Civitai's published SHA-256;
  a file without one is refused unless `--allow-unverified-download` is given (then `pack.json`
  records `"verified": false`). Interrupted downloads resume.
- Downloads are kept in `--cache-dir`, else `CAIPACK_CACHE_DIR`, else
  `~/Library/Caches/coreai-diffusion-converter` (an SDXL checkpoint is about 7 GB). The converter
  never deletes them.
- Wrong or unsupported models are refused from Civitai's metadata before any weight is downloaded:
  a LoRA given as the model (or the reverse), inpainting, refiner and image-to-image models,
  distilled base models, FLUX.2 Klein 9B, unknown base models, and versions that are unpublished or
  generation-only. Civitai base models map to families: SD 1.4/1.5 -> SD 1.x; SD 2.0/2.1 -> SD 2.x;
  SDXL 1.0, Pony, Illustrious, NoobAI -> SDXL; SD 3/3.5 -> SD 3.x; Flux.2 Klein 4B -> FLUX.2 Klein.
- Civitai's API carries no licence text: pass the model's licence with `--license-file` (and
  `--license-name`). The licence name defaults only where it is unambiguous (SD 1.x: CreativeML
  OpenRAIL-M; SD 2.x and SDXL 1.0: CreativeML Open RAIL++-M; SD 3.x: Stability AI Community
  License). Illustrious and NoobAI (Fair AI Public License 1.0-SD) and Pony (a modified
  OpenRAIL++-M) need `--license-name`.
- Each Civitai item's permissions are summarised in the pack's `NOTICE`. When a creator does not
  allow derivatives, the conversion continues with a warning (a converted or merged pack is a
  derivative; keep it for your own use) and the sentence is recorded in `CHANGES.md`. These
  summaries are informational copies of Civitai's settings, not legal advice.

## Licences and use

- This project ships no models.
- Every converted model keeps its own licence. The licence file travels inside the pack, and you
  must comply with it and with any acceptable-use policy it contains.
- Gated models need your own Hugging Face login (`hf auth login`) after accepting the licence on
  the model page.
- A LoRA's own terms apply in addition to the model's licence.
- Model and company names are used only to describe compatibility.
- This project is not affiliated with or endorsed by Apple, Stability AI, Black Forest Labs or
  Civitai.

## Importing

Packs can be imported by apps that support the .caipack format, e.g. Privacy AI.

## Development

```bash
uv run pytest                                   # fast tests (offline)
CAIPACK_INTEGRATION_SOURCE=<diffusers folder> uv run pytest -m slow   # real conversions
CAIPACK_SDXL_SOURCE=<diffusers SDXL folder> CAIPACK_PARITY_WORK=<dir> \
    uv run pytest -m slow tests/test_sdxl_parity.py -s                  # SDXL export parity
CAIPACK_LIVE=1 uv run pytest -m live            # real Civitai API requests (no downloads)
python scripts/identity_scan.py --tree --commits HEAD
```

Slow-test variables for `tests/test_integration_convert.py`: `CAIPACK_INTEGRATION_SOURCE` (SD 1.x
folder), `CAIPACK_INTEGRATION_SDXL_SOURCE` (SDXL folder), `CAIPACK_INTEGRATION_SDXL_LORA` (a Kohya
SDXL LoRA), `CAIPACK_INTEGRATION_SD1_SINGLE_FILE` and `CAIPACK_INTEGRATION_SD1_LORA` (an SD 1.x
single file and a Kohya LoRA with text-encoder weights), `CAIPACK_INTEGRATION_FLUX2_SOURCE` and
`CAIPACK_INTEGRATION_FLUX2_LORA` (a Klein 4B folder and a transformer LoRA).

Text-encoder LoRAs under transformers 5 (measured with diffusers 0.37.1, transformers 5.12.1,
peft 0.21.2): `CLIPTextModel` (SD 1.x/2.x and SDXL text encoder 1) is flat, and diffusers' text-encoder
LoRA loader fails on it (it looks ranks up under `text_model.` module names and raises, which would
also abort the UNet part), so those weights are merged by the converter itself.
`CLIPTextModelWithProjection` (SDXL text encoder 2, SD 3's CLIP encoders) still nests `text_model.`
and goes through diffusers. `tests/test_lora_fuse_tiny.py` pins both decisions.

Single-file checkpoints are rebuilt as a diffusers tree with transformers 5, whose tokenizer save
writes only `tokenizer.json`; the converter adds `vocab.json`, `merges.txt` and
`special_tokens_map.json` derived from it, which Core AI's tokenizer needs (0.1.0 packs made from
single files lack them and should be reconverted).

SDXL parity (Animagine XL 4.0, measured on an M4 Pro, GPU, 2026-10-08):

| Component (vs the torch module in float32) | fp16 | 4-bit (default) |
|---|---|---|
| Text encoder 1, hidden states: max abs / min token cosine | 0.294 / 0.999987 | 2.00 / 0.869 |
| Text encoder 2, hidden states: max abs / min token cosine | 0.145 / 0.99997 | 5.77 / 0.660 |
| Text encoder 2, pooled: max abs | 0.0018 | 0.185 |
| UNet, one mid-denoise step: max abs / cosine | 0.0024 / 0.99999994 | 0.225 / 0.99963 |
| VAE decoder (float32 in both): max abs / PSNR | 1.5e-5 / 136 dB | same |
| End to end, 25 steps, against diffusers with the same latents: PSNR (3 prompts) | 33.7-40.2 dB | 16.0-16.8 dB |

The fp16 pack reproduces diffusers almost exactly. The 4-bit pack (2.1 GB instead of 7.0 GB)
renders coherent, on-prompt images that differ from the fp16 ones in detail and colour; most of
that difference comes from the 4-bit UNet (fp16 text encoders with a 4-bit UNet: 16.1-17.8 dB;
4-bit text encoders with an fp16 UNet: 17.8-26.1 dB). The thresholds in `tests/test_sdxl_parity.py`
are these measurements with 25 % margin.

## Licence

Apache License 2.0, Copyright 2026 AcmeUp Inc. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The converter uses Apple's [coreai-models](https://github.com/apple/coreai-models) (BSD 3-Clause
License) as a dependency and adapts it at run time; no Apple code is included in this repository.
