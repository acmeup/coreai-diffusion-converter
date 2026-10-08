# coreai-diffusion-converter

Convert Stable Diffusion 1.x / 2.x / 3.x and FLUX.2 Klein diffusers models, including fine-tunes
and single-file `.safetensors` checkpoints, into one `.caipack` file for Apple Core AI on
iOS 27 and macOS 27.

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

# Check a pack (re-hashes every file) or print its description
uv run caipack validate "My Model.ios.caipack"
uv run caipack inspect "My Model.ios.caipack"
```

`--dry-run` prints the resolved plan (family, components, size, precision, steps, guidance,
scheduler, tuning, licence, download filter and output name) without exporting.

Other options: `--size {512,768,1024}`, `--precision {fp16,4bit}`, `--steps N`, `--guidance G`,
`--revision REV`, `--base REPO_OR_DIR` (configs and text encoders for a single-file checkpoint that
lacks them), `--license-name NAME`, `--allow-missing-license`, `--output-dir DIR`,
`--work-dir DIR`, `--keep-work`, `--overwrite`, `-v`.

Exit codes: 0 success, 2 usage, 3 unsupported model, 4 export failure, 5 validation failure.

The export runs offline: the network is used only to resolve and download a Hugging Face model.
Downloads go to your own Hugging Face cache (`HF_HOME`), which the converter never deletes.

## Supported families

| Family | Detected from | Sizes | Default precision |
|---|---|---|---|
| SD 1.x | `StableDiffusionPipeline`, 768-wide text conditioning | 512 | fp16 |
| SD 2.x | `StableDiffusionPipeline`, 1024-wide text conditioning | 512, 768 | fp16 |
| SD 3.x | `StableDiffusion3Pipeline` (exported without the T5 encoder) | 512, 1024 | 4-bit |
| FLUX.2 Klein | `Flux2KleinPipeline`, Klein 4B geometry | 512, 1024 | 4-bit |

Not supported: SDXL-based models, inpainting checkpoints (9-channel UNet), FLUX.2 dev,
image-to-image and video pipelines, LoRAs, and textual-inversion embeddings. FLUX.2 single-file
checkpoints are not supported; convert a FLUX.2 Klein diffusers folder or Hub id instead. Pickle
checkpoints (`.ckpt`, `.pt`, `.bin`) are never loaded.

Each pack is traced at one image size. `--target ios` defaults to 512; `--target macos` defaults
to the family's native size.

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

## Licences and use

- This project ships no models.
- Every converted model keeps its own licence. The licence file travels inside the pack, and you
  must comply with it and with any acceptable-use policy it contains.
- Gated models need your own Hugging Face login (`hf auth login`) after accepting the licence on
  the model page.
- Model and company names are used only to describe compatibility.
- This project is not affiliated with or endorsed by Apple, Stability AI or Black Forest Labs.

## Importing

Packs can be imported by apps that support the .caipack format, e.g. Privacy AI.

## Development

```bash
uv run pytest                                   # fast tests
CAIPACK_INTEGRATION_SOURCE=<diffusers folder> uv run pytest -m slow   # real conversions
python scripts/identity_scan.py --tree --commits HEAD
```

## Licence

Apache License 2.0, Copyright 2026 AcmeUp Inc. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The converter uses Apple's [coreai-models](https://github.com/apple/coreai-models) (BSD 3-Clause
License) as a dependency and adapts it at run time; no Apple code is included in this repository.
