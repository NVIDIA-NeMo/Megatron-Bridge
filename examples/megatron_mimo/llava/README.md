# Heterogeneous MIMO LLaVA

End-to-end **MegatronMIMO** recipes that train a LLaVA-style multimodal model
where each module runs on its **own, disjoint set of GPUs with its own
parallelism plan** (heterogeneous / "hetero" layout). A Vicuna-7B language
model, a CLIP ViT-L/14 vision encoder, and (optionally) a Whisper-base audio
encoder are wired into a single training graph, but the vision/audio encoders do
**not** inherit the LLM's tensor/pipeline/data-parallel layout — they get
whatever layout fits their shape and memory profile.

See the parent [../README.md](../README.md) for the MegatronMIMO overview and
the design paper.

> These scripts are written as **single-node, 8-GPU smoke / parallelism tests**
> (100 iterations, frozen LLM by default). They exercise correctness of the
> hetero-MIMO data path, checkpoint loading, and the many parallelism layouts —
> they are not full pretraining configs.

---

## What's in this directory

| File | Purpose |
| --- | --- |
| [megatron_mimo_training_llava.py](megatron_mimo_training_llava.py) | Training entrypoint: **Vicuna-7B + CLIP** (image only). Defines model configs, answer-masked dataset, per-module checkpoint loading, parallelism config. |
| [megatron_mimo_training_llava_audio.py](megatron_mimo_training_llava_audio.py) | Training entrypoint: **Vicuna-7B + CLIP + Whisper** (image + audio). Imports shared pieces from the script above and adds the audio encoder/projector. |
| [run_hetero_llava.sh](run_hetero_llava.sh) | One-shot launcher for the image-only recipe (default LLM TP=4 + CLIP TP=4). |
| [run_hetero_llava_audio.sh](run_hetero_llava_audio.sh) | One-shot launcher for the image+audio recipe (default LLM TP=4 + CLIP TP=2 + Whisper TP=2). |
| [run_hetero_llava_parallelism_tests.sh](run_hetero_llava_parallelism_tests.sh) | Sweeps many LLM/vision parallelism layouts (image only), reports PASS/FAIL per config. |
| [run_hetero_llava_parallelism_tests_unfrozen_llm.sh](run_hetero_llava_parallelism_tests_unfrozen_llm.sh) | Same sweep with the **LLM unfrozen** (trains the language model). |
| [run_hetero_llava_audio_parallelism_tests.sh](run_hetero_llava_audio_parallelism_tests.sh) | Parallelism sweep for the **image+audio** (3-module) recipe. |
| [run_hetero_llava_audio_parallelism_tests_unfrozen_llm.sh](run_hetero_llava_audio_parallelism_tests_unfrozen_llm.sh) | 3-module sweep with the **LLM unfrozen**. |
| [run_conversion_verification.sh](run_conversion_verification.sh) | Converts CLIP + LLM at TP=1/2/4 and numerically verifies each against HuggingFace. |
| [convert_hf_clip_to_megatron.py](convert_hf_clip_to_megatron.py) | HF CLIP → per-TP-rank Megatron `.pt` checkpoint (+ loader helper). |
| [convert_hf_llama_to_megatron.py](convert_hf_llama_to_megatron.py) | HF Llama/Vicuna → per-TP-rank Megatron `.pt` checkpoint (+ loader helper). |
| [verify_clip_conversion.py](verify_clip_conversion.py) | Loads converted CLIP into Megatron, compares hidden states vs HF. |
| [verify_llama_conversion.py](verify_llama_conversion.py) | Loads converted LLM into Megatron, compares logits vs HF. |
| [prepare_llava_pretrain_audio.sh](prepare_llava_pretrain_audio.sh) | Builds the audio-augmented dataset (download LLaVA-Pretrain → TTS-synthesize speech → merge JSON). |
| [synthesize_llava_pretrain_audio.py](synthesize_llava_pretrain_audio.py) | Worker for the above: NeMo FastPitch + HiFiGAN TTS, sharded & resumable, with a `merge` mode. |
| [whisper/](whisper/) | Megatron-native Whisper **encoder** package (model, layer specs, converter, verifier). |

---

## Model recipe

| Module | Source model | Megatron module | Key shape | Notes |
| --- | --- | --- | --- | --- |
| Language model | `lmsys/vicuna-7b-v1.5` | `GPTModel` | 32 layers, hidden 4096, RMSNorm, RoPE, SwiGLU | Vocab padded to **32256** for LLaVA special tokens. Frozen by default. |
| Vision encoder | `openai/clip-vit-large-patch14-336` | `CLIPViTNoCLS` (CLIPViTModel subclass) | 23 layers (penultimate output), hidden 1024, `quick_gelu` | Drops the CLS token to match HF LLaVA `mm_vision_select_feature='patch'` → 576 patch tokens. **No PP > 1.** |
| Audio encoder | `openai/whisper-base` | `WhisperEncoder` (Megatron-native) | 6 layers, d_model 512, 80 mel bins | 30 s audio → 3000 mel frames → **1500** encoder tokens. **No PP > 1.** Audio recipe only. |
| Vision projector | — | `MultimodalProjector` (MLP) | 1024 → 4096 | Trained by default. |
| Audio projector | — | `MultimodalProjector` (MLP) | 512 → 4096 | Trained by default (audio recipe). |

- **Special token IDs:** image `32000`, audio `32002`.
- **Topology:** `{"images": ["language"], "audios": ["language"], "language": []}` —
  encoder embeddings are spliced into the LLM token stream at the placeholder positions.
- **Tokenizer:** `llava-hf/llava-1.5-7b-hf`. **Processors:** CLIP image processor and `whisper-base` feature extractor.
- **Loss masking:** `_AnswerMaskedMimoDataset` computes loss on the assistant
  (`gpt`) turn only, matching HF LLaVA's `preprocess_plain` contract. The human
  prompt and modality placeholders are masked out (`-100`).

---

## Prerequisites

- **Hardware:** a single node with **8 GPUs** — the launchers hard-code
  `GPUS_PER_NODE=8`, `NUM_NODES=1` (the parallelism sweeps also accept `--gpus 4`/`2`).
- **Container:** the project container, providing PyTorch, Transformer Engine,
  Megatron-Core (with MIMO support), and Megatron-Bridge
  (`megatron.bridge.models.megatron_mimo`). The `run_hetero_*` launchers run
  everything through **`uv run`**, so `uv` must be on `PATH`;
  `run_conversion_verification.sh`, the converters, and the dataset-prep scripts
  use plain `python`.
- **Audio recipe only:** NeMo's speech collections (`nemo-toolkit[tts]`).
  `prepare_llava_pretrain_audio.sh` auto-installs them from `/opt/NeMo[tts]`
  when missing (skip with `SKIP_NEMO_BOOTSTRAP=1`).
- **Network/disk:** the first run downloads the HF source models and
  LLaVA-Pretrain (≈100+ GB of images), so allow time and space. Point
  `DATASET_ROOT` / `CHECKPOINT_BASE_DIR` at existing copies to skip downloads.

## Quick start

**All `run_hetero_*` scripts are self-contained.** On every run they:

1. **Ensure the dataset is present** — if `DATASET_ROOT` is unset they download
   LLaVA-Pretrain (and, for the audio scripts, build the audio-augmented tree)
   into the default workspace directory; if it's already there, the step is
   skipped. Pass `DATASET_ROOT=/path/to/dataset` to point at an existing copy and
   skip the download/build entirely.
2. **Convert and cache the checkpoints** — they run the HF→Megatron converters
   for the requested TP sizes under `CHECKPOINT_BASE_DIR`, reusing any that are
   already cached.

Then they launch `torchrun`. Run inside the project container, on a node with 8 GPUs.

```bash
# Image-only: Vicuna-7B (TP=4, ranks 0-3) + CLIP (TP=4, ranks 4-7)
./run_hetero_llava.sh

# Image + audio: LLM (TP=4, ranks 0-3) + CLIP (TP=2, ranks 4-5) + Whisper (TP=2, ranks 6-7)
./run_hetero_llava_audio.sh
```

Common overrides (environment variables):

```bash
# Train the LLM instead of freezing it (also lowers the LR: 1e-3 → 1e-4)
UNFREEZE_LLM=1 ./run_hetero_llava.sh

# Reproducible run: FP32, unfused attention, deterministic NCCL/cuDNN/TE, no grad clip
DETERMINISTIC=1 ./run_hetero_llava.sh

# Point at an already-prepared dataset / checkpoint cache instead of downloading
DATASET_ROOT=/data/llava_pretrain \
CHECKPOINT_BASE_DIR=/data/mimo_ckpts \
./run_hetero_llava.sh

# Change the parallelism layout (must cover all 8 ranks, modules non-overlapping)
MIMO_LLM_TP=2 MIMO_LLM_DP=2 MIMO_VISION_TP=2 MIMO_VISION_DP=2 MIMO_VISION_OFFSET=4 \
./run_hetero_llava.sh
```

---

## Run scripts in detail

### `run_hetero_llava.sh` (image-only)

Default layout on 8 GPUs: **LLM TP=4 on ranks 0-3, CLIP TP=4 on ranks 4-7.**
The script:

1. **Prepares the dataset** — if `DATASET_ROOT` is empty, downloads
   `liuhaotian/LLaVA-Pretrain` (`blip_laion_cc_sbu_558k.json` + `images.zip`)
   into `DATASET_DOWNLOAD_DIR` and extracts the images. Skipped if already present.
2. **Converts checkpoints** — runs the CLIP and Llama converters for the
   configured TP sizes, caching results under `CHECKPOINT_BASE_DIR/clip_tp{N}`
   and `llm_tp{N}`. Reused if already cached.
3. **Launches training** — `torchrun --nproc_per_node 8` on
   `megatron_mimo_training_llava.py` with `micro-batch-size 4`,
   `global-batch-size 96`, `train-iters 100`.

The parallelism layout is passed to the training script purely through
`MIMO_*` environment variables (see [Configuration reference](#configuration-reference)).

### `run_hetero_llava_audio.sh` (image + audio)

Default layout: **LLM TP=4 (ranks 0-3), CLIP TP=2 (ranks 4-5), Whisper TP=2 (ranks 6-7).**
Same structure as above, plus:

- Builds the **audio-augmented dataset** by invoking
  `prepare_llava_pretrain_audio.sh` when `DATASET_ROOT` is empty. To keep the
  TTS step cheap, it synthesizes audio for only `TRAIN_ITERS * GLOBAL_BATCH_SIZE`
  records (override with `LIMIT`).
- Converts an extra **Whisper** checkpoint (`whisper_tp{N}`).
- Passes `--hf-data-files blip_laion_cc_sbu_558k_with_audio.json` and
  `--audio-column audio` so the audio encoder is actually exercised.

### Parallelism test sweeps

`run_hetero_llava_parallelism_tests.sh` and its audio counterpart run a **list
of named layouts** back-to-back, converting checkpoints as needed and printing a
PASS/FAIL summary table. They abort on the first failure (unless a single config
is selected).

```bash
./run_hetero_llava_parallelism_tests.sh                  # all 8-GPU configs
./run_hetero_llava_parallelism_tests.sh --gpus 4         # 4-GPU config set
./run_hetero_llava_parallelism_tests.sh --config tp2_dp2_both   # one config
./run_hetero_llava_parallelism_tests.sh --deterministic  # FP32 deterministic mode
```

Each config is a `|`-delimited tuple. For the **image-only** sweep:

```
name|llm_tp|llm_pp|llm_dp|llm_offset|vision_tp|vision_pp|vision_dp|vision_offset|mbs
```

The **audio** sweep inserts the audio module before `mbs`:

```
name|llm_tp|llm_pp|llm_dp|llm_offset|vision_tp|vision_pp|vision_dp|vision_offset|audio_tp|audio_pp|audio_dp|audio_offset|mbs
```

The sweeps cover symmetric splits (e.g. `tp4_both`, `tp2_dp2_both`),
LLM pipeline-parallel layouts (`pp4_llm_*`, `tp2_pp2_llm_*`), and **asymmetric**
GPU partitions (e.g. LLM on 2 GPUs, encoders sharing the other 6).

**`*_unfrozen_llm.sh` variants** run the same kind of sweep but with
`--freeze-llm False` (the LLM is trained) and a lower LR (`1e-3 → 1e-4`, min
`2e-5 → 1e-5`). The image-only unfrozen sweep also uses a trimmed config list
with `mbs` reduced to `2` for the larger memory footprint; the audio unfrozen
sweep keeps the same configs as its frozen counterpart.

### `run_conversion_verification.sh`

Standalone correctness check for the converters (no MIMO training). For each TP
size in `TP_SIZES` (default `1 2 4`) it converts CLIP and the LLM, then runs the
matching `verify_*_conversion.py` under `torchrun` to compare Megatron outputs
against HuggingFace. Exits non-zero on the first failure.

```bash
bash run_conversion_verification.sh
bash run_conversion_verification.sh --models llm --tp-sizes "2 4"
bash run_conversion_verification.sh --ckpt-root /scratch --dtype bf16
```

---

## Checkpoint converters

The three converters share the same design and on-disk layout:

```
{output}/tp_rank_00/model_weights.pt   →  {"model": {param_name: tensor, ...}}
{output}/tp_rank_01/model_weights.pt
...
```

They download HF weights, remap parameter names to Megatron's naming, **fuse Q/K/V
into Megatron's interleaved QKV layout**, fuse SwiGLU gate/up (LLM only), and
shard each tensor across `--tensor-parallel-size` ranks (column-parallel on
`dim=0`, row-parallel on `dim=1`). With `--use-te` they emit Transformer-Engine
layer names (fused LayerNorm inside `linear_qkv`/`linear_fc1`) and `_extra_state`
placeholders for FP8 compatibility.

| Converter | HF model | Notable handling |
| --- | --- | --- |
| [convert_hf_clip_to_megatron.py](convert_hf_clip_to_megatron.py) | CLIP ViT-L/14-336 | Drops `post_layernorm`; CLS handled at model level; `conv_bias=False`. |
| [convert_hf_llama_to_megatron.py](convert_hf_llama_to_megatron.py) | Llama/Vicuna-7B | GQA-aware QKV interleave; SwiGLU gate+up fusion; `--megatron-vocab-size` zero-pads embedding/LM-head (e.g. 32256). |
| [whisper/convert_hf_whisper_to_megatron.py](whisper/convert_hf_whisper_to_megatron.py) | Whisper-base | Encoder only; HF k_proj has **no bias** → zero-filled in the fused QKV bias; final `layer_norm` → `ln_post`. |

Each file also exposes a `load_megatron_*_weights()` helper that can load a
checkpoint saved at a **different TP size** (it merges all shards and re-splits
to the model's TP). The training scripts, however, use their own
`_load_tp_rank_weights()` hook, which loads the single matching shard and (for
PP > 1) remaps globally-numbered layer keys to each pipeline stage's local
indices. **PP > 1 needs no separate conversion** — only TP size keys the cache.

Convert standalone (run once on any single GPU):

```bash
python convert_hf_llama_to_megatron.py \
    --hf-model lmsys/vicuna-7b-v1.5 --output /ckpts/llm_tp4 \
    --tensor-parallel-size 4 --use-te --megatron-vocab-size 32256

python convert_hf_clip_to_megatron.py \
    --hf-model openai/clip-vit-large-patch14-336 --output /ckpts/clip_tp4 \
    --tensor-parallel-size 4 --use-te

python whisper/convert_hf_whisper_to_megatron.py \
    --hf-model openai/whisper-base --output /ckpts/whisper_tp2 \
    --tensor-parallel-size 2 --use-te --verify
```

### Verifiers

`verify_clip_conversion.py`, `verify_llama_conversion.py`, and
`whisper/verify_whisper_conversion.py` load the converted weights into the
Megatron model under `torchrun` (one rank per TP shard), run a fixed input, and
compare against the HF reference (mean/max abs diff + cosine similarity, with
tolerances that allow for TE-kernel numerics). They exit non-zero on mismatch.
`run_conversion_verification.sh` orchestrates the CLIP/LLM pair across TP sizes.

---

## Datasets

### LLaVA-Pretrain (image-only)

[`liuhaotian/LLaVA-Pretrain`](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
— 558k BLIP-LAION-CC-SBU caption pairs:

- `blip_laion_cc_sbu_558k.json` — records with `id`, `image` (e.g. `00453/004531425.jpg`),
  and a `conversations` list (`human`/`gpt` turns).
- `images.zip` — extracts to 5-digit shard directories (`00000/`, `00001/`, … ≈ 107 GB)
  at the dataset root, which is where the JSON's relative image paths resolve.

The run scripts auto-download and extract this when `DATASET_ROOT` is empty
(into `DATASET_DOWNLOAD_DIR`, default `/workspace/llava_pretrain`); both steps
are skipped if already present.

### Audio-augmented dataset

`prepare_llava_pretrain_audio.sh` builds a speech-augmented copy of
LLaVA-Pretrain for the audio recipe. It writes a fresh tree under
`AUGMENTED_DATASET_DIR` (default `/workspace/llava_pretrain_audio_augmented`):

```
blip_laion_cc_sbu_558k.json             # records with image paths absolutised
blip_laion_cc_sbu_558k_with_audio.json  # same records + an "audio" field  ← consumed by training
audio/<prefix>/<id>.flac                # synthesized 16 kHz mono FLACs
audio_manifest/shard_NNNNN.jsonl        # per-shard manifest
shard_logs/shard_NNNNN.log              # per-shard stdout+stderr
```

Pipeline:

1. **Bootstrap NeMo speech** — `pip install -e /opt/NeMo[tts]` if
   `nemo.collections.tts` isn't importable (the LLM/Multimodal containers ship
   NeMo source without the speech collections). Disable with `SKIP_NEMO_BOOTSTRAP=1`.
2. **Prepare LLaVA-Pretrain** — download/extract as above (shared logic).
3. **Absolutise image paths** in the JSON so the 107 GB image tree need not be
   copied; honors `LIMIT` (first-N records).
4. **Resolve TTS models** — FastPitch + HiFiGAN `.nemo` files (curl'd from the
   NeMo registry into `TTS_CACHE`, or supplied via `TTS_NEMO`/`VOCODER_NEMO`).
5. **Synthesize** — launches `synthesize_llava_pretrain_audio.py` once per GPU
   (`NUM_SHARDS`, default = visible GPUs), each bound via `CUDA_VISIBLE_DEVICES`.
   **Resume-safe** — existing non-empty FLACs are skipped.
6. **Merge** — `--mode merge` joins the per-shard manifests into
   `blip_laion_cc_sbu_558k_with_audio.json`.

> ⚠️ `--limit` is applied **per shard**, so `LIMIT=1000` with `NUM_SHARDS=8`
> produces ~8000 samples. Use `NUM_SHARDS=1` for an exact count.

`synthesize_llava_pretrain_audio.py` pulls the first `human` turn from each
record, strips `<image>`/`<audio>` tokens, synthesizes 16 kHz mono speech
(FastPitch → HiFiGAN, resampled, capped at 30 s), and writes a FLAC plus a
manifest line. Audio is loaded as a numpy array at train time; non-16 kHz files
are rejected.

---

## Training entrypoints

Both `megatron_mimo_training_llava*.py` scripts follow the same flow:
initialize NCCL → build per-module model specs and the
`MegatronMIMOParallelismConfig` → build the answer-masked HF data provider →
register the per-module checkpoint-loading hook → assemble a `ConfigContainer`
→ call `pretrain_megatron_mimo()`.

Notes on the implementation worth knowing:

- **No global `parallel_state.initialize_model_parallel()`** — MIMO manages its
  own parallelism via `HyperCommGrids` / `pg_collection`. Each rank only
  materializes the module(s) it participates in; the checkpoint hook guards
  every load with an existence check so it's safe to call on all ranks.
- **`_wrap_iter`** rewraps data-loader batches into encoder-keyed inputs
  (`images → {"clip": {"x": ...}}`, `audios → {"whisper": {"input_features": ...}}`),
  moves them to GPU, and casts to the model dtype. The audio path additionally
  computes per-sample valid encoder-output lengths from the mel spectrogram
  (padding frames are zero), trims surplus audio placeholder tokens, and passes
  `seq_lengths` to the encoder.
- **PP propagation** — the LLM's PP size is copied into its `TransformerConfig`
  so Megatron builds the right per-stage layer count/offset.
- **Checkpointing** — `torch_dist` format, fully parallel/reshardable save;
  `save_rng=False` (MIMO RNG save is not yet supported upstream).

`megatron_mimo_training_llava_audio.py` reuses the shared LLaVA configs and
`_load_tp_rank_weights` from the image-only script and adds the Whisper encoder,
the audio projector, the `audios` modality, and the audio-aware data path.

### Key training arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--dataset-root` | *(required)* | LLaVA-Pretrain (or audio-augmented) root. |
| `--micro-batch-size` / `--global-batch-size` | 1 / 1 | `num_microbatches = GBS / MBS`. MBS must be divisible by every module's DP. |
| `--train-iters` | 2 | Training iterations. |
| `--language-model-checkpoint` | None | Converted LLM checkpoint dir (loaded into the LLM only). |
| `--vision-encoder-checkpoint` | None | Converted CLIP checkpoint dir. |
| `--audio-encoder-checkpoint` | None | Converted Whisper checkpoint dir (audio script). |
| `--freeze-llm` / `--freeze-vision` / `--freeze-projector` | True / True / False | Per-module freeze flags. |
| `--hf-data-files` | `blip_laion_cc_sbu_558k.json` | JSON under `--dataset-root` (audio script). |
| `--audio-column` | None | Enables the audio encoder when set (e.g. `audio`). |
| `--deterministic` | off | FP32 + unfused attention + deterministic kernels (slower, reproducible). |
| `--lr` / `--min-lr` / `--lr-warmup-iters` / `--clip-grad` | — | Optimizer/scheduler knobs. |

---

## `whisper/` package

A self-contained, **Megatron-native Whisper encoder** (the audio recipe does not
depend on HF Whisper at runtime):

- [whisper/whisper_model.py](whisper/whisper_model.py) — `WhisperEncoder`:
  two 1-D convs (the second stride-2, halving the time axis), frozen sinusoidal
  position embeddings, a `TransformerBlock`, and a final `ln_post`. `forward()`
  optionally accepts `seq_lengths` to drop padding-derived output tokens.
- [whisper/whisper_layer_specs.py](whisper/whisper_layer_specs.py) — TE and
  local layer specs (no-mask self-attention; identity `pre_mlp_layernorm` since
  TE fuses it into `linear_fc1`).
- [whisper/convert_hf_whisper_to_megatron.py](whisper/convert_hf_whisper_to_megatron.py) — converter + loader (+ shape verifier).
- [whisper/verify_whisper_conversion.py](whisper/verify_whisper_conversion.py) — numerical HF-vs-Megatron check.

---

## Configuration reference

All run scripts read these environment variables (sensible defaults shown):

**Parallelism (per module; `TP*PP*DP` must equal that module's GPU count):**

| Variable | Default (`run_hetero_llava.sh` / `_audio.sh`) | Meaning |
| --- | --- | --- |
| `MIMO_LLM_TP` / `_PP` / `_DP` / `_OFFSET` | 4 / 1 / 1 / 0 | LLM layout and first global rank. |
| `MIMO_VISION_TP` / `_PP` / `_DP` / `_OFFSET` | 4 (audio: 2) / 1 / 1 / 4 | Vision encoder layout. **PP must be 1.** |
| `MIMO_AUDIO_TP` / `_PP` / `_DP` / `_OFFSET` | 2 / 1 / 1 / 6 | Audio encoder layout (audio script). **PP must be 1.** |

**Dataset / checkpoints / behavior:**

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATASET_ROOT` | *(empty → auto-download)* | Existing dataset root; empty triggers download/build. |
| `DATASET_DOWNLOAD_DIR` | `/workspace/llava_pretrain` | Where LLaVA-Pretrain is downloaded. |
| `LLAVA_PRETRAIN_REPO` | `liuhaotian/LLaVA-Pretrain` | HF dataset repo. |
| `AUDIO_DATASET_DIR` | `/workspace/llava_pretrain_audio_augmented` | Audio-augmented output tree. |
| `HF_DATA_FILES` | `blip_laion_cc_sbu_558k_with_audio.json` | JSON consumed by the audio recipe. |
| `AUDIO_COLUMN` | `audio` | Column that enables the audio encoder. |
| `LIMIT` / `NUM_SHARDS` | `iters*gbs` / #GPUs | TTS synthesis sizing (per-shard limit!). |
| `CHECKPOINT_BASE_DIR` | `/workspace/megatron_mimo_checkpoints` | Converted-checkpoint cache (keyed by TP size). |
| `HF_LLM_MODEL` / `HF_VISION_MODEL` / `HF_AUDIO_MODEL` | vicuna-7b-v1.5 / clip-vit-large-patch14-336 / whisper-base | Source HF models. |
| `MEGATRON_VOCAB_SIZE` | 32256 | LLM vocab padding target. |
| `UNFREEZE_LLM` | 0 | `1` trains the LLM and lowers the LR. |
| `DETERMINISTIC` | 0 | `1` enables FP32/deterministic mode + Ring NCCL + no grad clip. |

---

## Parallelism rules

For every layout (enforced or assumed by the scripts):

- For each module, `TP * PP * DP` equals that module's GPU count.
- Modules occupy **non-overlapping** GPU sets, and together they cover **all**
  `GPUS_PER_NODE` ranks. `*_OFFSET` is the first global rank of each module.
- The **vision encoder (CLIPViT) and audio encoder (Whisper) do not support PP > 1** —
  only the LLM may use pipeline parallelism.
- `micro-batch-size` must be divisible by **every** module's DP size.
- For the 3-module audio sweep, **encoder DP must be ≥ LLM DP** (required for
  embedding alignment across batches).
- DP sub-sharding is handled per-module inside the forward step, so the config
  itself sets `data_parallel_size = 1` (all data-loading ranks see identical
  global micro-batches).

## Deterministic mode

`DETERMINISTIC=1` (or `--deterministic`) trades speed for reproducibility: FP32
precision, unfused attention, disabled CE-loss fusion, full activation
recompute, deterministic torch/cuDNN/TE algorithms, NCCL pinned to `Ring`/`Simple`,
and **gradient clipping disabled** (`clip-grad=0`, since the all-reduce of norms
is non-associative). W&B experiment names get a `-fp32` suffix.

## Known limitations

- **Single node only** — the launchers hard-code `NUM_NODES=1`; multi-node would
  need a different launch wrapper.
- **No pipeline parallelism for the encoders** — only the LLM may set PP > 1
  (CLIPViT and Whisper do not support it).
- **Smoke-test scale** — defaults run 100 iterations with the LLM frozen; treat
  these as correctness / parallelism tests, not converged pretraining recipes.
