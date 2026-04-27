# Nemotron-3 Nano Omni Examples

This directory contains example scripts for **Nemotron-3 Nano Omni**, a 30B-A3B
MoE multimodal model that jointly processes image, video, audio, and text
inputs. It pairs a MoE Mamba/attention hybrid language backbone with a RADIO
vision tower (static-resolution image path or dynamic-resolution temporal video
embedder) and a Parakeet sound encoder.

| Model | HF ID | Architecture |
|---|---|---|
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | MoE hybrid LM (Mamba+attn) + RADIO vision + Parakeet audio |

> **Verified hardware:** all conversion, inference, and training flows in
> this directory have been verified on **NVIDIA H100 80GB** nodes with 8
> GPUs per node. Other GPU SKUs may work but have not been tested.

## Workspace Configuration

All scripts in this directory use a `WORKSPACE` environment variable as the
base directory for checkpoints, datasets, and results, and `HF_MODEL_ID` as
the source HF model ID. Defaults:

```bash
export WORKSPACE=/workspace
export HF_MODEL_ID=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
```

The scripts expect the following layout:

- `${WORKSPACE}/models/<model-name>/` — converted Megatron checkpoint (created by `conversion.sh`)
- `${WORKSPACE}/models/<model-name>-hf-export/` — re-exported HF checkpoint
- `${WORKSPACE}/datasets/valor32k_avqa/energon/` — VALOR32K-AVQA Energon shards
- `${WORKSPACE}/assets/` — image / video / audio files used for `inference.sh` (auto-downloaded from the public HF model card on first run)
- `${WORKSPACE}/results/` — training outputs (checkpoints, tensorboard logs)

## Day-0 Code

Use the NeMo 26.04 container as the base image: `nvcr.io/nvidia/nemo:26.04`.

The Day-0 code lives on the following public branches:

| Repo | Branch | Remote |
|---|---|---|
| Megatron-Bridge | [`nemotron_3_omni`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/nemotron_3_omni) | `https://github.com/NVIDIA-NeMo/Megatron-Bridge.git` |
| Megatron-LM (submodule at `3rdparty/Megatron-LM`) | [`nemotron_3_omni`](https://github.com/NVIDIA/Megatron-LM/tree/nemotron_3_omni) | `https://github.com/NVIDIA/Megatron-LM.git` |

```bash
cd $WORKSPACE
git clone -b nemotron_3_omni https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git submodule update --init --recursive --depth 1
uv lock
uv sync
```

The `.gitmodules` already points the `3rdparty/Megatron-LM` submodule at
`https://github.com/NVIDIA/Megatron-LM.git`, and the recorded gitlink is
the tip of its `nemotron_3_omni` branch — `git submodule update --init
--recursive --depth 1` checks it out as a shallow clone automatically; no
extra remote/fetch step needed. The `--depth 1` flag dramatically reduces
clone time and disk usage (avoids fetching Megatron-LM's full history).
`uv lock` regenerates `uv.lock` so `megatron-core` resolves to the cloned
`3rdparty/Megatron-LM/` submodule rather than any pre-installed copy from
the container. `uv sync` then materializes the resulting environment.

> **`uv lock && uv sync` are mandatory before running any script in this
> repo.** The `uv.lock` in this repo pins `flashinfer-python==0.6.8.post1`
> to match the `flashinfer-cubin` pre-installed in the NeMo container.
> Skipping `uv sync` will leave a stale version installed, producing:
> ```
> RuntimeError: flashinfer-cubin version (0.6.8.post1) does not match flashinfer version (X).
> ```
> Re-run `uv sync` from `$WORKSPACE/Megatron-Bridge` to resolve it.

Verify that `megatron.core` and `megatron.bridge` resolve to the cloned
checkout (and not a pre-installed copy from the container):

```bash
uv run python -c "
import megatron.core, megatron.bridge
print('core:', megatron.core.__path__)
print('bridge:', megatron.bridge.__path__)
"
```

Expected output (replace `$WORKSPACE` with your actual value, e.g. `/workspace`):

```
core: ['$WORKSPACE/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core']
bridge: ['$WORKSPACE/Megatron-Bridge/src/megatron/bridge']
```

If either path points elsewhere (e.g. a site-packages location inside
the container), `uv` is resolving against a stale environment — re-run
`uv sync` from `$WORKSPACE/Megatron-Bridge` before continuing.

## Checkpoint Conversion

[conversion.sh](conversion.sh) covers HF → Megatron import, Megatron → HF
export, and a multi-GPU HF↔Megatron round-trip verification.

- **Import** writes `iter_0000000/`, `latest_train_state.pt`, and
  `latest_checkpointed_iteration.txt` under `${WORKSPACE}/models/<model-name>`.
  `--trust-remote-code` is required because the HF architecture
  (`NemotronH_Nano_Omni_Reasoning_V3`) ships custom modeling code.
- **Export** runs with `--not-strict`, which permits 4 expected-missing
  tensors (regenerated from config on the HF side):
  `sound_encoder.encoder.feature_extractor.featurizer.{fb,window}` and
  `vision_model.radio_model.input_conditioner.{norm_mean,norm_std}`.
- **Round-trip** loads HF → Megatron (TP=2, EP=2) and re-exports back to HF,
  diffing every tensor; all weights should match (✅) and the same 4
  expected-missing tensors are reported on re-export. The re-exported HF
  checkpoint is written under
  `${WORKSPACE}/models/<model-name>-hf-export/<model-name>/` (the round-trip
  script appends `<model-name>` to `--output-dir`, so it nests inside the
  same `-hf-export` tree as the standalone export step rather than
  overwriting the Megatron checkpoint).

> **Note:** During the export and round-trip steps you may see a line like
> `[ERROR] cache_position is part of NemotronHForCausalLM.forward's signature, but not documented.`
> This is a non-fatal docstring linter warning emitted by the `transformers`
> library on the model's custom `forward` signature. It does not indicate a
> conversion error; the script continues and completes successfully.

Run from `$WORKSPACE/Megatron-Bridge`:

```bash
bash examples/models/vlm/nemotron_3_omni/conversion.sh
```

## Inference

[inference.sh](inference.sh) drives
`examples/conversion/hf_to_megatron_generate_nemotron_omni.py` over the four
modality combinations exercised by the model:

| # | Modality | GPUs | Parallelism |
|---|---|---|---|
| 1 | Image + Text | 1 | — |
| 2 | Video + Text | 8 | TP=4, EP=4 |
| 3 | Audio + Text | 1 | — |
| 4 | Video + Audio + Text | 8 | TP=4, EP=2 |

The default assets are pulled automatically from
[here](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/tree/main/images)
on the first run. `curl` and `ffmpeg` must be available. The NeMo 26.04
container does **not** include `ffmpeg` — install it before running
`inference.sh`:

```bash
apt-get install -y ffmpeg
```

Override `IMAGE_PATH` / `VIDEO_PATH` / `AUDIO_PATH` with your own assets to
use different inputs; omit `--megatron_model_path` (set `MEGATRON_PATH=""`)
to convert HF → Megatron on the fly instead of reusing the imported
checkpoint.

**Expected outputs** (from the default HF demo assets):

- *Image + Text* — `Describe this image.` on `table.png` → "This image
  displays a table titled 'Technical Specifications' that compares the
  features of two NVIDIA GPU models: the H100 SXM and the H100 NVL. The
  table is organized into rows, each detailing a specific technical aspect,
  with columns for the H100 SXM and H100 NVL models. Key specifications
  compared include: Performance (FLOPS): The H100 SXM generally has higher
  performance …" (continues with full spec breakdown).
- *Video + Text* — `Describe what you see.` on `demo.mp4` → "A variety of
  plants with green leaves and some with red leaves are shown. Some plants
  have white flowers."
- *Audio + Text* — `Transcribe the audio.` on the audio track extracted
  from `demo.mp4` → "Hey guys, today I want you to have a look at all these
  vibrant colors and lush greenery. And when you move closer, see these
  delicate purple flowers peeking through the leaves. It's a perfect spot to
  relax and connect with the nature."
- *Video + Audio + Text* — `Describe the video and audio.` on `demo.mp4`
  with its extracted audio track → "A person is speaking while the camera
  pans over a field of plants with green, red, and white leaves and
  flowers."

Run from `$WORKSPACE/Megatron-Bridge`:

```bash
bash examples/models/vlm/nemotron_3_omni/inference.sh
```

## Training

All training scripts use the Nemotron-3-Nano-Omni-30B-A3B-Reasoning
pretrained checkpoint and enable in-batch sequence packing via
`dataset.pack_sequences_in_batch=True`. Default GPU layout per script:

- **Full SFT** — 2 nodes / 16 GPUs (full optimizer state for ~33 B params)
- **LoRA PEFT** — 1 node / 8 GPUs

The world size required by Megatron is
`PP * max(TP*CP, EP*ETP)`, *not* `PP * TP * EP * CP * ETP`. With TP=2 EP=8
CP=1 PP=1 ETP=1 that means `max(2, 8) = 8` GPUs are sufficient — LoRA fits
in one node because it only trains the adapters and skips the full Adam
state.

Before submitting, set these environment variables (the scripts inherit them
through `srun`):

1. `CONTAINER_IMAGE` — registry URI or local path to the training container image (e.g. `nvcr.io/nvidia/nemo:26.04`); required, set inside the script
2. `HF_TOKEN` — to pull the HF model config/tokenizer
3. `HF_HOME` — optional, to share the HF cache across jobs
4. `WANDB_API_KEY` — optional, to enable WandB logging

The two task flavors below are orthogonal — pick whichever dataset/modality
combo matches your target task and either full-parameter (SFT) or LoRA
(PEFT).

### Image-Text — CORD-V2

[CORD-V2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) is a
document-image parsing dataset (restaurant receipts → structured JSON). The
vision path uses one embedding per frame (`temporal_patch_dim=1`, no
temporal video embedder); `dynamic_resolution=True` is inherited from the
base config. Recipe base: `nemotron_omni_cord_v2_*_config` in
`src/megatron/bridge/recipes/nemotron_omni/nemotron_omni.py`.

| Mode | Script | Recipe |
|---|---|---|
| Full SFT | [slurm_sft_cord_v2.sh](slurm_sft_cord_v2.sh) | `nemotron_omni_cord_v2_sft_config` |
| LoRA | [slurm_peft_cord_v2.sh](slurm_peft_cord_v2.sh) | `nemotron_omni_cord_v2_peft_config` |

Parallelism (both): TP=2, EP=8, CP=1, MBS=2, GBS=16, packed sequences,
selective recompute. LoRA targets `linear_qkv`, `linear_proj`, `in_proj`,
`out_proj` (LM attention + Mamba projections); vision / sound encoders +
projections frozen.

```bash
sbatch examples/models/vlm/nemotron_3_omni/slurm_sft_cord_v2.sh
sbatch examples/models/vlm/nemotron_3_omni/slurm_peft_cord_v2.sh
```

### Audio-Video-Text — VALOR32K-AVQA

[VALOR32K-AVQA](https://inesriahi.github.io/valor32k-avqa-2/) is an audio-visual
multiple-choice QA dataset. This path exercises the temporal video
embedder: frames are fused in pairs (`temporal_patch_dim=2`,
`separate_video_embedder=True`) and audio is fed through the Parakeet
encoder. Recipe base: `nemotron_omni_valor32k_*_config`.

Prepare the Energon shards once:

```bash
uv run python examples/models/vlm/nemotron_3_omni/data/build_valor32k_avqa_shards.py \
  --output_dir ${WORKSPACE}/datasets/valor32k_avqa
```

| Mode | Script | Recipe |
|---|---|---|
| Full SFT | [slurm_sft_valor32k_avqa.sh](slurm_sft_valor32k_avqa.sh) | `nemotron_omni_valor32k_sft_config` |
| LoRA | [slurm_peft_valor32k_avqa.sh](slurm_peft_valor32k_avqa.sh) | `nemotron_omni_valor32k_peft_config` |

Parallelism (both): TP=2, EP=8, CP=1, MBS=2, packed sequences, selective
recompute. SFT uses GBS=16 and the recipe-default LR; LoRA uses GBS=64 and
LR=1e-4 (adapters target the language model only; vision encoder, vision
projection, sound encoder, and sound projection are frozen).

```bash
sbatch examples/models/vlm/nemotron_3_omni/slurm_sft_valor32k_avqa.sh
sbatch examples/models/vlm/nemotron_3_omni/slurm_peft_valor32k_avqa.sh
```

### Expected Training Dynamics

We provide a [Weights & Biases report](https://api.wandb.ai/links/nvidia-nemo-fw-public/5tdqkrmq) for the expected loss curves and grad norms.

## Evaluation

After training, two batch-inference scripts are provided to spot-check the
finetuned Megatron checkpoint on the same datasets used for training:

| Dataset | Script | Output |
|---|---|---|
| CORD-V2 | [cord_v2_inference.py](cord_v2_inference.py) | JSON of `{prompt, gold, prediction}` per sample plus image bytes for eyeballing |
| VALOR32K-AVQA | [valor32k_avqa_inference.py](valor32k_avqa_inference.py) | Per-sample predictions and an aggregate multiple-choice accuracy |

Example invocations (8 GPUs, single node). The slurm scripts tag
`OUTPUT_DIR` with the run config (`<recipe>_<sft|lora>_<RUN_TAG>`, where
`RUN_TAG=tp${TP}etp${ETP}ep${EP}cp${CP}pack${PACKED_SEQ}_iter${TRAIN_ITERS}`),
so substitute the matching `RUN_TAG` from your training job into
`--megatron_model_path` below:

```bash
uv run torchrun --nproc-per-node=8 \
  examples/models/vlm/nemotron_3_omni/cord_v2_inference.py \
    --hf_model_path "$HF_MODEL_ID" \
    --megatron_model_path ${WORKSPACE}/results/nemotron_omni_cord_v2_sft_config_sft_<RUN_TAG>/checkpoints \
    --tp 4 --ep 2 \
    --max_samples 100 \
    --output ${WORKSPACE}/results/cord_v2_eval.json

uv run torchrun --nproc-per-node=8 \
  examples/models/vlm/nemotron_3_omni/valor32k_avqa_inference.py \
    --hf_model_path "$HF_MODEL_ID" \
    --megatron_model_path ${WORKSPACE}/results/nemotron_omni_valor32k_sft_config_sft_<RUN_TAG>/checkpoints \
    --data_root ${WORKSPACE}/datasets/valor32k_avqa \
    --tp 4 --ep 2 \
    --max_samples 500 \
    --output ${WORKSPACE}/results/valor32k_eval.json
```

> **These scripts are intentionally simple and run one sample at a time —
> they are very slow and only intended as sanity checks of the trained
> checkpoint.** For real inference / serving (batched, KV-cached,
> production-grade throughput), please use vLLM with the re-exported HF
> checkpoint produced by `conversion.sh` (`Megatron → HF` export step)
> instead of these scripts.

## LoRA Merge

After LoRA training, merge adapter weights back into the base Megatron
checkpoint. The script reads the base checkpoint path from
`run_config.yaml` inside the LoRA checkpoint directory, so `--pretrained`
is usually not required. Pass `--tp` to match the parallelism of the base
checkpoint.

Run from `$WORKSPACE/Megatron-Bridge`:

```bash
uv run torchrun --nproc-per-node=<NUM_GPUS> examples/peft/merge_lora.py \
    --lora-checkpoint <LORA_CHECKPOINT_DIR>/iter_<NNNNNNNN> \
    --hf-model-path "$HF_MODEL_ID" \
    --output <MERGED_OUTPUT_DIR> \
    --tp <TP_SIZE>
```

The merged checkpoint is a standard Megatron checkpoint and can be used
directly for inference or re-exported to HF format using the standard
export flow (see the Checkpoint Conversion section).

If the node does not have enough GPU memory, add `--cpu` to load and merge
entirely on CPU (no GPU required, but slower).

### LoRA Adapter Export

Export LoRA adapter weights to HuggingFace PEFT format
(`adapter_config.json` + `adapter_model.safetensors`, ~50 MB). This
lightweight format can be shared and loaded with the `peft` library
without distributing the full base model.

```bash
uv run python examples/conversion/adapter/export_adapter.py \
    --hf-model-path "$HF_MODEL_ID" \
    --lora-checkpoint <LORA_CHECKPOINT_DIR>/iter_<NNNNNNNN> \
    --output <ADAPTER_OUTPUT_DIR> \
    --trust-remote-code
```

The output directory contains:

- `adapter_config.json` — LoRA configuration (rank, alpha, target modules)
- `adapter_model.safetensors` — adapter weights only (~50 MB)
