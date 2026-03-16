# WAN 2.1 Examples

This directory contains example scripts for [WAN 2.1](https://github.com/Wan-Video/Wan2.1) (text-to-video/image) with Megatron-Bridge: dataset preparation, checkpoint conversion, pretraining, and inference. Built on [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) and [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), it supports advanced parallelism strategies (data, tensor, sequence, and context parallelism) and optimized kernels (e.g., Transformer Engine fused attention).

All commands below assume you run them from the **Megatron-Bridge repository root** unless noted. Use `uv run` when you need the project's virtualenv (e.g. `uv run python ...`, `uv run torchrun ...`).

## Workspace Configuration

Use a `WORKSPACE` environment variable as the base directory for checkpoints and results. Default is `/workspace`. Override it if needed:

```bash
export WORKSPACE=/your/custom/path
```

Suggested layout:

- `${WORKSPACE}/checkpoints/wan/` – Megatron WAN checkpoints (after import)
- `${WORKSPACE}/checkpoints/wan_hf/` – Hugging Face WAN model (download or export)
- `${WORKSPACE}/datasets/wan/` – Processed WebDataset shards
- `${WORKSPACE}/results/wan/` – Training outputs (pretrain)

---

## 1. Dataset Preparation

This recipe uses NVIDIA's [Megatron-Energon](https://github.com/NVIDIA/Megatron-Energon) as an efficient multi-modal data loader. Datasets should be in the WebDataset-compatible format (typically sharded `.tar` archives). Energon supports large-scale distributed loading, sharding, and sampling for video-text and image-text pairs. Set `dataset.path` to your WebDataset directory or shard pattern. See Megatron-Energon docs for format details, subflavors, and advanced options.

If you do not have a dataset yet or only need to validate performance/plumbing, see the "Quick Start with Mock Dataset" section under Pretraining below.

### Preparation example

Starting with a directory containing raw `.mp4` videos and their corresponding `.json` metadata files containing captions, you can turn the data into WAN-ready WebDataset shards using the helper script [prepare_dataset_wan.py](prepare_dataset_wan.py). We then use Energon to process those shards and create its metadata. After this, you can set the training script's `dataset.path` argument to the output processed data folder and start training.

```bash
# 1) Define your input (raw videos) and output (WebDataset shards) folders
DATASET_SRC=/opt/raw_videos           # contains .mp4 and per-video .jsonl captions
DATASET_PATH=${WORKSPACE}/datasets/wan # output WebDataset shards

# 2) (Optional) If your WAN models require auth on first download
export HF_TOKEN=<your_huggingface_token>

# 3) Create WAN shards with latents + text embeddings
#    WAN's VAE encoder and T5 encoder extract video latents and caption embeddings
#    offline before training. Key arguments:
#      --height/--width: resize target (832x480 supported for both 1.3B and 14B)
#      --center-crop: center crop to exact target size after resize
#      --mode: process "video" or "frames" of video
uv run torchrun --nproc_per_node=8 \
  examples/diffusion/recipes/wan/prepare_dataset_wan.py \
  --video_folder "${DATASET_SRC}" \
  --output_dir "${DATASET_PATH}" \
  --output_format energon \
  --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --mode video \
  --height 480 \
  --width 832 \
  --resize_mode bilinear \
  --center-crop

# 4) Use Energon to process shards and create its metadata/spec
energon prepare "${DATASET_PATH}"
# In the interactive prompts:
# - Enter a train/val/test split, e.g., "8,1,1"
# - When asked for the sample type, choose: "Crude sample (plain dict for cooking)"
```

**What gets produced:**

- Each shard contains:
  - `pth`: WAN video latents
  - `pickle`: text embeddings
  - `json`: side-info (text caption, sizes, processing choices, etc.)
- Energon writes a `.nv-meta` directory with dataset info and a `dataset.yaml` you can version-control.

In the training config, point `dataset.path` to the processed data output directory: `dataset.path=${DATASET_PATH}`.

---

## 2. Checkpoint Conversion

The script [conversion/convert_checkpoints.py](conversion/convert_checkpoints.py) converts between Hugging Face (diffusers) and Megatron checkpoint formats.

**Source model:** [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) (or a local clone).

### Download the Hugging Face model (optional)

If you want a local copy before conversion:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir ${WORKSPACE}/checkpoints/wan_hf/wan2.1 \
  --local-dir-use-symlinks False
```

### Import: Hugging Face → Megatron

Convert a Hugging Face WAN model to Megatron format:

```bash
uv run python examples/diffusion/recipes/wan/conversion/convert_checkpoints.py import \
  --hf-model ${WORKSPACE}/checkpoints/wan_hf/wan2.1 \
  --megatron-path ${WORKSPACE}/checkpoints/wan/wan2.1
```

The Megatron checkpoint is written under `--megatron-path` (e.g. `.../wan2.1/iter_0000000/`). Use that path for inference and fine-tuning.

### Export: Megatron → Hugging Face

Export a Megatron checkpoint back to Hugging Face (e.g. for use in diffusers). You must pass the **reference** HF model (for config and non-DiT components) and the **Megatron iteration directory**:

```bash
uv run python examples/diffusion/recipes/wan/conversion/convert_checkpoints.py export \
  --hf-model ${WORKSPACE}/checkpoints/wan_hf/wan2.1 \
  --megatron-path ${WORKSPACE}/checkpoints/wan/wan2.1/iter_0000000 \
  --hf-path ${WORKSPACE}/checkpoints/wan_hf/wan2.1_export
```

**Note:** The exported directory contains only the DiT transformer weights. For a full pipeline (VAE, text encoders, etc.), duplicate the original HF checkpoint directory and replace the `transformer` folder with the exported one.

---

## 3. Pretraining

The script [pretrain_wan.py](pretrain_wan.py) runs WAN pretraining with Hydra-style YAML config and CLI overrides. Example configs for 1.3B and 14B are provided under [`conf/`](conf/) (see `wan_1_3B.yaml` and `wan_14B.yaml`).

### Sequence packing

This recipe leverages sequence packing to maximize throughput. When a batch contains videos with different shapes or resolutions, naive batching with padding wastes significant computation on padded tokens. Sequence packing stacks multiple samples (with different resolutions) into a single sequence instead. When using sequence packing:

- Set `train.micro_batch_size=1` and `dataset.micro_batch_size=1`
- Ensure `model.qkv_format=thd` (required with context parallelism and recommended with sequence packing)

### Training mode

The script exposes a `--training-mode` flag with `pretrain` and `finetune` presets for flow-matching hyperparameters. Pretraining uses noisier, biased sampling (e.g., logit-normal, higher `logit_std`, lower `flow_shift`) for stability and broad learning, while finetuning uses uniform, lower-noise settings (e.g., uniform sampling, lower `logit_std`, higher `flow_shift`) to refine details and improve quality.

### Quick start with mock dataset

If you want to run without a real dataset (for debugging or performance measurement), pass `--mock`:

```bash
uv run torchrun --nproc_per_node=8 examples/diffusion/recipes/wan/pretrain_wan.py \
  --config-file examples/diffusion/recipes/wan/conf/wan_1_3B.yaml \
  --training-mode pretrain \
  --mock
```

You may adjust mock shapes (`F_latents`, `H_latents`, `W_latents`) and packing behavior (`number_packed_samples`) in `WanMockDataModuleConfig` (see `src/megatron/bridge/diffusion/recipes/wan/wan.py`) to simulate different data scenarios.

### With a custom config

Copy one of the provided configs and edit it with your settings:

```bash
cp examples/diffusion/recipes/wan/conf/wan_1_3B.yaml examples/diffusion/recipes/wan/conf/my_wan.yaml
# Edit my_wan.yaml to set:
# - dataset.path: Path to your WebDataset directory
# - train.global_batch_size/micro_batch_size: Keep micro_batch_size=1
# - model.tensor_model_parallel_size / model.context_parallel_size: Based on GPUs
# - checkpoint.save and checkpoint.load: Checkpoint directory
```

Then run:

```bash
uv run torchrun --nproc_per_node=$NUM_GPUS examples/diffusion/recipes/wan/pretrain_wan.py \
  --training-mode pretrain \
  --config-file examples/diffusion/recipes/wan/conf/my_wan.yaml
```

### With CLI overrides

You can also override any config values from the command line:

```bash
uv run torchrun --nproc_per_node=$NUM_GPUS examples/diffusion/recipes/wan/pretrain_wan.py \
  --config-file examples/diffusion/recipes/wan/conf/my_wan.yaml \
  --training-mode pretrain \
  dataset.path=${WORKSPACE}/datasets/wan \
  train.global_batch_size=8 \
  train.micro_batch_size=1 \
  model.tensor_model_parallel_size=2 \
  model.context_parallel_size=4 \
  checkpoint.save=${WORKSPACE}/checkpoints/wan/wan2.1 \
  checkpoint.load=${WORKSPACE}/checkpoints/wan/wan2.1
```

**Note**: If you use `logger.wandb_project` and `logger.wandb_exp_name`, export `WANDB_API_KEY`.

For more details on arguments, see the [Megatron-Bridge docs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/megatron-lm-to-megatron-bridge.md).

---

## 4. Inference

The script [inference_wan.py](inference_wan.py) runs text-to-video generation with a Megatron-format WAN checkpoint. Set `--checkpoint_step` to use a specific checkpoint iteration, `--sizes` and `--frame_nums` to specify video shape, and `--sample_steps` (default 50) for the number of noise diffusion steps.

```bash
uv run torchrun --nproc_per_node=1 \
  examples/diffusion/recipes/wan/inference_wan.py \
  --task t2v-1.3B \
  --frame_nums 81 \
  --sizes 480*832 \
  --checkpoint_dir ${WORKSPACE}/checkpoints/wan/wan2.1 \
  --checkpoint_step 10000 \
  --prompts "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --sample_steps 50
```

**Note**: Current inference path is single-GPU. Parallel inference is not yet supported.
