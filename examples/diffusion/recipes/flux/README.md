# FLUX Examples

This directory contains example scripts for the FLUX diffusion model (text-to-image) with Megatron-Bridge: checkpoint conversion and inference. Pretraining and fine-tuning use the generic `scripts/training/run_recipe.py` entry point.

All commands below assume you run them from the **Megatron-Bridge repository root** unless noted. Use `uv run` when you need the project’s virtualenv (e.g. `uv run python ...`, `uv run torchrun ...`).

## Workspace Configuration

Use a `WORKSPACE` environment variable as the base directory for checkpoints and results. Default is `/workspace`. Override it if needed:

```bash
export WORKSPACE=/your/custom/path
```

Suggested layout:

- `${WORKSPACE}/checkpoints/flux/` – Megatron FLUX checkpoints (after import)
- `${WORKSPACE}/checkpoints/flux_hf/` – Hugging Face FLUX model (download or export)
- `${WORKSPACE}/results/flux/` – Training outputs (pretrain/finetune)

---

## 1. Checkpoint Conversion

The script [conversion/convert_checkpoints.py](conversion/convert_checkpoints.py) converts between Hugging Face (diffusers) and Megatron checkpoint formats.

**Source model:** [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (or a local clone).

### Download the Hugging Face model (optional)

If you want a local copy before conversion:

```bash
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir ${WORKSPACE}/checkpoints/flux_hf/flux.1-dev \
  --local-dir-use-symlinks False
```

**Note**: It is recommended to save the checkpoint because we will need to reuse the VAE and text encoders for the inference pipeline later as well.

### Import: Hugging Face → Megatron

Convert a Hugging Face FLUX model to Megatron format:

```bash
uv run python examples/diffusion/recipes/flux/conversion/convert_checkpoints.py import \
  --hf-model ${WORKSPACE}/checkpoints/flux_hf/flux.1-dev \
  --megatron-path ${WORKSPACE}/checkpoints/flux/flux.1-dev
```

The Megatron checkpoint is written under `--megatron-path` (e.g. `.../flux.1-dev/iter_0000000/`). Use that path for inference and fine-tuning.

### Export: Megatron → Hugging Face

Export a Megatron checkpoint back to Hugging Face (e.g. for use in diffusers). You must pass the **reference** HF model (for config and non-DiT components) and the **Megatron iteration directory**:

```bash
uv run python examples/diffusion/recipes/flux/conversion/convert_checkpoints.py export \
  --hf-model ${WORKSPACE}/checkpoints/flux_hf/flux.1-dev \
  --megatron-path ${WORKSPACE}/checkpoints/flux/flux.1-dev/iter_0000000 \
  --hf-path ${WORKSPACE}/checkpoints/flux_hf/flux.1-dev_export
```

**Note:** The exported directory contains only the DiT transformer weights. For a full pipeline (VAE, text encoders, etc.), copy the original HF repo and replace its `transformer` folder with the exported one.

---

## 2. Inference

The script [inference_flux.py](inference_flux.py) runs text-to-image generation with a Megatron-format FLUX checkpoint. You need:

- **FLUX checkpoint:** Megatron DiT (e.g. from the import step above).
- **VAE:** Path to VAE weights (often inside the same HF repo as FLUX, e.g. `transformer` sibling directory or a separate VAE checkpoint).
- **Text encoders:** T5 and CLIP are loaded from Hugging Face by default; you can override with local paths.

### Single prompt (default 1024×1024, 10 steps)

```bash
uv run python examples/diffusion/recipes/flux/inference_flux.py \
  --flux_ckpt ${WORKSPACE}/checkpoints/flux/flux.1-dev/iter_0000000 \
  --vae_ckpt ${WORKSPACE}/checkpoints/flux_hf/flux.1-dev/vae \
  --prompts "a dog holding a sign that says hello world" \
  --output_path ./flux_output
```


**VAE path:** If you downloaded FLUX.1-dev with `huggingface-cli`, the VAE is usually in the same repo (e.g. `${WORKSPACE}/checkpoints/flux_hf/flux.1-dev/vae`); use the path to the VAE subfolder or the main repo, depending on how the pipeline expects it.

---

## 3. Dataset Preparation

This section describes how to obtain image–text data (example: [GRIT](https://huggingface.co/datasets/zzliang/GRIT)), download images, run [prepare_energon_dataset_flux.py](prepare_energon_dataset_flux.py), and package an Energon dataset for FLUX pretraining or fine-tuning.

### 3.1. Download source data (GRIT)

The GRIT dataset is hosted on Hugging Face and consists of image and text metadata.

**Hugging Face:** [zzliang/GRIT](https://huggingface.co/datasets/zzliang/GRIT)

**Important:** The initial clone/download only retrieves metadata (URLs, captions). Images must be downloaded separately using those URLs (see below).

### 3.2. Cloning the Hugging Face repository

The repository uses Git Large File Storage (LFS). Install LFS if needed:

```bash
apt-get update
apt-get install -y git-lfs
```

Clone GRIT:

```bash
git lfs install
git clone https://huggingface.co/datasets/zzliang/GRIT
```

After cloning, image–text metadata is stored as Parquet shards under **`GRIT/grit-20m/*.parquet`** (not the raw images; those are fetched in the next steps).

### 3.3. Image download tool (`img2dataset`)

Install [img2dataset](https://github.com/rom1504/img2dataset) to fetch images from URL columns in the metadata:

```bash
uv pip install img2dataset
```

### 3.4. Downloading images with `img2dataset`

Point `--url_list` at a Parquet file under `grit-20m/` (example below uses one shard; run additional commands for other `*.parquet` files in that directory as needed, or use your `img2dataset` version’s supported way to consume multiple Parquet inputs):

```bash
img2dataset --url_list /path/to/GRIT/grit-20m/coyo_0_snappy.parquet \
            --input_format "parquet" \
            --url_col "url" \
            --caption_col "caption" \
            --output_folder /path/to/GRIT/grit_images \
            --processes_count 4 \
            --thread_count 64 \
            --image_size 256 \
            --resize_only_if_bigger=True \
            --resize_mode="keep_ratio" \
            --skip_reencode=True \
            --save_additional_columns '["id","noun_chunks","ref_exps","clip_similarity_vitb32","clip_similarity_vitl14"]' \
            --enable_wandb False
```

**Note:** The number of successfully downloaded image–text pairs is often smaller than the row count in the Parquet file because some URLs are broken or unreachable.

### 3.5. Model preparation (embeddings and Energon dataset)

**Generate embeddings** (T5, CLIP, VAE, etc.) and write WebDataset-style output for FLUX. `--data_folder` should be the `img2dataset` output directory (here `grit_images`):

```bash
uv run python examples/diffusion/recipes/flux/prepare_energon_dataset_flux.py \
  --data_folder /path/to/GRIT/grit_images/ \
  --output_dir /path/to/GRIT/grit_wds \
  --center-crop
```

**Prepare the Energon dataset** from that output:

```bash
energon prepare /path/to/GRIT/grit_wds
```

For details on the Energon dataset format, `energon prepare`, and the data loader, see the [Megatron-Energon documentation](https://nvidia.github.io/Megatron-Energon/).

When prompted, use a train/val/test split such as **8 / 1 / 1**, answer **Y** to confirmations as needed, and choose **Crude sample (11)** when offered.

### 3.6. Training with the prepared dataset

Use the prepared path as `dataset.path` for pretraining or fine-tuning (see [§4 Pretraining](#4-pretraining) and [§5 Fine-Tuning](#5-fine-tuning)):

```bash
uv run torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
  --recipe flux_12b_pretrain_config \
  --step_func flux_step \
  dataset.path=/path/to/GRIT/grit_wds/
```

---

## 4. Pretraining

Run FLUX pretraining with the generic **run_recipe** script (same entry point as for LLM training).

**Recipe:** [megatron.bridge.diffusion.recipes.flux.flux.flux_12b_pretrain_config](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/diffusion/recipes/flux/flux.py)

From the **Megatron-Bridge repository root**:

**Mock data (no dataset path):**

```bash
uv run torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
  --recipe flux_12b_pretrain_config \
  --step_func flux_step
```

**Real data (WebDataset path):** Set `dataset.path` so the recipe uses real data instead of mock:

```bash
uv run torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
  --recipe flux_12b_pretrain_config \
  --step_func flux_step \
  dataset.path=${WORKSPACE}/data/my_flux_wds/
```

**With CLI overrides (iters, LR, batch size, etc.):**

```bash
uv run torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
  --recipe flux_12b_pretrain_config \
  --step_func flux_step \
  dataset.path=${WORKSPACE}/data/my_flux_wds/ \
  train.train_iters=10000 \
  train.global_batch_size=16 \
  optimizer.lr=1e-4
```

**Small datasets (e.g. &lt; 100 examples):** Use a smaller `dataset.num_workers` so each DataLoader worker gets samples (e.g. `dataset.num_workers=2`), and set `train.global_batch_size` appropriately (e.g. 8 for 64 examples on 8 GPUs).

For preparing real data (GRIT `grit-20m/*.parquet` → img2dataset → embeddings → Energon), see [§3 Dataset Preparation](#3-dataset-preparation). For other WebDataset workflows, see the Megatron-Bridge data tutorials.

---

## 5. Fine-Tuning

Run FLUX fine-tuning with the generic **run_recipe** script. Set the pretrained checkpoint via the **checkpoint.pretrained_checkpoint** CLI override (path to the Megatron checkpoint directory or a specific iteration, e.g. `.../flux.1-dev` or `.../flux.1-dev/iter_0000000`):

**Resume / finetune from a checkpoint:**

```bash
uv run torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
  --recipe flux_12b_sft_config \
  --step_func flux_step \
  checkpoint.pretrained_checkpoint=${WORKSPACE}/checkpoints/flux/flux.1-dev/iter_0000000
```

**With real data and overrides:**

```bash
uv run torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
  --recipe flux_12b_sft_config \
  --step_func flux_step \
  checkpoint.pretrained_checkpoint=${WORKSPACE}/checkpoints/flux/flux.1-dev/iter_0000000 \
  dataset.path=${WORKSPACE}/data/my_flux_wds/ \
  train.global_batch_size=8 \
  optimizer.lr=5e-6
```

**Note:** Loss might be exploded if you attempt to finetune a pretrained checkpoint on mock dataset for testing purpose.

---

## Summary: End-to-End Flow

1. **Conversion (HF → Megatron)**  
   Download FLUX.1-dev (optional), then run the `import` command. Use the created `iter_0000000` path as your Megatron checkpoint.

2. **Inference**  
   Run [inference_flux.py](inference_flux.py) with `--flux_ckpt` (Megatron `iter_*` path), `--vae_ckpt`, and `--prompts`.

3. **Dataset preparation (optional, for real training data)**  
   Follow [§3 Dataset Preparation](#3-dataset-preparation): GRIT clone → `grit-20m/*.parquet` → `img2dataset` → `prepare_energon_dataset_flux.py` → `energon prepare` → use the resulting directory as `dataset.path`.

4. **Pretraining**  
   Run `scripts/training/run_recipe.py --recipe flux_12b_pretrain_config --step_func flux_step` (optionally with `dataset.path=...` for real data and CLI overrides).

5. **Fine-Tuning**  
   Run `scripts/training/run_recipe.py --recipe flux_12b_sft_config --step_func flux_step checkpoint.pretrained_checkpoint=<path>` (optionally with `dataset.path=...` and overrides).

For more details, see the recipe in `src/megatron/bridge/diffusion/recipes/flux/flux.py` and `scripts/training/run_recipe.py`.
