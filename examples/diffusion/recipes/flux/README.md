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

## 3. Pretraining

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

For data preprocessing and WebDataset format, see the Megatron-Bridge data tutorials.

---

## 4. Fine-Tuning

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

3. **Pretraining**  
   Run `scripts/training/run_recipe.py --recipe flux_12b_pretrain_config --step_func flux_step` (optionally with `dataset.path=...` for real data and CLI overrides).

4. **Fine-Tuning**  
   Run `scripts/training/run_recipe.py --recipe flux_12b_sft_config --step_func flux_step checkpoint.pretrained_checkpoint=<path>` (optionally with `dataset.path=...` and overrides).

For more details, see the recipe in `src/megatron/bridge/diffusion/recipes/flux/flux.py` and `scripts/training/run_recipe.py`.
