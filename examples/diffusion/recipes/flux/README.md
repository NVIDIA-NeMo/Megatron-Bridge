# FLUX Examples

This directory contains example scripts for the FLUX diffusion model (text-to-image) with Megatron-Bridge: checkpoint conversion, inference, pretraining, and fine-tuning.

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

**Note**: It is recommended to save the checkpoint because we will need to reuse the VAE, text encoders for the inference pipeline later as well.

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

The script [pretrain_flux.py](pretrain_flux.py) runs FLUX pretraining with the `pretrain_config()` recipe. Configuration can be overridden Hydra-style CLI keys.

**Recipe:** [megatron.bridge.diffusion.recipes.flux.flux.pretrain_config](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/diffusion/recipes/flux/flux.py)

### Quick run with mock data (single node, 8 GPUs)

```bash
uv run torchrun --nproc_per_node=8 examples/diffusion/recipes/flux/pretrain_flux.py --mock
```

### With CLI overrides only

```bash
uv run torchrun --nproc_per_node=8 examples/diffusion/recipes/flux/pretrain_flux.py --mock \
  model.tensor_model_parallel_size=4 \
  train.train_iters=10000 \
  optimizer.lr=1e-4
```


### Flow matching options

```bash
uv run torchrun --nproc_per_node=8 examples/diffusion/recipes/flux/pretrain_flux.py --mock \
  --timestep-sampling logit_normal \
  --flow-shift 1.0 \
  --use-loss-weighting
```

Before pretraining with real data, set the dataset in the recipe or in your YAML/CLI (e.g. `data_paths`, dataset blend, and cache paths). For data preprocessing, see the Megatron-Bridge data tutorials.

---

## 4. Fine-Tuning

The script [finetune_flux.py](finetune_flux.py) fine-tunes a pretrained FLUX checkpoint (Megatron format). It loads model weights and resets optimizer and step count; config can be overridden via YAML and CLI like pretrain.

Point `--load-checkpoint` at the **Megatron checkpoint directory** (either the base dir, e.g. `.../flux.1-dev`, or a specific iteration, e.g. `.../flux.1-dev/iter_0000000`):

```bash
uv run torchrun --nproc_per_node=8 examples/diffusion/recipes/flux/finetune_flux.py \
  --load-checkpoint ${WORKSPACE}/checkpoints/flux/flux.1-dev/iter_0000000 \
  --mock
```

**Note**: If you pass a path that ends with an `iter_XXXXXXX` directory, the script loads that iteration; otherwise it uses the latest iteration under the given path.

**Note**: Loss might explode if you are using a mock dataset.

---

## Summary: End-to-End Flow

1. **Conversion (HF → Megatron)**  
   Download FLUX.1-dev (optional), then run the `import` command. Use the created `iter_0000000` path as your Megatron checkpoint.

2. **Inference**  
   Run [inference_flux.py](inference_flux.py) with `--flux_ckpt` (Megatron `iter_*` path), `--vae_ckpt`, and `--prompts`.

3. **Pretraining**  
   Run [pretrain_flux.py](pretrain_flux.py) with `--mock` or your data config; optionally use `--config-file` and CLI overrides.

4. **Fine-tuning**  
   Run [finetune_flux.py](finetune_flux.py) with `--load-checkpoint` set to a Megatron checkpoint (import or pretrain/finetune output), then `--mock` or your data and overrides.

For more details, see the docstrings in each script and the recipe in `src/megatron/bridge/diffusion/recipes/flux/flux.py`.
