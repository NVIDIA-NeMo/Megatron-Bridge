# Ministral 3 - Vision Language Model

This directory contains conversion and inference scripts for the Ministral 3
3B, 8B, and 14B Base vision-language checkpoints.

For model introduction and architecture details, see the [Ministral 3 documentation](../../../../docs/models/mistral/ministral3.md).

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:

- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Supported Base Checkpoints

Select a checkpoint with `MODEL_SIZE`. The scripts use the following model and
multi-GPU topology defaults:

| `MODEL_SIZE` | Hugging Face checkpoint | Validated revision | TP | PP |
|---|---|---|---:|---:|
| `3B` (default) | `mistralai/Ministral-3-3B-Base-2512` | `6f9c4b12a95b139af68670a6713616b757923735` | 2 | 1 |
| `8B` | `mistralai/Ministral-3-8B-Base-2512` | `d4883f9b36aa2e5d775730d3fdba3d30de51a8ef` | 2 | 1 |
| `14B` | `mistralai/Ministral-3-14B-Base-2512` | `5b0ceedbb42dff466ae60b258ba296f32da51384` | 4 | 1 |

`HF_MODEL_ID`, `MODEL_NAME`, `TP`, `PP`, `EP`, `ETP`, and
`NPROC_PER_NODE` can be overridden when validating a local snapshot or a
different topology. To reproduce an immutable revision, download that snapshot
and pass its local path through `HF_MODEL_ID` while setting `MODEL_NAME` to the
checkpoint name.

## Checkpoint Conversion and Validation

The conversion script runs HF-to-Megatron import, Megatron-to-HF export, and a
multi-GPU round-trip check that reloads the imported Megatron checkpoint.

```bash
MODEL_SIZE=3B bash examples/models/mistral/ministral3/conversion.sh
MODEL_SIZE=8B bash examples/models/mistral/ministral3/conversion.sh
MODEL_SIZE=14B bash examples/models/mistral/ministral3/conversion.sh
```

## Inference

The inference script generates from the original HF source, the imported
Megatron checkpoint, and the exported HF source with the same image and prompt.

```bash
MODEL_SIZE=3B bash examples/models/mistral/ministral3/inference.sh
MODEL_SIZE=8B bash examples/models/mistral/ministral3/inference.sh
MODEL_SIZE=14B bash examples/models/mistral/ministral3/inference.sh
```

Override `IMAGE_PATH`, `PROMPT`, and `MAX_NEW_TOKENS` to use a different
image, prompt, or generation length. Base checkpoints do not provide a chat
template; the shared VLM helper inserts the checkpoint's image token and sends
the raw completion-style prompt through the processor.

## Finetune Recipes

- See: [bridge.recipes.ministral3](../../../../docs/apidocs/bridge/bridge.recipes.ministral3.md)
- Available recipes:
  - `ministral3_3b_sft_config`: Finetuning for 3B VL model
  - `ministral3_8b_sft_config`: Finetuning for 8B VL model
  - `ministral3_14b_sft_config`: Finetuning for 14B VL model
  - `ministral3_3b_peft_config`: Finetuning for 3B VL model with PEFT support
  - `ministral3_8b_peft_config`: Finetuning for 8B VL model with PEFT support
  - `ministral3_14b_peft_config`: Finetuning for 14B VL model with PEFT support

Before training, ensure the following environment variables are set:
1. `SAVE_DIR`: checkpoint and log saving directory
2. `HF_TOKEN`: to download models from HF Hub (if required)
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets
4. `WANDB_API_KEY`: (optional) to enable WandB logging

### Pretrain

Pretraining is not verified for this model.

### Supervised Fine-Tuning (SFT)

See the [sft_unpacked.sh](sft_unpacked.sh) script for full parameter fine-tuning with configurable model parallelisms.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

See the [peft_unpacked.sh](peft_unpacked.sh) script for LoRA fine-tuning with configurable tensor and pipeline parallelism.

### Recommended Configurations

| Model | Mode | TP | PP | Global Batch Size | Learning Rate | Hardware |
|-------|------|----|----|-------------------|---------------|----------|
| Ministral 3 3B | Full SFT | 1 | 1 | 32-64 | 5e-6 | 8 GPUs |
| Ministral 3 3B | LoRA/DoRA | 1 | 1 | 64-128 | 1e-4 | 8 GPUs |
| Ministral 3 8B | Full SFT | 2 | 1 | 32-64 | 5e-6 | 8 GPUs |
| Ministral 3 8B | LoRA/DoRA | 1 | 1 | 64-128 | 1e-4 | 8 GPUs |
| Ministral 3 14B | Full SFT | 4 | 1 | 16-32 | 5e-6 | 8 GPUs |
| Ministral 3 14B | LoRA/DoRA | 2 | 1 | 32-64 | 1e-4 | 8 GPUs |

**Note:** LoRA/DoRA significantly reduces memory requirements, allowing for larger batch sizes and fewer GPUs.

### Expected Training Dynamics
We provide a [Weights & Biases report](https://api.wandb.ai/links/nvidia-nemo-fw-public/h32cflfn) for the expected loss curves and grad norms.

## Evaluation

Coming soon.
