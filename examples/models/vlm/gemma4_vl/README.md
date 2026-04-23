# Gemma 4 VL Examples

This directory contains example scripts for the Gemma 4 26B-A4B vision-language model.

Gemma 4 26B-A4B is a Mixture-of-Experts (MoE) VLM with a SigLIP vision encoder and a 26B sparse language model (4B active parameters). It requires dedicated handling compared to dense VLMs due to its MoE architecture and expert parallelism requirements.

## Requirements

Gemma 4 requires `transformers>=5.5.0`. The project lockfile pins an older version, so upgrade before running any script:

```bash
uv pip install -q --upgrade 'transformers>=5.5.0' mistral_common
```

All scripts in this directory run `uv run --no-sync` to prevent `uv` from reverting the upgrade.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted Megatron checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

### Import HF → Megatron

```bash
uv pip install -q --upgrade 'transformers>=5.5.0'
uv run --no-sync python examples/conversion/convert_checkpoints.py import \
    --hf-model google/gemma-4-26B-A4B \
    --megatron-path ${WORKSPACE}/models/gemma-4-26B-A4B
```

### Export Megatron → HF

```bash
uv run --no-sync python examples/conversion/convert_checkpoints.py export \
    --hf-model google/gemma-4-26B-A4B \
    --megatron-path ${WORKSPACE}/models/gemma-4-26B-A4B/iter_0000000 \
    --hf-path ${WORKSPACE}/models/gemma-4-26B-A4B-hf-export
```

See the [conversion.sh](conversion.sh) script for more examples including multi-GPU round-trip validation.

## Inference

Gemma 4 uses a dedicated inference script (`hf_to_megatron_generate_gemma4.py`) instead of the generic VLM inference script. This is required because Gemma 4's processor uses a different input format for image tokens.

### Text-only

```bash
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_to_megatron_generate_gemma4.py \
    --hf_model_path google/gemma-4-26B-A4B \
    --prompt "The capital of France is" \
    --max_new_tokens 20 \
    --tp 4 --pp 2
```

### Vision + Text (HF weights)

Use the instruction-tuned model (`-it`) for image+text queries — the base model has no chat template and requires manual image token injection.

```bash
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_to_megatron_generate_gemma4.py \
    --hf_model_path google/gemma-4-26B-A4B-it \
    --image_path "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/Demos/sample-data/GoldenGate.png" \
    --prompt "What is shown in this image?" \
    --max_new_tokens 50 \
    --tp 4 --pp 2
```

### Vision + Text (imported Megatron checkpoint)

```bash
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_to_megatron_generate_gemma4.py \
    --hf_model_path google/gemma-4-26B-A4B \
    --megatron_model_path ${WORKSPACE}/models/gemma-4-26B-A4B/iter_0000000 \
    --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG" \
    --prompt "What animal is on the candy?" \
    --max_new_tokens 50 \
    --tp 4 --pp 2
```

Note: when loading a Megatron checkpoint for VLM inference, use the base model (`gemma-4-26B-A4B`, not `-it`) as the `--hf_model_path` to match the checkpoint's tokenizer.

See the [inference.sh](inference.sh) script for all three steps.

**Expected output (Golden Gate image):**
```
======== GENERATED TEXT OUTPUT ========
Image: https://.../GoldenGate.png
Prompt: What is shown in this image?
New tokens: This image shows the Golden Gate Bridge in San Francisco, California.
The photo captures the iconic red suspension bridge spanning the Golden Gate
strait, with the Marin Headlands visible in the background.
=======================================
```

## Finetune Recipes

Available recipes:
- `gemma4_vl_26b_sft_config` — Full supervised fine-tuning
- `gemma4_vl_26b_peft_config` — LoRA parameter-efficient fine-tuning

Before training, ensure the following environment variables are set:
1. `WORKSPACE`: base directory for checkpoints and results (default: `/workspace`)
2. `HF_TOKEN`: to download models from HF Hub (if required)
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets
4. `WANDB_API_KEY`: (optional) to enable WandB logging; set `WANDB_MODE=disabled` to turn off

### Supervised Fine-Tuning (SFT)

For single-node interactive runs, see [sft.sh](sft.sh).

For multi-node Slurm jobs, see [slurm_sft.sh](slurm_sft.sh). Default configuration: TP=2, PP=1, EP=8 on 2 nodes (16 GPUs).

```bash
# Override defaults via environment variables
PRETRAINED_CHECKPOINT=${WORKSPACE}/models/gemma-4-26B-A4B \
TP=2 PP=1 EP=8 \
sbatch --nodes=2 slurm_sft.sh
```

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

For single-node interactive runs, see [peft.sh](peft.sh).

For multi-node Slurm jobs, see [slurm_peft.sh](slurm_peft.sh). Default configuration: TP=2, PP=1, EP=4 on 1 node (8 GPUs).

```bash
sbatch slurm_peft.sh
```

> **Important:** LoRA fine-tuning of MoE models requires `EP > 1`. With `EP=1`, multiple TP ranks produce duplicate checkpoint keys for expert LoRA adapters (`CheckpointingException: Duplicate ShardedObject keys`). Setting `EP >= TP` (so that `DP >= EP`) satisfies this constraint.

### Recommended Configurations

| Mode | TP | PP | EP | Nodes | Global Batch Size | Learning Rate | Notes |
|------|----|----|----|----|-------------------|---------------|-------|
| Full SFT | 2 | 1 | 8 | 2 | 32 | 5e-5 | Max EP=DP=8; vision unfrozen; no activation recompute |
| Full SFT | 4 | 2 | 1 | 1 | 32 | 5e-5 | `recompute_granularity="selective"`; freeze vision |
| LoRA | 2 | 1 | 4 | 1 | 32 | 2e-4 | EP=4 required (see note above) |

> **Note:** Do not use `recompute_granularity="full"`. Megatron's `CheckpointFunction` does not support non-tensor (tuple) arguments, causing a `TypeError` at runtime. Use `"selective"` instead.

### Checkpoint Formats

By default, SFT training produces a **`dp_reshardable`** checkpoint (`use_distributed_optimizer=True` is the Megatron default). This format embeds model weights inside the distributed optimizer's sharded parameter buffers. It is compact but **not loadable** by bridge inference or evaluation scripts (`bridge.load_megatron_model()` fails because model ShardedTensor keys are absent from the checkpoint metadata).

To produce a **`torch_dist`** checkpoint (required for inference and evaluation), add these two flags:

```
optimizer.use_distributed_optimizer=False
ddp.use_distributed_optimizer=False
```

| Format | Size per rank | Loadable for inference/eval |
|--------|---------------|------------------------------|
| `dp_reshardable` (default) | ~11 GB | No |
| `torch_dist` | ~40–44 GB | Yes |

> **Single-node TP=4,PP=2 save timeout:** When saving a `torch_dist` checkpoint with TP=4,PP=2 on a single node, each rank writes ~40–44 GB. Rank 0's shard takes ~14 minutes. The other ranks call the post-save barrier first and wait, triggering the default 10-minute NCCL watchdog. Add `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` to prevent the watchdog from killing the process:
> ```bash
> TORCH_NCCL_ASYNC_ERROR_HANDLING=0 uv run --no-sync python -m torch.distributed.run ...
> ```

### Expert Parallelism Constraint

For MoE models, EP must divide DP (data parallel degree), where `DP = world_size / (TP × PP)`. For example, with 8 GPUs, TP=2, PP=1: DP=4, so EP ∈ {1, 2, 4}. EP=1 is not allowed for LoRA (see above), so use EP=4 for single-node LoRA.

## Evaluation

> **Prerequisite:** Evaluation scripts use `bridge.load_megatron_model()` which requires a **`torch_dist`** checkpoint. The default SFT configuration produces a `dp_reshardable` checkpoint (not loadable). To produce a loadable checkpoint, run SFT with `optimizer.use_distributed_optimizer=False ddp.use_distributed_optimizer=False`. See [Checkpoint Formats](#checkpoint-formats) above.

Two evaluation scripts are provided for measuring SFT quality on CORD-v2:

**Teacher-forced accuracy** (fast, measures token-level accuracy on the full validation set):
```bash
python examples/models/vlm/gemma4_vl/eval_sft_cord_v2.py \
    --megatron_model_path ${WORKSPACE}/results/gemma4_vl_sft_tp2_pp1_ep8/iter_0000040 \
    --tp 4 --pp 2
```

**Autoregressive generation** (slow, generates output tokens and compares to ground truth):
```bash
python examples/models/vlm/gemma4_vl/eval_sft_autoregressive.py \
    --megatron_model_path ${WORKSPACE}/results/gemma4_vl_sft_tp2_pp1_ep8/iter_0000040 \
    --tp 4 --pp 2
```

For batch evaluation on Slurm, see [slurm_eval_sft.sh](slurm_eval_sft.sh).

After 40 SFT iterations on CORD-v2, expected teacher-forced token accuracy is ~98%.
