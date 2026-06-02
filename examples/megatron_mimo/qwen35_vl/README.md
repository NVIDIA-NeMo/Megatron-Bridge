# Qwen3.5-VL MegatronMIMO Examples

This directory contains example scripts for the current Qwen3.5-VL
MegatronMIMO path. The goal is to demonstrate the supported non-colocated MIMO
workflow, not to claim that all MegatronMIMO features are complete.

For the standard non-MIMO Qwen3.5-VL examples, see
`examples/models/qwen/qwen35_vl/`.

This example currently targets one narrow workflow: dense Qwen3.5-VL
MegatronMIMO conversion plus non-colocated full-parameter SFT on HF
conversation data. See [Current Support](#current-support) before adapting the
scripts to a new layout or dataset.

## Files

| File | Purpose |
|---|---|
| `conversion.sh` | Converts HF Qwen3.5-VL checkpoints to MegatronMIMO format and exports back to HF for a round-trip check. |
| `finetune_qwen35_vl.py` | Standalone MegatronMIMO HF-data SFT runner. |
| `slurm_sft.sh` | Multi-node Slurm launcher for the validated 27B non-colocated SFT layout (33 active ranks on 5 x 8-GPU nodes). |

## Workspace

Set `WORKSPACE` to a shared filesystem path visible on all nodes and inside the
container. The examples default to `/workspace`, but most clusters should
override it:

```bash
export WORKSPACE=/path/to/shared/workspace
export EXPERIMENT_ROOT=${WORKSPACE}/qwen35_vl_mimo
```

The scripts use:

- `${EXPERIMENT_ROOT}/models/mimo/` for converted MegatronMIMO checkpoints
  when following the commands below.
- `${EXPERIMENT_ROOT}/results/mimo/` for Slurm run outputs.

If you set `EXPERIMENT_ROOT` explicitly, outputs are written under that path.

## Conversion

Convert the default 0.8B model with one language rank and one image rank:

```bash
MIMO_MODEL_ROOT=${EXPERIMENT_ROOT}/models/mimo
WORKSPACE=${MIMO_MODEL_ROOT} \
bash examples/megatron_mimo/qwen35_vl/conversion.sh
```

Convert the 27B model for the layout used by the Slurm example:

```bash
MIMO_MODEL_ROOT=${EXPERIMENT_ROOT}/models/mimo
WORKSPACE=${MIMO_MODEL_ROOT} \
MODEL_NAME=Qwen3.5-27B \
LANGUAGE_TP=4 \
LANGUAGE_DP=1 \
LANGUAGE_RANK_OFFSET=0 \
VISION_TP=1 \
VISION_DP=1 \
VISION_RANK_OFFSET=4 \
bash examples/megatron_mimo/qwen35_vl/conversion.sh
```

This writes the MegatronMIMO checkpoint to
`${MIMO_MODEL_ROOT}/Qwen3.5-27B-mimo` and an HF export check to
`${MIMO_MODEL_ROOT}/Qwen3.5-27B-mimo-export-hf`.

## Local Smoke

A small 2-rank random-initialized smoke run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py \
    --hf-model Qwen/Qwen3.5-0.8B \
    --experiment-root ${EXPERIMENT_ROOT} \
    --allow-random-init \
    --component language=tp=1,dp=1,rank_offset=0 \
    --component images=tp=1,dp=1,rank_offset=1 \
    --global-batch-size 8 \
    --micro-batch-size 2 \
    --seq-length 2048 \
    --train-iters 2
```

## Multi-Node Slurm

`slurm_sft.sh` launches the validated 27B non-colocated SFT layout: 32 language
ranks (TP=4, PP=4, DP=2) + 1 image rank (TP=1, PP=1, DP=1) = 33 active ranks,
packed across 5 x 8-GPU nodes via an MPMD srun chain (8 + 8 + 8 + 8 + 1 tasks
per node; 7 GPUs idle on the 5th node).

Submit with the validated 27B defaults after pointing `WORKSPACE` at your
converted checkpoint root:

```bash
WORKSPACE=/path/to/shared/workspace \
  sbatch examples/megatron_mimo/qwen35_vl/slurm_sft.sh
```

All user-facing knobs (container image / mounts, tokens and caches, paths,
hyperparameters, MIMO parallelism layout) live in a single `USER
CONFIGURATION` block at the top of the script. Every knob is env-overridable,
so you can change defaults at submit time without editing the file:

```bash
sbatch --export=ALL,SEQ_LENGTH=2048,TRAIN_ITERS=100 \
  examples/megatron_mimo/qwen35_vl/slurm_sft.sh
```

`PRETRAINED_CHECKPOINT` defaults to
`${EXPERIMENT_ROOT}/models/mimo/Qwen3.5-27B-mimo`, matching the path produced
by the 27B `conversion.sh` invocation above. Run outputs are written under
`${EXPERIMENT_ROOT}/results/mimo/${RUN_NAME}`.

Edit the `#SBATCH` directives near the top of the script to match your
cluster's account, partition, and time limit. Node count is fixed at 5 — that
is the minimum required for the validated 33-rank MIMO layout under full-node
exclusive allocations.

## Current Support

Supported:

- Dense Qwen3.5-VL models: `0.8B`, `2B`, `4B`, `9B`, and `27B`.
- HF to MegatronMIMO conversion with two components: `language` and `images`.
- Non-colocated training where language and image ranks are disjoint.
- HF conversation datasets through `HFDatasetConversationProvider`.
- CORD-v2 SFT through `--dataset-maker cord_v2`.
- Converted MegatronMIMO checkpoint loading before DDP wrapping.
- Random initialization for smoke and performance-only checks.
- Optional PyTorch profiler or Nsys hooks through the runner flags.

Not supported or not validated in this example:

- Qwen3.5-VL MoE variants.
- MTP training. The example disables MTP layers.
- Packed sequences.
- Energon datasets.
- Text-only sparse-ratio experiments.
- Colocated MIMO layouts.
- MIMO throughput/FLOPs logging. Heterogeneous FLOPs accounting is not wired
  yet.
- MIMO evaluation and matched validation/test artifacts. The current runner is
  focused on train-loop execution.
