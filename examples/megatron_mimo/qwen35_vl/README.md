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
| `slurm_sft.sh` | Slurm launcher for the non-colocated MIMO SFT path. |

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

## Slurm SFT

The 27B Slurm example uses:

- Language: TP4, PP4, DP2 on ranks 0-31.
- Images: TP1, PP1, DP1 on rank 32.
- Active ranks: 33.
- MIMO microbatch size: 2.
- Language-local microbatch size: 1.
- Global batch size: 32.
- Sequence length: 2048.

Before submission, set at least:

```bash
export WORKSPACE=/path/to/shared/workspace
export CONTAINER_IMAGE=/path/to/container.sqsh
```

Then submit the Slurm launcher:

```bash
sbatch examples/megatron_mimo/qwen35_vl/slurm_sft.sh
```

Set `PRETRAINED_CHECKPOINT` to load a converted MegatronMIMO checkpoint. If
neither `PRETRAINED_CHECKPOINT` nor `LOAD_CHECKPOINT` is set, the launcher uses
random initialization and passes `--allow-random-init`.

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
