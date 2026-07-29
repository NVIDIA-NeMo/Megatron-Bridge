# MiniMax-M3 Examples

This directory contains real-checkpoint conversion and inference examples for
[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3). The bridge imports
the vision tower, both multimodal projectors, and the sparse-MoE language
backbone. The vision stack is replicated across tensor-parallel ranks.

MiniMax-M3's Lightning Indexer is not yet executed by Megatron: the text model
uses full causal attention. Its 228 tensors are stored as frozen
MiniMax-specific model state. During export, their values come from the native
Megatron checkpoint rather than source-weight passthrough. The standard export
workflow still uses the Hugging Face model reference for metadata and the
source shard map; resolving a cold cache may download source shards. The
Indexer tensors are not updated during Megatron training, so a post-training
export keeps the imported indexer rather than a trained sparse-attention
indexer.

The bundled H100 pretraining and SQuAD SFT recipes remain text-only: they use
the checkpoint-compatible text provider and do not instantiate the vision,
projector, or Lightning Indexer state. VLM training requires a multimodal
dataset and the VLM training step; this change targets complete VLM checkpoint
import and export.

## Hardware requirements

MiniMax-M3 has about 428B parameters stored in bf16. The published index
declares about 869 GB, while the safetensors headers contain about 854 GB of
tensor payload. The supplied Slurm jobs use 32 GPUs with `TP=1`, `PP=1`, and
`EP=32`, which is a conservative layout for 80 GB GPUs. Hardware with larger
GPU memory can reduce `EP` and the node count as long as `EP` divides 128.

## Setup

Set the container and mounts without placing credentials in the scripts:

```bash
export CONTAINER_IMAGE=/path/to/megatron-bridge.sqsh
export CONTAINER_MOUNTS=/shared:/shared
export HF_HOME=/shared/cache/huggingface
export UV_CACHE_DIR=/shared/cache/uv
export HF_TOKEN=your_token_if_required
export SLURM_ACCOUNT=your_slurm_account
```

The current checkout is mounted at `/opt/Megatron-Bridge` automatically and
must be on storage visible from the compute nodes. Fully populate the shared
model cache before starting either 45-minute compute job; the checkpoint
contains about 854 GB of safetensors payload:

```bash
hf download MiniMaxAI/MiniMax-M3
```

## Conversion round-trip

Run [slurm_conversion.sh](slurm_conversion.sh) from a Slurm login node to
submit the real HF checkpoint through `convert.sh roundtrip`. The job imports
it into a distributed Megatron model and exports every bridged tensor back in
memory. It compares those tensors with the original checkpoint and skips
writing a second roughly 854 GB safetensors copy.

```bash
bash examples/models/minimax/minimax_m3/slurm_conversion.sh
```

Success is reported only when all bridged language, vision, projector, and
Lightning Indexer parameters match within the round-trip script's standard
tolerances. The Indexer weights use the same mapped model state for in-memory
round trips, native checkpoints, and persisted Hugging Face export. Pass
optional launcher settings after the wrapper, such as
`--srun-arg=--mpi=pmix` when the cluster requires it.

## Inference

Submit [slurm_inference.sh](slurm_inference.sh) to convert the real checkpoint,
apply the checkpoint's chat template with thinking disabled, and greedily
generate a short response with Megatron-Core:

```bash
mkdir -p logs
sbatch --account="${SLURM_ACCOUNT}" examples/models/minimax/minimax_m3/slurm_inference.sh
```

The run is successful when it completes without missing-weight or forward
errors and the generated answer is coherent for the prompt.

## Validation status

The complete real `MiniMaxAI/MiniMax-M3` VLM checkpoint completed GPU import
to a native Megatron checkpoint and GPU export back to Hugging Face on 32 H100
80 GB GPUs with `TP=1`, `PP=1`, `EP=32`, and `ETP=1`. The imported checkpoint
passed an exact 23,416-tensor persisted-state audit. The exported checkpoint
also passed strict persisted validation: its key-to-shard map and all 59
safetensors shards are byte-for-byte identical to the pinned source, covering
23,416 tensors and 854,172,958,720 bytes of tensor payload. Configuration,
tokenizer probes, and multimodal processor inputs match the source.
Transformers 5.12.1 natively reloads both source and export through its
MiniMax-M3 checkpoint conversion mapping with zero missing, unexpected,
mismatched, or error keys.

One deterministic full-VLM greedy generation on the same 32-H100 layout
processed `docs/images/tp1.png`, generated 33 tokens under a 64-token maximum,
reached EOS naturally, and included the image-only OCR target
`Pre-LayerNorm`.
