# MiniMax-M3 Examples

This directory contains real-checkpoint conversion and inference examples for
[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3). The bridge imports
the language backbone from the multimodal checkpoint; the vision tower,
projector, lightning-indexer weights, and MTP modules are not converted.

## Hardware requirements

MiniMax-M3 has about 428B parameters stored in bf16 (the published checkpoint
is about 869 GB). The supplied Slurm jobs use 32 GPUs with `TP=1`, `PP=1`, and
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
download is about 869 GB:

```bash
hf download MiniMaxAI/MiniMax-M3
```

## Conversion round-trip

Run [slurm_conversion.sh](slurm_conversion.sh) from a Slurm login node to
submit the real HF checkpoint through `convert.sh roundtrip`. The job imports
it into a distributed Megatron model and exports every bridged tensor back in
memory. It compares those tensors with the original checkpoint and skips
writing a second 869 GB copy.

```bash
bash examples/models/minimax/minimax_m3/slurm_conversion.sh
```

Success is reported only when all bridged language-model parameters match
within the round-trip script's standard tolerances. This is an in-memory
verification; standalone Hugging Face checkpoint export is not supported
because the bridge intentionally omits the multimodal modules. Pass optional
launcher settings after the wrapper, such as `--srun-arg=--mpi=pmix` when the
cluster requires it.

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

## Validated configuration

The real `MiniMaxAI/MiniMax-M3` checkpoint was validated on 32 H100 80 GB
GPUs with `TP=1`, `PP=1`, `EP=32`, and `ETP=1`. All 1,053 mapped parameter
tasks passed the in-memory HF → Megatron → HF round-trip check within the
script's standard tolerances. With the chat template enabled, Megatron
generated a coherent continuation beginning:

> The sky appears blue because of Rayleigh scattering, where sunlight
> interacts with Earth's atmosphere.
