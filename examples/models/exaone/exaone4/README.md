# EXAONE 4.0 Examples

This directory contains example scripts for EXAONE 4.0 language models.

The examples target `LGAI-EXAONE/EXAONE-4.0-1.2B`, a dense 1.2B parameter model. The scripts use
single-GPU tensor and pipeline parallelism settings by default.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By
default, this is set to `/workspace`.

    export WORKSPACE=/your/custom/path

## Checkpoint Conversion

See [conversion.sh](conversion.sh) for Hugging Face to Megatron import, Megatron to Hugging Face export, and
round-trip validation.

    ./examples/models/exaone/exaone4/conversion.sh

## Inference

See [inference.sh](inference.sh) for text generation with:

- the Hugging Face checkpoint
- the imported Megatron checkpoint
- the exported Hugging Face checkpoint

    PROMPT="Explain checkpoint conversion in one paragraph." ./examples/models/exaone/exaone4/inference.sh

The default scripts use `--tp 1 --pp 1` and `--nproc_per_node=1`, which is suitable for the 1.2B dense model on a
single GPU.
