# Qwen2.5-Omni Examples

This directory contains example scripts for **Qwen2.5-Omni thinker-side support** in Megatron Bridge.

For model introduction and implementation notes, see the [Qwen2.5-Omni documentation](../../../../docs/models/vlm/qwen2.5-omni.md).

## Current Scope

These examples cover:

- reduced single-GPU smoke checkpoint creation
- Hugging Face → Megatron checkpoint import
- Megatron → Hugging Face checkpoint export
- single-rank multimodal thinker smoke inference with one real local image+audio sample (via `examples/conversion/hf_to_megatron_qwen25_omni_smoke.py`)

These examples do **not** cover:

- full training runs
- distributed parallel validation beyond single-rank smoke
- talker / token2wav audio-output paths beyond what the base HF checkpoint needs
- Megatron inference with `inference_params` for all production settings

## Workspace Configuration

All scripts default to a repo-local cache workspace:

```bash
export WORKSPACE=$PWD/.cache/qwen25_omni_examples
```

You can override it if needed. The default directory structure is:

- `${WORKSPACE}/hf/` - reduced local HF smoke checkpoints
- `${WORKSPACE}/megatron/` - imported Megatron checkpoints
- `${WORKSPACE}/export/` - exported HF checkpoints
- `${WORKSPACE}/tmp/` - temporary files
- `${WORKSPACE}/hf_home/` - Hugging Face cache used by the examples

## Required Local Assets

These examples assume the following local assets are available:

```bash
export SOURCE_HF_MODEL=/path/to/Qwen2.5-Omni-7B
export SAMPLE_PARQUET=/path/to/multimodal_eval_samples.parquet
```

The example smoke checkpoint keeps the original hidden dimensions intact and only trims layer counts, which keeps the HF config compatible while making single-GPU validation practical.

## Checkpoint Conversion

Run the full local smoke conversion flow:

```bash
export SOURCE_HF_MODEL=/path/to/Qwen2.5-Omni-7B
bash examples/models/vlm/qwen25_omni/conversion.sh
```

This script will:

1. create a reduced thinker-only HF smoke checkpoint
2. import that checkpoint into Megatron format
3. export the imported Megatron checkpoint back to HF format

## Inference

Run local single-rank multimodal thinker smoke inference:

```bash
export SAMPLE_PARQUET=/path/to/multimodal_eval_samples.parquet
bash examples/models/vlm/qwen25_omni/inference.sh
```

This script runs:

- HF thinker smoke inference from the reduced local checkpoint
- Megatron thinker smoke inference from the imported Megatron checkpoint
- HF thinker smoke inference from the exported HF checkpoint

All runs use one real image+audio sample from the local parquet file.
