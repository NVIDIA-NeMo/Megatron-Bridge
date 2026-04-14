# Qwen3-Omni Examples

This directory contains example scripts for **Qwen3-Omni thinker-side support** in Megatron Bridge.

For model introduction and implementation notes, see the [Qwen3-Omni documentation](../../../../docs/models/vlm/qwen3-omni.md).

## Current Scope

These examples cover:

- reduced single-GPU smoke checkpoint creation
- Hugging Face -> Megatron checkpoint import
- Megatron -> Hugging Face checkpoint export
- single-rank multimodal thinker smoke inference with one real local image+audio sample

These examples do **not** cover:

- distributed parallel validation beyond single-rank smoke
- talker / code2wav audio-output checkpoints
- Megatron inference with `inference_params`

## Workspace Configuration

All scripts default to a repo-local cache workspace:

```bash
export WORKSPACE=$PWD/.cache/qwen3_omni_examples
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
export SOURCE_HF_MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct
export SAMPLE_PARQUET=/path/to/multimodal_eval_samples.parquet
```

The example smoke checkpoint keeps the original hidden dimensions intact and only trims layer counts, which keeps the HF config compatible while making single-GPU validation practical.

## Checkpoint Conversion

Run the full local smoke conversion flow:

```bash
export SOURCE_HF_MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct
bash examples/models/vlm/qwen3_omni/conversion.sh
```

This script will:

1. create a reduced thinker-only HF smoke checkpoint
2. import that checkpoint into Megatron format
3. export the imported Megatron checkpoint back to HF format

## Inference

Run local single-rank multimodal thinker smoke inference:

```bash
export SAMPLE_PARQUET=/path/to/multimodal_eval_samples.parquet
bash examples/models/vlm/qwen3_omni/inference.sh
```

This script runs:

- HF thinker smoke inference from the reduced local checkpoint
- Megatron thinker smoke inference from the imported Megatron checkpoint
- HF thinker smoke inference from the exported HF checkpoint

All runs use one real image+audio sample from the local parquet file.

## Training (local)

The training recipe entrypoint is:

```bash
bash examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp.sh
```

Required environment variables:

```bash
export HF_MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct
export TRAIN_JSONL=/path/to/train.jsonl
```

Optional overrides:

- `WORKSPACE` (default: `${PWD}/.cache/qwen3_omni_train`)
- `RESULTS_DIR` / `LOG_DIR` (default: under `WORKSPACE`)
