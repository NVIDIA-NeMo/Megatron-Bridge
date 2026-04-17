# Qwen3-Omni Examples

This directory contains example scripts for **Qwen3-Omni thinker-side support** in Megatron Bridge.

For model introduction and implementation notes, see the [Qwen3-Omni documentation](../../../../docs/models/vlm/qwen3-omni.md).

## Current Scope

These examples cover:

- reduced single-GPU smoke checkpoint creation
- Hugging Face -> Megatron checkpoint import
- Megatron -> Hugging Face checkpoint export
- local inference with reduced HF, imported Megatron, and exported HF checkpoints

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
export VIDEO_PATH=/path/to/example.mp4
```

The example smoke checkpoint keeps the original hidden dimensions intact and only trims layer counts, which keeps the HF config compatible while making single-GPU validation practical.

## Data Preparation (Omni Bench)

If you have the Omni Bench parquet data locally, you can convert it into JSONL plus extracted media assets:

```bash
python examples/models/vlm/qwen3_omni/convert_omni_bench_to_jsonl.py \
  --input-root ./Omni_Bench_fix_simple \
  --output-root ./omni_bench_fix_simple \
  --splits train test
```

This produces:
- `./omni_bench_fix_simple/train/train.jsonl`
- `./omni_bench_fix_simple/test/test.jsonl`
- extracted media under `./omni_bench_fix_simple/{split}/media`

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

Run local inference across the reduced HF checkpoint, imported Megatron checkpoint, and exported HF checkpoint:

```bash
export VIDEO_PATH=/path/to/example.mp4
bash examples/models/vlm/qwen3_omni/inference.sh
```

If you want to use the audio track from the video during understanding:

```bash
export VIDEO_PATH=/path/to/example.mp4
export USE_AUDIO_IN_VIDEO=1
bash examples/models/vlm/qwen3_omni/inference.sh
```

You can also use a remote video URL instead of a local file:

```bash
export VIDEO_URL=https://example.com/example.mp4
bash examples/models/vlm/qwen3_omni/inference.sh
```

This script reuses [examples/conversion/hf_to_megatron_generate_omni_lm.py](../../../conversion/hf_to_megatron_generate_omni_lm.py) and runs:

- inference with the reduced local HF smoke checkpoint
- inference with the imported Megatron checkpoint
- inference with the exported HF checkpoint

Useful overrides:

- `PROMPT` (default: `What is happening in this video?`)
- `MAX_NEW_TOKENS` (default: `50`)
- `NPROC_PER_NODE` / `TP` / `PP` / `EP` / `ETP` (default: `1`)
- `DRY_RUN=1` to print commands without executing them

The omni inference helper depends on `qwen-omni-utils[decord]` for video/audio preprocessing.

## Training (local)

The training recipe entrypoint is:

```bash
bash examples/models/vlm/qwen3_omni/local_train_thinker_full.sh
```

Required environment variables:

```bash
export HF_MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct
export TRAIN_JSONL=/path/to/train.jsonl
```

Optional overrides:

- `WORKSPACE` (default: `${PWD}/.cache/qwen3_omni_train`)
- `RESULTS_DIR` / `LOG_DIR` (default: under `WORKSPACE`)

To apply the 4-node TP2/PP2/EP8/SP preset used in our 32-GPU validation:

```bash
export PRESET=4node_tp2_ep8_sp
bash examples/models/vlm/qwen3_omni/local_train_thinker_full.sh
```
