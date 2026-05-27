# Qwen3-ASR - Audio Speech Recognition Model

This directory contains example scripts for Qwen3-ASR audio speech recognition models.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```



## Checkpoint Conversion

### Import HF → Megatron
To import the HF model to your desired Megatron path:
```bash
uv run python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3-ASR-1.7B \
  --megatron-path ${WORKSPACE}/models/Qwen3-ASR-1.7B
```

### Export Megatron → HF
```bash
uv run python examples/conversion/convert_checkpoints.py export \
  --hf-model Qwen/Qwen3-ASR-1.7B \
  --megatron-path ${WORKSPACE}/models/Qwen3-ASR-1.7B/iter_0000000 \
  --hf-path ${WORKSPACE}/models/Qwen3-ASR-1.7B-hf-export
```

### Round-trip Validation
```bash
uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
  --hf-model-id Qwen/Qwen3-ASR-1.7B \
  --megatron-load-path ${WORKSPACE}/models/Qwen3-ASR-1.7B/iter_0000000 \
  --trust-remote-code \
  --tp 2 --pp 1
```

## Inference

### Run Inference on HuggingFace Checkpoint

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/conversion/hf_to_megatron_generate_audio_lm.py \
  --hf_model_path Qwen/Qwen3-ASR-1.7B \
  --audio_url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac" \
  --prompt "" \
  --tp 2 \
  --max_new_tokens 50
```

### Run Inference on Converted Megatron Checkpoint

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/conversion/hf_to_megatron_generate_audio_lm.py \
  --hf_model_path Qwen/Qwen3-ASR-1.7B \
  --megatron_model_path ${WORKSPACE}/models/Qwen3-ASR-1.7B/iter_0000000 \
  --audio_url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac" \
  --prompt "" \
  --tp 2 \
  --max_new_tokens 50
```

Note:
- `--megatron_model_path` is optional. If not specified, the script will convert the model in-memory and then run forward.
- You can also use local audio files: `--audio_path /path/to/audio.wav`

See the [inference.sh](inference.sh) script for the full runnable commands.

**Expected output:**
```
======== GENERATED TEXT OUTPUT ========
Audio: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac
Prompt:
Generated: system

user

assistant
language English<asr_text>Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
=======================================
```
