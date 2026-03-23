# Qwen2.5-Omni Examples

This directory contains example scripts for Qwen2.5-Omni multimodal models.

Qwen2.5-Omni supports simultaneous processing of images, video, audio, and text using a dense Qwen2 language backbone with multimodal RoPE (mrope).

| Model | HF ID | Architecture | Params |
|---|---|---|---|
| Qwen2.5-Omni-7B | `Qwen/Qwen2.5-Omni-7B` | Dense (Qwen2) + Vision + Audio | 7B |

## Prerequisites

Audio and video processing requires `qwen-omni-utils` with `decord`. Install it into the project environment:

```bash
uv pip install qwen-omni-utils[decord]
```

Audio extraction from video (`--use_audio_in_video`) additionally requires `ffmpeg`. If `apt-get install ffmpeg` is unavailable (e.g. in a container), install it via `imageio-ffmpeg`:

```bash
uv pip install imageio-ffmpeg
ln -sf $(uv run python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") /usr/local/bin/ffmpeg
```

> **Note:** `--use_audio_in_video` requires a **local file** passed via `--video_path`. Audio extraction does not work with `--video_url` because `audioread` cannot stream audio directly from a URL.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable for the base directory. Default: `/workspace`.

```bash
export WORKSPACE=/your/custom/path
```

## Checkpoint Conversion

See [conversion.sh](conversion.sh) for checkpoint conversion examples.

### Import HF → Megatron

```bash
python examples/conversion/convert_checkpoints.py import \
    --hf-model Qwen/Qwen2.5-Omni-7B \
    --megatron-path ${WORKSPACE}/models/Qwen2.5-Omni-7B
```

### Export Megatron → HF

```bash
python examples/conversion/convert_checkpoints.py export \
    --hf-model Qwen/Qwen2.5-Omni-7B \
    --megatron-path ${WORKSPACE}/models/Qwen2.5-Omni-7B/iter_0000000 \
    --hf-path ${WORKSPACE}/models/Qwen2.5-Omni-7B-hf-export
```

### Round-trip Validation

```bash
python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id Qwen/Qwen2.5-Omni-7B \
    --megatron-load-path ${WORKSPACE}/models/Qwen2.5-Omni-7B/iter_0000000 \
    --tp 2 --pp 1
```

## Inference

See [inference.sh](inference.sh) for multimodal generation with:
- Hugging Face checkpoint
- Imported Megatron checkpoint (after [conversion.sh](conversion.sh) import)
- Exported HF checkpoint

The default parallelism for 7B is `--tp 2` (2 GPUs). For larger variants scale TP accordingly.

### Example: Video only

```bash
uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_to_megatron_generate_omni_lm.py \
    --hf_model_path Qwen/Qwen2.5-Omni-7B \
    --video_url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/audio_visual.mp4" \
    --prompt "What was the first sentence the boy said when he met the girl?" \
    --max_new_tokens 64 \
    --tp 2
```

### Example: Video + Audio (requires ffmpeg and a local file)

```bash
# Download the video first
wget -O /path/to/video.mp4 "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/audio_visual.mp4"

uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_to_megatron_generate_omni_lm.py \
    --hf_model_path Qwen/Qwen2.5-Omni-7B \
    --video_path /path/to/video.mp4 \
    --prompt "What was the first sentence the boy said when he met the girl?" \
    --use_audio_in_video \
    --max_new_tokens 64 \
    --tp 2
```
