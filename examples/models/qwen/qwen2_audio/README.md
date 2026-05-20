# Qwen2-Audio - Audio Language Model

This directory contains example scripts for Qwen2-Audio audio-language models.

## Inference

### Run Inference from HuggingFace Checkpoint

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/conversion/hf_to_megatron_generate_audio_lm.py \
  --hf_model_path Qwen/Qwen2-Audio-7B-Instruct \
  --audio_url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac" \
  --prompt "Describe what you hear in this audio." \
  --tp 2 \
  --max_new_tokens 50
```

Note:
- You can also use local audio files: `--audio_path /path/to/audio.wav`

See the [inference.sh](inference.sh) script for the full runnable commands.

**Expected output:**
```
======== GENERATED TEXT OUTPUT ========
Audio: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac
Prompt: Describe what you hear in this audio.
Generated: system
You are a helpful assistant.
user
Audio 1:
Describe what you hear in this audio.
assistant
I heard a man speaking in English with the phrase 'Mister Quiller is the apostle of the middle classes and we are glad to welcome his gospel.'
=======================================
```
