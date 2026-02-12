# Qwen2-Audio Megatron Bridge

This directory contains examples for using Qwen2-Audio models with Megatron-Bridge.

## Inference

### 1. Direct Inference from HuggingFace

```bash
# Run inference with audio from URL
uv run python -m torch.distributed.run --nproc_per_node=2 examples/conversion/hf_to_megatron_generate_audio_lm.py \
  --hf_model_path "Qwen/Qwen2-Audio-7B-Instruct" \
  --audio_url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3" \
  --prompt "What's that sound?" \
  --max_new_tokens 50 --tp 2
```

### 2. Convert to Megatron Format

```bash
# Convert HuggingFace checkpoint to Megatron format
uv run python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen2-Audio-7B-Instruct \
  --megatron-path /workspace/models/Qwen2-Audio-7B-Instruct
```

### 3. Run Inference from Megatron Checkpoint

```bash
# Run inference from converted Megatron checkpoint
uv run python -m torch.distributed.run examples/conversion/hf_to_megatron_generate_audio_lm.py \
  --hf_model_path "Qwen/Qwen2-Audio-7B-Instruct" \
  --megatron_model_path /workspace/models/Qwen2-Audio-7B-Instruct/iter_0000000 \
  --audio_url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3" \
  --prompt "What's that sound?" \
  --max_new_tokens 50
```

### Expected Outputs

```
Generation step 0
Step 0: output shape=torch.Size([1, 133, 156032]), var=10.6469
Top 5: [('It', 17.209945678710938), ('Sh', 15.799845695495605), ('Glass', 15.264859199523926), ('The', 14.33051586151123), ('That', 14.097867012023926)]
Selected: 'It' (id=2132)
Generation step 1
Step 1: output shape=torch.Size([1, 134, 156032]), var=10.6324
Top 5: [("'s", 21.237674713134766), (' is', 19.858510971069336), (' sounds', 17.625566482543945), (' was', 14.162233352661133), (' seems', 11.964351654052734)]
Selected: ''s' (id=594)
Generation step 2
Step 2: output shape=torch.Size([1, 135, 156032]), var=10.6352
Top 5: [(' the', 19.65313720703125), (' a', 15.226014137268066), (' glass', 14.023796081542969), (' sounds', 12.189066886901855), (' sh', 10.67518424987793)]
Selected: ' the' (id=279)
Generation step 3
Step 3: output shape=torch.Size([1, 136, 156032]), var=10.6320
Top 5: [(' sound', 21.06254768371582), (' sh', 13.064013481140137), (' crashing', 12.673016548156738), (' glass', 12.11919116973877), (' crash', 11.293011665344238)]
Selected: ' sound' (id=5112)
Generation step 4
Step 4: output shape=torch.Size([1, 137, 156032]), var=10.6132
Top 5: [(' of', 22.771255493164062), (' effect', 11.854891777038574), ('行动计划', 10.352171897888184), (' glass', 9.680936813354492), (' when', 9.299120903015137)]
Selected: ' of' (id=315)
Generation step 5
Generation step 6
Generation step 7
Generation step 8
======== GENERATED TEXT OUTPUT ========
Audio: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3
Prompt: What's that sound?
Generated: system
You are a helpful assistant.
user
Audio 1: 
What's that sound?
assistant
It's the sound of glass breaking.
```