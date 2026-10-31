# MiniCPM-o 4.5 understanding

This integration combines MiniCPM-o 4.5's Hugging Face vision/audio encoders and
projectors with Megatron's Qwen3-compatible language decoder. It supports text,
image-text, audio-text, and combined image/audio understanding inputs.

The published checkpoint is `openbmb/MiniCPM-o-4_5`. Its understanding stage has
approximately 9B parameters. TTS weights are excluded, and exported configurations
set `init_tts=false`. This is not a speech-generation or streaming model.

## Reused components

`MiniCPMOBridge` inherits Qwen3's configuration and weight mappings; its builder
delegates decoder construction to `GPTModelBuilder`. QKV and gated-MLP conversion
use the shared mapping classes. Replicated perception gradients use the same
`hook_hf_module_setattr_for_tp_grad_sync` utility as Qwen-VL and Qwen-Omni.

The perception modules remain MiniCPM-o's native vision encoder, resampler,
Whisper audio encoder, and projectors. MiniCPM-o uses ordinary Qwen3 RoPE and
per-sample media spans, so it does not use Qwen-VL's MRoPE/DeepStack decoder or
replace its encoders with Qwen encoders. The audio adapter only translates the
native encoder's legacy return contract and delegates attention math to HF.

## Model construction

Use the builder API, with explicit permission to execute the checkpoint's custom
code:

```python
import os
import torch
from megatron.bridge import AutoBridge

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
torch.distributed.init_process_group("nccl")

bridge = AutoBridge.from_hf_pretrained(
    "openbmb/MiniCPM-o-4_5", trust_remote_code=True
)
config = bridge.get_model_config()
config.tensor_model_parallel_size = 2
config.pipeline_model_parallel_size = 1
config.context_parallel_size = 1
config.sequence_parallel = False
config.bf16 = True
config.params_dtype = torch.bfloat16
models = bridge.get_model(config, load_weights=True, wrap_with_ddp=False)
```

Run distributed construction with one process per GPU. The model config exposes
independent `freeze_language_model`, `freeze_vision_model`, `freeze_audio_model`,
`freeze_vision_projection`, and `freeze_audio_projection` flags. By default,
perception modules are frozen and the language decoder is trainable. Tensor-parallel
ranks synchronize gradients of replicated perception parameters when they are
trainable.

The forward method accepts dense token IDs, positions, a boolean causal attention
mask, and `media` produced with the checkpoint processor. Media boundaries are
per sample; they must not be flattened across a batch. Images use `pixel_values`,
`tgt_sizes`, and `image_bound`; audio uses `audio_features`, `audio_feature_lens`,
and `audio_bounds`.

## Checkpoint conversion

See the [conversion example](../../../examples/models/minicpmo/README.md) for
shared CLI import/export commands. Keep the original checkpoint's remote code,
tokenizer, and processor available when exporting. The exported checkpoint
contains the understanding stage only. Strict export excludes only the declared
`tts.` source module, including when TTS and understanding tensors share a source
shard. Missing language, vision, audio, or projector weights still fail export.

## Current scope

- MiniCPM-o **4.5** only; 2.6 is rejected rather than treated as Qwen3.
- Dense, non-streaming training forwards with TP; PP=CP=1 and SP disabled.
- No packed sequences, MTP, quantized weights, or incremental Megatron inference
  cache. Generation through a separate rollout backend is a separate integration.
- No SFT/LoRA recipe or speech-generation support is implied by the model entry.

Exact checkpoint round-trips and forward correlation are different checks. BF16
implementations can choose different top tokens when their logits are close;
weight equality alone does not establish forward or RL correctness.
