# Qwen2.5-Omni

Qwen2.5-Omni is a multimodal Qwen family model with text, image, video, and audio inputs. Megatron Bridge implements Qwen2.5-Omni under `megatron.bridge.models.qwen_omni`: `Qwen25OmniBridge` maps Hugging Face `Qwen2_5OmniForConditionalGeneration` checkpoints to `Qwen25OmniModel` / `Qwen25OmniModelProvider`.

The current implementation is focused on checkpoint conversion and training-oriented multimodal forward paths. It supports single-rank functional validation for text, vision, and audio inputs; distributed parallel validation beyond that is a follow-up.

## Current Support

- Hugging Face to Megatron Bridge checkpoint conversion for Qwen2.5-Omni checkpoints (e.g. `Qwen/Qwen2.5-Omni-7B`)
- Megatron Bridge to Hugging Face export for the same model family
- Multimodal weight mappings for the thinker (language, vision, audio tower, talker/token2wav passthrough where applicable)
- Recipe helpers for 7B-scale finetune and pretrain configurations (see `megatron.bridge.recipes.qwen_vl.qwen25_vl`)

## Known Limitations

- Megatron inference with `inference_params` may be incomplete depending on workload
- `packed_seq_params` is not fully exercised for Omni-specific paths
- Vision runtime may not support all parallelism modes; validate for your cluster
- Example smoke flows assume user-provided local multimodal assets (see `examples/models/vlm/qwen25_omni/`)

## Hugging Face Model Cards

- Qwen2.5-Omni-7B: `https://huggingface.co/Qwen/Qwen2.5-Omni-7B`

## Related Docs

- Related VLM: [Qwen2.5-VL](qwen2.5-vl.md)
- Related VLM: [Qwen3-VL](qwen3-vl.md)
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)
