# Qwen3-Omni

Qwen3-Omni is a multimodal Qwen family model with text, image, video, and audio inputs. Megatron Bridge support for Qwen3-Omni reuses the existing Qwen3-VL language and vision path, and adds Qwen3-Omni-specific audio handling and checkpoint mappings.

The current implementation is focused on checkpoint conversion and training-oriented multimodal forward paths. It supports single-rank functional validation for text, vision, and audio inputs, and keeps distributed parallel validation as a follow-up step.

## Current Support

- Hugging Face to Megatron Bridge checkpoint conversion for `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Megatron Bridge to Hugging Face export for the same model family
- Text, image, video, and audio multimodal forward paths
- Qwen3-Omni-specific multimodal RoPE handling for Megatron Bridge runtime
- Single-GPU smoke validation with a vertically trimmed checkpoint

## Implementation Notes

- The language backbone uses Megatron-Core modules through the existing Qwen3-VL GPT path.
- The vision and audio towers currently reuse Hugging Face Qwen3-Omni components.
- Multimodal embeddings are fused before the language decoder.
- The current validation path focuses on correctness and checkpoint interoperability rather than distributed performance tuning.

## Validation

The current test coverage includes:

- Provider unit tests
- Bridge unit tests
- Model unit tests for text, vision, and audio paths
- Functional checkpoint conversion tests
- End-to-end smoke tests using user-provided multimodal assets

## Hugging Face Model Cards

- Qwen3-Omni-30B-A3B-Instruct: `https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct`

## Related Docs

- Related VLM: [Qwen3-VL](qwen3-vl.md)
- Related VLM: [Qwen 3.5](qwen35-vl.md)
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)
