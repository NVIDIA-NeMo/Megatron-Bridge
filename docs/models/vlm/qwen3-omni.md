# Qwen3-Omni

Qwen3-Omni is a multimodal Qwen family model with text, image, video, and audio inputs. Megatron Bridge support for Qwen3-Omni reuses the existing Qwen3-VL language and vision path, and adds Qwen3-Omni-specific audio handling and checkpoint mappings.

The current implementation focuses on checkpoint conversion, training-oriented multimodal forward paths, and smoke-level validation. It includes a full example workflow (HF -> Megatron -> HF export, single-rank inference) and a multi-node training recipe entrypoint.

## Current Support

- Hugging Face to Megatron Bridge checkpoint conversion for `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Megatron Bridge to Hugging Face export for the same model family
- Text, image, video, and audio multimodal forward paths
- Qwen3-Omni-specific multimodal RoPE handling for Megatron Bridge runtime
- Single-GPU smoke validation with a vertically trimmed checkpoint
- Multi-node training recipe entrypoint (see Qwen3-Omni examples)
- L0 conversion test coverage for Qwen3-Omni

## Known Limitations

- Megatron inference with `inference_params` is not implemented yet
- `packed_seq_params` is not implemented yet
- Vision runtime does not support sequence parallel yet
- Distributed parallelism validation beyond single-rank functional coverage is not included yet
- Functional smoke tests require user-provided local multimodal assets

## Hugging Face Model Cards

- Qwen3-Omni-30B-A3B-Instruct: `https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct`

## Examples

Qwen3-Omni examples are maintained here:

- `examples/models/vlm/qwen3_omni/README.md`
- `examples/models/vlm/qwen3_omni/conversion.sh`
- `examples/models/vlm/qwen3_omni/inference.sh`
- `examples/models/vlm/qwen3_omni/local_train_thinker_4node_tp2_ep8_sp.sh`

## Related Docs

- Related VLM: [Qwen3-VL](qwen3-vl.md)
- Related VLM: [Qwen 3.5](qwen35-vl.md)
- Recipe usage: [Recipe usage](../../recipe-usage.md)
- Customizing the training recipe configuration: [Configuration overview](../../training/config-container-overview.md)
- Training entry points: [Entry points](../../training/entry-points.md)
