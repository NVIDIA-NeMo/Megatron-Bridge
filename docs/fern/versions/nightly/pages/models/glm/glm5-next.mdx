# GLM-5.3-Flash

GLM-5.3-Flash is a GLM vision-language model. Megatron Bridge registers its Hugging Face architecture, `Glm5NextForConditionalGeneration` (`model_type` `glm5_next`), through a dedicated `GLM53FlashBridge`. GLM-5.3-Flash is a different architecture from GLM-5.3 and is not handled by `GLM5Bridge`; for GLM-5.3, see the [GLM-5.2 / GLM-5.3 guide](glm5-2.md).

## Architecture Notes

- The language model is built as a Megatron Core `HybridModel`. Each Hugging Face decoder layer becomes two physical layers: an attention layer (KDA linear attention or Dynamic Sparse Attention with Multi-Latent Attention) followed by an FFN layer (dense MLP or MoE). The layer pattern is derived from the checkpoint's `layer_types` and `mlp_layer_types`.
- Every decoder layer is wrapped in mHC multi-stream hyper-connections.
- MoE layers use sigmoid routing with expert bias and an always-on shared expert.
- Multi-token prediction (MTP) layers in the checkpoint map to a repeated DSA + MoE MTP block.
- The Hugging Face `Glm5NextVisionModel` vision tower is loaded replicated on the stages that own the embeddings. Image and video features are inserted into the language-model embeddings; when both `pixel_values` and `pixel_values_videos` are absent, the model runs text-only.
- Block-scaled FP8 checkpoint weights are dequantized on import. Exported checkpoints contain plain weights without the `_scale_inv` tensors.
- `GLM53FlashModelProvider` exposes `freeze_language_model`, `freeze_vision_model`, and `freeze_vision_projection` for parameter-efficient training.
- The Hugging Face model classes come from Transformers; use the repository's supported dependency versions.

## Usage

`AutoBridge` resolves a `glm5_next` checkpoint to `GLM53FlashBridge`, and `get_model_config()` returns a `GLM53FlashModelConfig`:

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("/path/to/glm-5.3-flash")
model_config = bridge.get_model_config()
```

GLM-5.3-Flash does not yet have training recipes, an examples README, or a verification card.

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/glm5next`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/glm5next)
- Unit tests: [`tests/unit_tests/models/glm5next`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/tests/unit_tests/models/glm5next)
