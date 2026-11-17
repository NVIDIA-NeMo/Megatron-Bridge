# Qwen3-VL

Qwen3-VL is the latest generation of vision-language models from Alibaba Cloud, supporting multimodal understanding across text, images, and videos. Qwen3-VL includes both dense models and Mixture-of-Experts (MoE) variants for improved efficiency.

NeMo Megatron Bridge supports finetuning Qwen3-VL models (8B dense and 30B MoE variants).

```{tip}
We use the following environment variables throughout this page
- `HF_MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct` (or `Qwen/Qwen3-VL-30B-A3B-Instruct` for MoE)
- `MEGATRON_MODEL_PATH=/models/Qwen3-VL-8B-Instruct` (feel free to set your own path)
Unless explicitly stated, any megatron model path in the commands below should NOT contain the iteration number 
`iter_xxxxxx`. For more details on checkpointing, please see 
[here](../../training/checkpointing.md#checkpoint-contents)
```

## Examples

For checkpoint conversion, inference, finetuning recipes, and step-by-step training guides, see the [Qwen3-VL Examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/qwen/qwen3_vl/README.md).

## RoPE fusion

Qwen3-VL applies rotary position embeddings per token: the language model rotates
each token by its own multimodal (MRoPE) angles, and the vision encoder rotates
each patch by its 2D angles. With `model.apply_rope_fusion=True`, both run
Transformer Engine's fused RoPE kernel. The kernel computes in FP32 and rounds
once, the same math as `apply_rotary_pos_emb_in_fp32=True`, which the vision
encoder already uses. The vision encoder follows the language model's setting.
Inputs the kernel cannot take, such as tensors on the CPU, use the unfused
implementation, with one warning per reason. Fusion is off by default.

## Hugging Face Model Cards
- Qwen3-VL-8B: `https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct`
- Qwen3-VL-30B-A3B (MoE): `https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct`
- Qwen3-VL-235B-A22B (MoE): `https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct`
