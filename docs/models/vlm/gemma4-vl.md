# Gemma 4 VL (26B-A4B MoE)

[Google's Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it)
is a Mixture-of-Experts vision-language model (26B total, 4B active
parameters). It pairs a 128-expert top-k=8 MoE language backbone with a
SigLIP vision tower, dual sliding/global attention, and K=V tying on the
full-attention layers.

NeMo Megatron Bridge supports HF↔Megatron conversion, full SFT, and LoRA
PEFT on image-text datasets. The finetuned model can be re-exported to
🤗 Hugging Face format for downstream evaluation or deployment.

For the full setup, conversion, inference, training, and LoRA merge /
adapter export workflows, see
[`examples/models/vlm/gemma4_vl/README.md`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/vlm/gemma4_vl/README.md).
