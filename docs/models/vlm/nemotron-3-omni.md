# Nemotron-3 Nano Omni

[NVIDIA Nemotron-3 Nano Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
is a 30B-A3B MoE multimodal reasoning model that jointly processes image,
video, audio, and text inputs. It pairs a MoE Mamba/attention hybrid
language backbone with a RADIO
vision tower (static-resolution image path or dynamic-resolution
temporal video embedder) and a Parakeet sound encoder.

NeMo Megatron Bridge supports HF↔Megatron conversion, full SFT, and LoRA
PEFT on image-text and audio-video-text datasets. The finetuned model can
be re-exported to 🤗 Hugging Face format for downstream evaluation or
deployment.

```{important}
**Day-0 release.** This model is supported on dedicated public branches:

| Repo | Branch |
|---|---|
| Megatron-Bridge | [`nemotron_3_omni`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/nemotron_3_omni) |
| Megatron-LM (submodule at `3rdparty/Megatron-LM`) | [`nemotron_3_omni`](https://github.com/NVIDIA/Megatron-LM/tree/nemotron_3_omni) |
```

For the full setup, conversion, inference, training, evaluation, and LoRA
merge / adapter export workflows, see
[`examples/models/vlm/nemotron_3_omni/README.md`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/nemotron_3_omni/examples/models/vlm/nemotron_3_omni/README.md).
