# ERNIE 5.5

**ERNIE 5.5 Flash** is an efficiency-focused model in the ERNIE 5.5 family: a sparse Mixture-of-Experts (MoE) language model with **280B total parameters and 14B actived parameters per token**, pre-trained from scratch on **33.5T tokens** with a training context of up to **128K**; fine-tuning and inference support a **1M-token** context. Different domains, reasoning tasks and agentic applications call for different data, training objectives and behavioral policies; by releasing the weights at the end of pre-training, we let developers build on the model's general capabilities and choose for themselves where to take it next. Because further training and agentic use keep adding to compute and memory cost, we made efficiency a key design goal from the start, revisiting each component, from attention and residual connections to the MoE layers and the output head, and treating activated compute, KV cache and long-context cost as part of one overall design. The same long-term view shaped the data and the training system: pre-training data emphasizes capabilities that are hard to build in afterwards, such as repository-scale code and long-context understanding, and the training system was co-designed with the architecture so that its efficiency choices also train efficiently at scale.

**ERNIE-5.5-Flash-Base** is the checkpoint at the end of pre-training, before any mid- or post-training, and can serve as a common starting point for domain adaptation, instruction tuning and reinforcement-learning research. Under a unified evaluation pipeline it delivers strong results across knowledge, reasoning, math, code, multilingual and long-context benchmarks.

Megatron Bridge supports ERNIE 5.5 (`model_type: ernie5_5_moe`) with Hugging Face ↔ Megatron checkpoint conversion
and supervised fine-tuning on NVIDIA GB300. For requirements, data preparation, training and export to Hugging Face
format, see the
[ERNIE 5.5 examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/ernie55/README.md).
The examples also cover a fine-tuning recipe for NVIDIA GB200.
