# EXAONE

Megatron Bridge supports Hugging Face conversion for EXAONE 4.0 dense language
models, EXAONE 4.5 vision-language models, and K-EXAONE MoE language models.
Checked-in H100 recipes cover EXAONE 4.0 1.2B, EXAONE 4.5 VL 33B,
K-EXAONE-236B-A23B, and K-EXAONE 2.0 750B-A37B.

Use the model-specific examples for commands and supported parallelism:

- [EXAONE 4.0](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/exaone/exaone4)
- [EXAONE 4.5 VL](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/exaone/exaone45)
- [K-EXAONE MoE](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/exaone/exaone_moe)

The EXAONE 4.0 and EXAONE 4.5 examples use the shared
`scripts/conversion/convert.sh import` entry point. The K-EXAONE examples use
their model-specific distributed round-trip workflows. K-EXAONE 2.0 also has a checked-in
[model verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/k-exaone-2/card.yaml)
that records the current verification status of individual workflows.
