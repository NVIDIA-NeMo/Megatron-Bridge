# Nemotron 3.5 Lightning

Nemotron 3.5 Lightning uses the same architecture as Nemotron 3 Nano, but its
checkpoint was trained natively with multi-token prediction (MTP).

Day-0 support for Nemotron 3.5 Lightning is available through the
`nvcr.io/nvidia/nemo:26.06.01` container plus the
[custom Megatron Bridge 0.5.1 branch and release README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/nemotron-3.5-lightning-mb-0.5.1/examples/models/nemotron/nemotron_3_5_lightning/README.md).
The model is also available on the
[`main`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main) and
[`r0.6.0`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/r0.6.0)
branches through the `nvcr.io/nvidia/nemo:26.08` container.

See the
[Nemotron 3.5 Lightning model verification card](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/model_verification_cards/nemotron-3.5-lightning/card.yaml)
for verification scripts and results.
