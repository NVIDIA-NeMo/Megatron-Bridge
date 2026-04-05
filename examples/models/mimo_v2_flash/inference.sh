#!/usr/bin/env bash
# MiMo-V2-Flash: text generation via Megatron
set -e

HF_MODEL=XiaomiMiMo/MiMo-V2-Flash

uv run python examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path ${HF_MODEL} \
    --prompt "Explain the concept of mixture of experts in neural networks:" \
    --trust-remote-code
