#!/usr/bin/env bash
# MiMo-V2-Flash: HF <-> Megatron conversion
set -e

WORKSPACE=${WORKSPACE:-/workspace}
MODEL_NAME=MiMo-V2-Flash
HF_MODEL=XiaomiMiMo/${MODEL_NAME}
TP=2; PP=4; EP=32

# Import HF -> Megatron
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model ${HF_MODEL} \
    --megatron-path ${WORKSPACE}/${MODEL_NAME} \
    --torch-dtype bfloat16 \
    --trust-remote-code

# Compare logits (requires multi-GPU)
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path ${HF_MODEL} \
    --megatron_model_path ${WORKSPACE}/${MODEL_NAME} \
    --prompt "Hello, how are you?" \
    --tp ${TP} --pp ${PP} --ep ${EP}

# Export Megatron -> HF
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model ${HF_MODEL} \
    --megatron-path ${WORKSPACE}/${MODEL_NAME}/iter_0000000 \
    --hf-path ${WORKSPACE}/${MODEL_NAME}-hf-export

# Roundtrip validation
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id ${HF_MODEL} --tp ${TP} --pp ${PP} --ep ${EP} \
    --trust-remote-code
