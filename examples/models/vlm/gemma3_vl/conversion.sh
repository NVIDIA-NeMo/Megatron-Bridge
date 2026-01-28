# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# Import HF → Megatron
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model google/gemma-3-4b-it \
    --megatron-path ${WORKSPACE}/models/gemma-3-4b-it

# Export Megatron → HF
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model google/gemma-3-4b-it \
    --megatron-path ${WORKSPACE}/models/gemma-3-4b-it/iter_0000000 \
    --hf-path ${WORKSPACE}/models/gemma-3-4b-it-hf-export

# Round-trip validation
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
      --hf-model-id google/gemma-3-4b-it --tp 2 --pp 2
