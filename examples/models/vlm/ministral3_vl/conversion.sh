# Import HF → Megatron
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model mistralai/Ministral-3-3B-Instruct-2512-BF16 \
    --megatron-path /models/Ministral-3-3B-Instruct-2512-BF16

# Export Megatron → HF
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model mistralai/Ministral-3-3B-Instruct-2512 \
    --megatron-path /models/Ministral-3-3B-Instruct-2512-BF16/iter_0000000 \
    --hf-path /models/Ministral-3-3B-Instruct-2512-BF16-hf-export \
    --not-strict # To avoid "*.extra_state" warnings

# Round-trip validation
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id mistralai/Ministral-3-3B-Instruct-2512-BF16 --tp 2 --pp 2
