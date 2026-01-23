# Import HF → Megatron
python examples/conversion/convert_checkpoints.py import \
    --hf-model google/gemma-3-4b-it \
    --megatron-path /models/gemma-3-4b-it

# Export Megatron → HF
python examples/conversion/convert_checkpoints.py export \
    --hf-model google/gemma-3-4b-it \
    --megatron-path /models/gemma-3-4b-it/iter_0000000 \
    --hf-path ./gemma3-vl-hf-export

# Round-trip validation
torchrun --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
      --hf-model-id google/gemma-3-4b-it --tp 2 --pp 2
