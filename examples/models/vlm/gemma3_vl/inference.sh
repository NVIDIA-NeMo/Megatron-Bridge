# Inference with Hugging Face checkpoints
uv run torchrun --nproc_per_node=4 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path google/gemma-3-4b-it \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 100 \
    --tp 2 \
    --pp 2

# Inference with imported Megatron checkpoints
uv run torchrun --nproc_per_node=4 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path google/gemma-3-4b-it \
    --megatron_model_path /models/gemma-3-4b-it/iter_0000000 \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 100 \
    --tp 2 \
    --pp 2

# Inference with exported HF checkpoints
uv run torchrun --nproc_per_node=4 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path /models/gemma-3-4b-it-hf-export \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 100 \
    --tp 2 \
    --pp 2
