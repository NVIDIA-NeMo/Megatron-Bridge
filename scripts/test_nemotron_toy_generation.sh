#!/bin/bash
# Test Llama Nemotron generation with toy model on 2 GPUs

set -e

echo "=========================================="
echo "Llama Nemotron Generation Test (Toy Model)"
echo "=========================================="

# First create a toy model
echo "Creating toy model..."
python3 << 'EOF'
import json
import torch
from pathlib import Path
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

# Toy model config (same as in tests)
config_dict = {
    "architectures": ["LlamaForCausalLM"],
    "head_dim": 128,
    "hidden_size": 1024,
    "intermediate_size": 2816,
    "max_position_embeddings": 8192,
    "num_attention_heads": 8,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 32000,
}

model_dir = Path("/tmp/llama_nemotron_toy")
model_dir.mkdir(exist_ok=True)

# Create and save model
config = LlamaConfig(**config_dict)
model = LlamaForCausalLM(config).bfloat16()
model.save_pretrained(model_dir, safe_serialization=True)

# Save config
with open(model_dir / "config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

# Create minimal tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(model_dir)
except:
    tokenizer_config = {
        "tokenizer_class": "GPT2Tokenizer",
        "vocab_size": 32000,
    }
    with open(model_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

print(f"✓ Toy model saved to {model_dir}")
EOF

echo ""
echo "Testing generation with TP=2..."
echo ""

# Run with torchrun
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    scripts/test_nemotron_generation.py \
    --model-id /tmp/llama_nemotron_toy \
    --tp 2 \
    --pp 1 \
    --prompt "The capital of France is"

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="






