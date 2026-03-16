#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# Kimi K2.5 VL — Toy Model Compare Verification
#
# Creates a tiny Kimi K2.5 VL model (2 layers, 4 experts) and verifies that
# HF and Megatron produce matching logits via compare.py.
#
# Prerequisites:
#   - Access to a Kimi K2.5 config directory with custom .py modeling files
#   - Single node with 8 GPUs
#
# Usage (interactive inside container):
#   bash examples/models/vlm/kimi_k25/verify_compare.sh
# ==============================================================================
set -euo pipefail

WORKDIR="${WORKDIR:-/opt/Megatron-Bridge}"
cd "$WORKDIR"

# Source config directory (must contain config.json + custom .py files)
SOURCE_DIR="${SOURCE_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/aot/kimi/kimi-k2.5-small}"
TOY_DIR="${TOY_DIR:-/tmp/kimi_k25_toy}"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
export HF_HOME="${HF_HOME:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/yuya/HF_HOME}"

echo "======================================"
echo "Step 1: Creating toy Kimi K2.5 VL model"
echo "======================================"
uv run python examples/models/vlm/kimi_k25/create_toy_model.py \
    --source-dir "$SOURCE_DIR" \
    --output-dir "$TOY_DIR" \
    --num-hidden-layers 2 \
    --num-experts 4 \
    --num-experts-per-tok 2 \
    --vt-num-hidden-layers 4

echo ""
echo "======================================"
echo "Step 2: Running compare.py (text-only, single GPU)"
echo "======================================"
rm -rf nemo_experiments
uv run python examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path "$TOY_DIR" \
    --prompt "Hello, how are you?" \
    --trust_remote_code

echo ""
echo "======================================"
echo "Step 3: Running compare.py (text-only, TP=2 EP=4)"
echo "======================================"
rm -rf nemo_experiments
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path "$TOY_DIR" \
    --prompt "Hello, how are you?" \
    --trust_remote_code \
    --tp 2 --ep 4

echo ""
echo "======================================"
echo "Step 4: Running compare.py (with synthetic vision, EP=8)"
echo "======================================"
rm -rf nemo_experiments
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path "$TOY_DIR" \
    --prompt "Describe this image." \
    --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --trust_remote_code \
    --ep 8

echo ""
echo "======================================"
echo "All verification steps complete!"
echo "======================================"
