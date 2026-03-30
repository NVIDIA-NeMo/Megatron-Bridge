#!/usr/bin/env bash
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
set -euo pipefail

# ==============================================================================
# Qwen3.5 LLM-only SFT on SQuAD (HF VLM checkpoint, vision weights skipped)
#
# Uses scripts/training/run_recipe.py with:
#   - Recipes from megatron.bridge.recipes (qwen35_llm_*_sft_config)
#   - --step_func vlm_step (text-only batches; no images)
#
# From the Bridge repository root:
#   ./examples/models/vlm/qwen35_vl/llm_sft.sh
#
# Environment overrides:
#   NPROC_PER_NODE  GPUs per node (default: 8)
#   RECIPE          Recipe name (default: qwen35_llm_800m_sft_config)
#   HF_PATH         HF model id or local directory (default: Qwen/Qwen3.5-0.8B)
#   EXTRA_ARGS      Extra args passed to run_recipe.py (Hydra-style overrides)
#
# Examples:
#   RECIPE=qwen35_llm_2b_sft_config HF_PATH=Qwen/Qwen3.5-2B ./examples/models/vlm/qwen35_vl/llm_sft.sh
#
#   EXTRA_ARGS="train.train_iters=500 checkpoint.save=/tmp/qwen35_llm_ckpt" \
#     ./examples/models/vlm/qwen35_vl/llm_sft.sh
#
# Optional: pin dataset via run_recipe (default SQuAD matches the recipe):
#   EXTRA_ARGS="--dataset llm-finetune dataset.dataset_name=squad" \
#     ./examples/models/vlm/qwen35_vl/llm_sft.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RECIPE="${RECIPE:-qwen35_llm_800m_sft_config}"
HF_PATH="${HF_PATH:-Qwen/Qwen3.5-0.8B}"

# shellcheck disable=SC2086
exec uv run torchrun --nproc_per_node="${NPROC_PER_NODE}" scripts/training/run_recipe.py \
  --recipe "${RECIPE}" \
  --step_func vlm_step \
  --hf_path "${HF_PATH}" \
  ${EXTRA_ARGS:-}
