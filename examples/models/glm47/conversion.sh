#!/usr/bin/env bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# GLM-4.7 / GLM-4.7-Flash Conversion Examples
#
# GLM-4.7 (Glm4MoeForCausalLM):  MoE, 160 experts top-8, ~358B params
# GLM-4.7-Flash (Glm4MoeLiteForCausalLM): MLA+MoE, 64 experts top-4, ~30B params
#
# GLM-4.7-Flash fits on a single 8-GPU node with EP=8.
# GLM-4.7 requires multi-node (see slurm_conversion.sh).
# ==============================================================================

set -xeuo pipefail

WORKSPACE=${WORKSPACE:-/workspace}

# ── GLM-4.7-Flash (single-node, ~30B, MLA+MoE) ─────────────────────────

GLM47_FLASH_HF=zai-org/GLM-4.7-Flash

# Single-GPU round-trip (small models only; Flash is ~30B so may OOM on 1 GPU)
# uv run python examples/conversion/hf_megatron_roundtrip.py \
#     --hf-model-id $GLM47_FLASH_HF

# Multi-GPU round-trip with EP=8
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id $GLM47_FLASH_HF --ep 8

# Multi-GPU round-trip with TP=2 EP=4
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id $GLM47_FLASH_HF --tp 2 --ep 4

# Import HF → Megatron checkpoint
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model $GLM47_FLASH_HF \
    --megatron-path ${WORKSPACE}/models/GLM-4.7-Flash

# Export Megatron → HF checkpoint
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model $GLM47_FLASH_HF \
    --megatron-path ${WORKSPACE}/models/GLM-4.7-Flash/iter_0000000 \
    --hf-path ${WORKSPACE}/models/GLM-4.7-Flash-hf-export
