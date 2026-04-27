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

# ==============================================================================
# Nemotron-3 Nano Omni - Checkpoint conversion (HF <-> Megatron) + roundtrip
# ==============================================================================

set -xeuo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
HF_MODEL_ID=${HF_MODEL_ID:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}
MODEL_NAME=$(basename "$HF_MODEL_ID")

MEGATRON_PATH=${WORKSPACE}/models/${MODEL_NAME}
HF_EXPORT_PATH=${WORKSPACE}/models/${MODEL_NAME}-hf-export

# Import HF -> Megatron
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model "$HF_MODEL_ID" \
    --megatron-path "$MEGATRON_PATH" \
    --trust-remote-code

# Export Megatron -> HF (--not-strict allows 4 expected-missing tensors that
# are regenerated from config on the HF side: sound_encoder featurizer fb/window
# and vision_model input_conditioner norm_mean/norm_std)
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model "$HF_MODEL_ID" \
    --megatron-path "$MEGATRON_PATH" \
    --hf-path "$HF_EXPORT_PATH" \
    --not-strict

# Round-trip validation (multi-GPU): HF -> Megatron -> HF and compare weights.
# The roundtrip script appends ${MODEL_NAME} to --output-dir, so passing
# ${HF_EXPORT_PATH} here keeps the re-exported HF checkpoint nested under
# the same -hf-export tree (instead of overwriting ${MEGATRON_PATH}).
uv run python -m torch.distributed.run --nproc_per_node=4 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id "$HF_MODEL_ID" \
    --output-dir "${HF_EXPORT_PATH}" \
    --trust-remote-code \
    --tp 2 --ep 2 --not-strict
