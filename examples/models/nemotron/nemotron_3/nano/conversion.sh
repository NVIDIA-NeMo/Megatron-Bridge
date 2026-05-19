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

set -xeuo pipefail

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

MODEL_NAME=NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
HF_MODEL_ID=nvidia/$MODEL_NAME

# Import HF → Megatron
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model $HF_MODEL_ID \
    --megatron-path ${WORKSPACE}/models/$MODEL_NAME \
    --trust-remote-code

# Export Megatron → HF
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model $HF_MODEL_ID \
    --megatron-path ${WORKSPACE}/models/$MODEL_NAME/iter_0000000 \
    --hf-path ${WORKSPACE}/models/$MODEL_NAME-hf-export

# Round-trip validation
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id $HF_MODEL_ID \
    --megatron-load-path ${WORKSPACE}/models/$MODEL_NAME/iter_0000000 \
    --tp 2 --ep 8 \
    --trust-remote-code