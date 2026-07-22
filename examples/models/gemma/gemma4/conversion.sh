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

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# Force text-only bridge (Gemma4ForCausalLM / Gemma4DenseProvider).
# gemma-4-E4B-it is Gemma4ForConditionalGeneration in HF; without this flag
# the VL bridge is selected and vision/audio modules are imported unnecessarily.
export GEMMA4_CONVERSION_MODE=text

# Import HF → Megatron (Dense E4B base model)
./scripts/conversion/convert.sh import \
    --hf-model google/gemma-4-E4B-it \
    --megatron-path ${WORKSPACE}/models/gemma-4-E4B-it

# Export Megatron → HF
./scripts/conversion/convert.sh export \
    --hf-model google/gemma-4-E4B-it \
    --megatron-path ${WORKSPACE}/models/gemma-4-E4B-it/iter_0000000 \
    --hf-path ${WORKSPACE}/models/gemma-4-E4B-it-hf-export

# Round-trip validation
uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id google/gemma-4-E4B-it \
    --output-dir ${WORKSPACE}/results/gemma-4-E4B-it-roundtrip \
    --tp 2 --pp 1
