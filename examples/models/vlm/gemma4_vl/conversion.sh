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

# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# gemma4 requires transformers>=5.5.0; the lockfile pins 5.3.0.
# Upgrade first, then use --no-sync so uv run does not revert the upgrade.
uv pip install -q --upgrade 'transformers>=5.5.0' mistral_common

# Import HF → Megatron (base model)
uv run --no-sync python examples/conversion/convert_checkpoints.py import \
    --hf-model google/gemma-4-26B-A4B \
    --megatron-path ${WORKSPACE}/models/gemma-4-26B-A4B

# Import HF → Megatron (instruction-tuned model, needed for inference.sh step 3)
uv run --no-sync python examples/conversion/convert_checkpoints.py import \
    --hf-model google/gemma-4-26B-A4B-it \
    --megatron-path ${WORKSPACE}/models/gemma-4-26B-A4B-it

# Export Megatron → HF
uv run --no-sync python examples/conversion/convert_checkpoints.py export \
    --hf-model google/gemma-4-26B-A4B \
    --megatron-path ${WORKSPACE}/models/gemma-4-26B-A4B/iter_0000000 \
    --hf-path ${WORKSPACE}/models/gemma-4-26B-A4B-hf-export

# Round-trip validation (multi-GPU)
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
      --hf-model-id google/gemma-4-26B-A4B --tp 2 --pp 2
