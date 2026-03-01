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

# MiniMax-M2 (MoE: 256 experts, top-8, ~230GB fp8)
# Requires multi-node for full model. Adjust TP/EP for available resources.
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path MiniMaxAI/MiniMax-M2 \
    --prompt "What is artificial intelligence?" \
    --max_new_tokens 100 \
    --tp 2 --ep 4 \
    --trust-remote-code
