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

# Use text-only bridge so inference goes through GPTModel, not Gemma4VLModel.
# gemma-4-E4B-it is Gemma4ForConditionalGeneration in HF; without this flag the
# VL bridge is selected and the full VL model is loaded for every text inference call.
export GEMMA4_CONVERSION_MODE=text

# Prompts use the Gemma 4 IT chat template so the instruction-tuned model
# produces coherent answers.  The base model (gemma-4-E4B) accepts raw text
# completions; the IT model requires this wrapping to avoid repetitive output.
PROMPT1="<start_of_turn>user
What is the capital of France?<end_of_turn>
<start_of_turn>model
"

PROMPT2="<start_of_turn>user
Explain the concept of entropy in simple terms.<end_of_turn>
<start_of_turn>model
"

# Inference directly from HuggingFace checkpoint (text only)
uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path google/gemma-4-E4B-it \
    --prompt "${PROMPT1}" \
    --max_new_tokens 20 \
    --tp 2 \
    --pp 1

# Inference from imported Megatron checkpoint
# Requires conversion.sh to have been run first (step 1 imports the model).
uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path google/gemma-4-E4B-it \
    --megatron_model_path ${WORKSPACE}/models/gemma-4-E4B-it/iter_0000000 \
    --prompt "${PROMPT2}" \
    --max_new_tokens 50 \
    --tp 2 \
    --pp 1
