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
#
# CI_TIMEOUT=60
# GPU_COUNT=8
set -xeuo pipefail

readonly TRAINER_SCRIPT="tests/special_e2e/run_ppo_trainer_megatron.sh"

cleanup() {
    rm -rf checkpoints
}
trap cleanup EXIT

## Use our checkout
pip3 install git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@main --no-deps --no-build-isolation
pip3 install git+https://github.com/NVIDIA/Megatron-LM.git@main --no-deps --no-build-isolation
pip3 install "nvidia-modelopt[torch]>=0.37.0"
## cd to verl checkout root
pip3 install -r requirements-test.txt
pip3 install -r requirements.txt
pip3 install --no-deps -e .
pip3 install math-verify

python3 examples/data_preprocess/gsm8k.py --local_dataset_path "${HOME}/models/hf_data/gsm8k"

cleanup

ray stop --force
ALL_OFFLOAD=True \
VAL_BEFORE_TRAIN=True \
TEST_FREQ=1 \
SAVE_FREQ=1 \
LR_WARMUP_STEPS=1 \
TOTAL_TRAIN_STEPS=2 \
COMMON_VPP=2 \
MODEL_ID=Qwen/Qwen3-0.6B \
USE_MBRIDGE=True \
VANILLA_MBRIDGE=False \
VALUE_VANILLA_MBRIDGE=False \
bash "${TRAINER_SCRIPT}"

ray stop --force
ALL_OFFLOAD=False \
USE_MBRIDGE=True \
VANILLA_MBRIDGE=False \
USE_MEGATRON_FSDP=True \
TOTAL_TRAIN_STEPS=2 \
MODEL_ID=Qwen/Qwen3-0.6B \
COMMON_PP=1 \
COMMON_VPP=null \
COMMON_CP=1 \
COMMON_TP=1 \
INFER_TP=1 \
bash "${TRAINER_SCRIPT}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    ++critic.megatron.override_transformer_config.gradient_accumulation_fusion=False
