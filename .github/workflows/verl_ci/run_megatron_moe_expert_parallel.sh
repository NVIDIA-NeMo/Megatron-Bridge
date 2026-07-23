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

common_env=(
    ADV_ESTIMATOR=grpo
    USE_DUMMY_MODEL=True
    DUMMY_MODEL_CONFIG_PATH=tests/special_e2e/ppo_trainer/expert_parallel/qwen2moe_minimal.json
    PPO_MAX_TOKEN_LEN=1024
    FWD_MAX_TOKEN_LEN=1024
    MAX_PROMPT_LENGTH=512
    MAX_RESPONSE_LENGTH=512
    MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507
    USE_MBRIDGE=True
    VANILLA_MBRIDGE=False
    VALUE_VANILLA_MBRIDGE=False
    COMMON_PP=2
    COMMON_VPP=null
    COMMON_CP=1
    COMMON_TP=4
    COMMON_ETP=1
    USE_DIST_CKPT=False
    ALL_OFFLOAD=True
    SKIP_SAVE_HF_MODEL=1
)

ray stop --force
env "${common_env[@]}" \
    COMMON_EP=4 \
    INFER_TP=8 \
    bash "${TRAINER_SCRIPT}"

ray stop --force
env "${common_env[@]}" \
    COMMON_EP=4 \
    INFER_TP=2 \
    ROLLOUT_QUANTIZATION=fp8 \
    bash "${TRAINER_SCRIPT}"

cleanup

ray stop --force
env "${common_env[@]}" \
    LORA_RANK=8 \
    CRITIC_LORA_RANK=8 \
    COMMON_EP=2 \
    INFER_TP=8 \
    LORA_MERGE=True \
    bash "${TRAINER_SCRIPT}"
