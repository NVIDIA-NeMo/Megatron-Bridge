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

WORKSPACE=${WORKSPACE:-/nfs/ml-training-ssd/users/liuwei/qwen3_omni_examples}
SAMPLE_PARQUET=${SAMPLE_PARQUET:-/home/luban/liuwei/omni/ref/projects/Omni_Bench_fix_simple/test.parquet}
SMOKE_MODEL_NAME=${SMOKE_MODEL_NAME:-Qwen3-Omni-30B-A3B-Instruct-smoke}
TMPDIR=${TMPDIR:-${WORKSPACE}/tmp}
HF_HOME=${HF_HOME:-${WORKSPACE}/hf_home}
PYTHON=${PYTHON:-python}
PYTHONPATH=${PYTHONPATH:-src}

HF_SMOKE_PATH=${HF_SMOKE_PATH:-${WORKSPACE}/hf/${SMOKE_MODEL_NAME}}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${WORKSPACE}/megatron/${SMOKE_MODEL_NAME}/iter_0000000}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-${WORKSPACE}/export/${SMOKE_MODEL_NAME}}

export TMPDIR HF_HOME PYTHONPATH
mkdir -p "${TMPDIR}" "${HF_HOME}"

"${PYTHON}" examples/conversion/hf_to_megatron_qwen3_omni_smoke.py \
  --hf-model-path "${HF_SMOKE_PATH}" \
  --sample-parquet "${SAMPLE_PARQUET}"

"${PYTHON}" examples/conversion/hf_to_megatron_qwen3_omni_smoke.py \
  --hf-model-path "${HF_SMOKE_PATH}" \
  --megatron-model-path "${MEGATRON_MODEL_PATH}" \
  --sample-parquet "${SAMPLE_PARQUET}" \
  --tp 1 --pp 1 --ep 1 --etp 1

"${PYTHON}" examples/conversion/hf_to_megatron_qwen3_omni_smoke.py \
  --hf-model-path "${HF_EXPORT_PATH}" \
  --sample-parquet "${SAMPLE_PARQUET}"
