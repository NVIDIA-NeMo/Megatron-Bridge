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

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

WORKSPACE=${WORKSPACE:-${ROOT_DIR}/.cache/qwen25_omni_examples}
SOURCE_HF_MODEL=${SOURCE_HF_MODEL:-}
SMOKE_MODEL_NAME=${SMOKE_MODEL_NAME:-Qwen2.5-Omni-7B-smoke}
TEXT_LAYERS=${TEXT_LAYERS:-2}
VISION_DEPTH=${VISION_DEPTH:-2}
AUDIO_LAYERS=${AUDIO_LAYERS:-2}
TMPDIR=${TMPDIR:-${WORKSPACE}/tmp}
HF_HOME=${HF_HOME:-${WORKSPACE}/hf_home}
PYTHON=${PYTHON:-python}
PYTHONPATH=${PYTHONPATH:-${ROOT_DIR}/src:${ROOT_DIR}/3rdparty/Megatron-LM}

HF_SMOKE_PATH=${HF_SMOKE_PATH:-${WORKSPACE}/hf/${SMOKE_MODEL_NAME}}
MEGATRON_PATH=${MEGATRON_PATH:-${WORKSPACE}/megatron/${SMOKE_MODEL_NAME}}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-${WORKSPACE}/export/${SMOKE_MODEL_NAME}}

export TMPDIR HF_HOME PYTHONPATH
mkdir -p "${WORKSPACE}/hf" "${WORKSPACE}/megatron" "${WORKSPACE}/export" "${TMPDIR}" "${HF_HOME}"
cd "${ROOT_DIR}"

if [[ -z "${SOURCE_HF_MODEL}" ]]; then
  echo "SOURCE_HF_MODEL must point to a local HF Qwen2.5-Omni checkpoint or model id." >&2
  exit 1
fi

"${PYTHON}" examples/models/vlm/qwen25_omni/create_smoke_checkpoint.py \
  --source-model-path "${SOURCE_HF_MODEL}" \
  --output-dir "${HF_SMOKE_PATH}" \
  --text-layers "${TEXT_LAYERS}" \
  --vision-depth "${VISION_DEPTH}" \
  --audio-layers "${AUDIO_LAYERS}"

"${PYTHON}" examples/conversion/convert_checkpoints.py import \
  --hf-model "${HF_SMOKE_PATH}" \
  --megatron-path "${MEGATRON_PATH}" \
  --torch-dtype bfloat16

"${PYTHON}" examples/conversion/convert_checkpoints.py export \
  --hf-model "${HF_SMOKE_PATH}" \
  --megatron-path "${MEGATRON_PATH}/iter_0000000" \
  --hf-path "${HF_EXPORT_PATH}"

echo "HF smoke checkpoint: ${HF_SMOKE_PATH}"
echo "Megatron checkpoint: ${MEGATRON_PATH}/iter_0000000"
echo "HF export checkpoint: ${HF_EXPORT_PATH}"
