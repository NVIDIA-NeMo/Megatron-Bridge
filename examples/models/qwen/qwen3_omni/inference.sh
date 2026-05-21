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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

WORKSPACE=${WORKSPACE:-${ROOT_DIR}/.cache/qwen3_omni_examples}
SMOKE_MODEL_NAME=${SMOKE_MODEL_NAME:-Qwen3-Omni-30B-A3B-Instruct-smoke}
TMPDIR=${TMPDIR:-${WORKSPACE}/tmp}
HF_HOME=${HF_HOME:-${WORKSPACE}/hf_home}
PYTHONPATH=${PYTHONPATH:-${ROOT_DIR}/src:${ROOT_DIR}/3rdparty/Megatron-LM}

HF_SMOKE_PATH=${HF_SMOKE_PATH:-${WORKSPACE}/hf/${SMOKE_MODEL_NAME}}
MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${WORKSPACE}/megatron/${SMOKE_MODEL_NAME}/iter_0000000}
HF_EXPORT_PATH=${HF_EXPORT_PATH:-${WORKSPACE}/export/${SMOKE_MODEL_NAME}}

PROMPT=${PROMPT:-What is happening in this video?}
VIDEO_PATH=${VIDEO_PATH:-}
VIDEO_URL=${VIDEO_URL:-}
USE_AUDIO_IN_VIDEO=${USE_AUDIO_IN_VIDEO:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-50}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-1}
ETP=${ETP:-1}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}
DRY_RUN=${DRY_RUN:-0}

export TMPDIR HF_HOME PYTHONPATH
mkdir -p "${TMPDIR}" "${HF_HOME}"
cd "${ROOT_DIR}"

if [[ -z "${VIDEO_PATH}" && -z "${VIDEO_URL}" ]]; then
  echo "Either VIDEO_PATH or VIDEO_URL must be provided for omni inference." >&2
  exit 1
fi

if [[ -n "${VIDEO_PATH}" && -n "${VIDEO_URL}" ]]; then
  echo "Please set only one of VIDEO_PATH or VIDEO_URL." >&2
  exit 1
fi

if [[ -n "${VIDEO_PATH}" && ! -f "${VIDEO_PATH}" ]]; then
  echo "VIDEO_PATH does not exist: ${VIDEO_PATH}" >&2
  exit 1
fi

COMMON_ARGS=(
  --prompt "${PROMPT}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --tp "${TP}"
  --pp "${PP}"
  --ep "${EP}"
  --etp "${ETP}"
)

if [[ -n "${VIDEO_PATH}" ]]; then
  COMMON_ARGS+=(--video_path "${VIDEO_PATH}")
else
  COMMON_ARGS+=(--video_url "${VIDEO_URL}")
fi

if [[ "${USE_AUDIO_IN_VIDEO}" == "1" ]]; then
  COMMON_ARGS+=(--use_audio_in_video)
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  COMMON_ARGS+=(--trust_remote_code)
fi

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] %q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

run_cmd uv run python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
  examples/conversion/hf_to_megatron_generate_omni_lm.py \
  --hf_model_path "${HF_SMOKE_PATH}" \
  "${COMMON_ARGS[@]}"

run_cmd uv run python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
  examples/conversion/hf_to_megatron_generate_omni_lm.py \
  --hf_model_path "${HF_SMOKE_PATH}" \
  --megatron_model_path "${MEGATRON_MODEL_PATH}" \
  "${COMMON_ARGS[@]}"

run_cmd uv run python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" \
  examples/conversion/hf_to_megatron_generate_omni_lm.py \
  --hf_model_path "${HF_EXPORT_PATH}" \
  "${COMMON_ARGS[@]}"
