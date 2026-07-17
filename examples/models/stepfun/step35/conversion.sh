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

# Workspace directory for checkpoints and logs
WORKSPACE=${WORKSPACE:-/workspace}

# HF model id or a local snapshot path
HF_MODEL=${HF_MODEL:-stepfun-ai/Step-3.5-Flash}
MEGATRON_CKPT_PATH=${MEGATRON_CKPT_PATH:-${WORKSPACE}/models/stepfun-ai/Step-3.5-Flash}
LOG_DIR=${LOG_DIR:-${WORKSPACE}/logs}

mkdir -p "${LOG_DIR}" "$(dirname "${MEGATRON_CKPT_PATH}")"

echo "[convert] HF model:        ${HF_MODEL}"
echo "[convert] Megatron output: ${MEGATRON_CKPT_PATH}"
echo "[convert] Log dir:         ${LOG_DIR}"

# Single-rank import: the bridge handles per-expert / per-layer mapping itself,
# so a single process is enough for the conversion step.
./scripts/conversion/convert.sh import \
    --hf-model "${HF_MODEL}" \
    --megatron-path "${MEGATRON_CKPT_PATH}" \
    2>&1 | tee "${LOG_DIR}/convert_step35_megatron.log"

echo "[convert] Done. Checkpoint saved to: ${MEGATRON_CKPT_PATH}"
echo "[convert] Inference example: MEGATRON_MODEL_PATH=${MEGATRON_CKPT_PATH}/iter_0000000 bash examples/models/stepfun/step35/inference.sh"

# Export Megatron -> HF (uncomment to round-trip)
# ./scripts/conversion/convert.sh export \
#     --hf-model "${HF_MODEL}" \
#     --megatron-path "${MEGATRON_CKPT_PATH}/iter_0000000" \
#     --hf-path "${MEGATRON_CKPT_PATH}-hf-export"
