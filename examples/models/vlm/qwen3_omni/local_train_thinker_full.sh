#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)

export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/3rdparty/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

WORKSPACE=${WORKSPACE:-${ROOT_DIR}/.cache/qwen3_omni_train}
HF_HOME=${HF_HOME:-${WORKSPACE}/hf_home}
TMPDIR=${TMPDIR:-${WORKSPACE}/tmp}
RESULTS_DIR=${RESULTS_DIR:-${WORKSPACE}/results}
LOG_DIR=${LOG_DIR:-${WORKSPACE}/logs}
LOCAL_PY_SHIM_DIR=${LOCAL_PY_SHIM_DIR:-${WORKSPACE}/py_shims}

mkdir -p "${WORKSPACE}" "${HF_HOME}" "${TMPDIR}" "${RESULTS_DIR}" "${LOG_DIR}" "${LOCAL_PY_SHIM_DIR}"

cat > "${LOCAL_PY_SHIM_DIR}/sitecustomize.py" <<'PY'
"""Local training shims for py311 compatibility."""

import typing

if not hasattr(typing, "override"):
    from typing_extensions import override as _override

    typing.override = _override
PY

export PYTHONPATH="${LOCAL_PY_SHIM_DIR}:${PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}

HF_MODEL_PATH=${HF_MODEL_PATH:-}
THINKER_ONLY_MIRROR_DIR=${THINKER_ONLY_MIRROR_DIR:-${WORKSPACE}/hf_thinker_only}
LOCAL_DATA_ROOT=${LOCAL_DATA_ROOT:-}
TRAIN_JSONL=${TRAIN_JSONL:-}
VALID_JSONL=${VALID_JSONL:-}
TEST_JSONL=${TEST_JSONL:-}
DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-0}
DATASET_PERSISTENT_WORKERS=${DATASET_PERSISTENT_WORKERS:-False}

if [[ -z "${HF_MODEL_PATH}" ]]; then
  echo "[error] HF_MODEL_PATH is required (local HF checkpoint or model id)." >&2
  exit 1
fi

if [[ -n "${LOCAL_DATA_ROOT}" ]]; then
  TRAIN_JSONL=${TRAIN_JSONL:-${LOCAL_DATA_ROOT}/train/train.jsonl}
  VALID_JSONL=${VALID_JSONL:-${LOCAL_DATA_ROOT}/test/test.jsonl}
  TEST_JSONL=${TEST_JSONL:-${LOCAL_DATA_ROOT}/test/test.jsonl}
fi

if [[ -z "${TRAIN_JSONL}" ]]; then
  echo "[error] TRAIN_JSONL is required (set TRAIN_JSONL or LOCAL_DATA_ROOT)." >&2
  exit 1
fi

RECIPE=${RECIPE:-qwen3_omni_30b_a3b_sft_preloaded_config}
STEP_FUNC=${STEP_FUNC:-qwen3_omni_step}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
TRAIN_ITERS=${TRAIN_ITERS:-20}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
EVAL_ITERS=${EVAL_ITERS:-0}
EVAL_INTERVAL=${EVAL_INTERVAL:-0}
LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-False}
VIT_GRADIENT_CHECKPOINTING=${VIT_GRADIENT_CHECKPOINTING:-False}
MULTIMODAL_ATTN_IMPL=${MULTIMODAL_ATTN_IMPL:-auto}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto}
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-}
RECOMPUTE_MODULES=${RECOMPUTE_MODULES:-}
FREEZE_LANGUAGE_MODEL=${FREEZE_LANGUAGE_MODEL:-False}
FREEZE_VISION_MODEL=${FREEZE_VISION_MODEL:-False}
FREEZE_AUDIO_MODEL=${FREEZE_AUDIO_MODEL:-False}
OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-}
USE_PRECISION_AWARE_OPTIMIZER=${USE_PRECISION_AWARE_OPTIMIZER:-}
OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-}
OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-}
ALIGN_PARAM_GATHER=${ALIGN_PARAM_GATHER:-}
USE_PYTORCH_PROFILER=${USE_PYTORCH_PROFILER:-}
USE_NSYS_PROFILER=${USE_NSYS_PROFILER:-}
PROFILE_STEP_START=${PROFILE_STEP_START:-}
PROFILE_STEP_END=${PROFILE_STEP_END:-}
PROFILE_RANKS=${PROFILE_RANKS:-}
PYTORCH_PROFILER_COLLECT_SHAPES=${PYTORCH_PROFILER_COLLECT_SHAPES:-}
PYTORCH_PROFILER_COLLECT_CALLSTACK=${PYTORCH_PROFILER_COLLECT_CALLSTACK:-}
PYTORCH_PROFILER_COLLECT_CHAKRA=${PYTORCH_PROFILER_COLLECT_CHAKRA:-}
RECORD_MEMORY_HISTORY=${RECORD_MEMORY_HISTORY:-}
NVTX_RANGES=${NVTX_RANGES:-}
NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-${DISTRIBUTED_NODE_COUNT:-${WORLD_SIZE:-1}}}
TASK_ROLE=${DISTRIBUTED_TASK_ROLE:-}
TASK_INDEX=${VC_TASK_INDEX:-}
NODE_RANK=${NODE_RANK:-}
MASTER_ADDR=${MASTER_ADDR:-${VC_MASTER_HOSTS:-${MASTER_HOST:-${TORCH_MASTER_ADDR:-127.0.0.1}}}}
MASTER_PORT=${MASTER_PORT:-${PET_MASTER_PORT:-${TORCH_MASTER_PORT:-29500}}}
WANDB_MODE=${WANDB_MODE:-disabled}
RUN_NAME=${RUN_NAME:-qwen3_omni_full_train_baseline}
ENABLE_PERF_SUMMARY=${ENABLE_PERF_SUMMARY:-1}
PERF_TAG=${PERF_TAG:-full}
LOG_PATH=${LOG_PATH:-${LOG_DIR}/${RUN_NAME}_${PERF_TAG}.log}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${RESULTS_DIR}/${RUN_NAME}/tb_logs}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export WANDB_MODE HF_HOME TMPDIR PYTORCH_CUDA_ALLOC_CONF
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-}
export MAX_PIXELS=${MAX_PIXELS:-}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-}

if [[ "${MASTER_ADDR}" == *,* ]]; then
    MASTER_ADDR=${MASTER_ADDR%%,*}
fi

if [[ -z "${NODE_RANK}" ]]; then
    if [[ -n "${TASK_ROLE}" && -n "${TASK_INDEX}" ]]; then
        if [[ "${TASK_ROLE}" == "master" ]]; then
            NODE_RANK=${TASK_INDEX}
        else
            NODE_RANK=$((TASK_INDEX + 1))
        fi
    else
        NODE_RANK=${RANK:-${GROUP_RANK:-${SLURM_PROCID:-0}}}
    fi
fi

resolve_master_addr_ipv4() {
    local addr=$1
    "${PYTHON_BIN}" - "$addr" <<'PY'
import re
import socket
import sys

addr = sys.argv[1]
match = re.match(r"^(\d+)-(\d+)-(\d+)-(\d+)(?:\.|$)", addr)
if match:
    print(".".join(match.groups()))
    raise SystemExit(0)

try:
    infos = socket.getaddrinfo(addr, None, socket.AF_INET, socket.SOCK_STREAM)
except socket.gaierror:
    print(addr)
    raise SystemExit(0)

seen = set()
for info in infos:
    ip = info[4][0]
    if ip not in seen:
        print(ip)
        break
    seen.add(ip)
else:
    print(addr)
PY
}

MASTER_ADDR=$(resolve_master_addr_ipv4 "${MASTER_ADDR}")
export MASTER_ADDR MASTER_PORT
export TORCH_MASTER_ADDR="${MASTER_ADDR}"
export TORCH_MASTER_PORT="${MASTER_PORT}"

prepare_thinker_only_hf_path() {
    local src_path=$1
    local mirror_root=$2
    local python_bin=$3

    local config_path="${src_path}/config.json"
    if [[ ! -f "${config_path}" ]]; then
        echo "${src_path}"
        return 0
    fi

    local mirror_path
    mirror_path=$("${python_bin}" - "${src_path}" "${mirror_root}" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
mirror_root = Path(sys.argv[2]).resolve()
config_path = src / "config.json"

with config_path.open() as f:
    config = json.load(f)

if not config.get("enable_audio_output", False):
    print(str(src))
    raise SystemExit(0)

mirror = mirror_root / f"{src.name}-thinker-only"
mirror.mkdir(parents=True, exist_ok=True)

for child in src.iterdir():
    dst = mirror / child.name
    if child.name == "config.json":
        continue
    if dst.exists() or dst.is_symlink():
        continue
    try:
        os.symlink(child, dst, target_is_directory=child.is_dir())
    except FileExistsError:
        pass

patched = dict(config)
patched["enable_audio_output"] = False
patched["talker_config"] = None
patched["code2wav_config"] = None

tmp = mirror / f"config.json.tmp.{os.getpid()}"
with tmp.open("w") as f:
    json.dump(patched, f, indent=2, ensure_ascii=False)
    f.write("\n")
tmp.replace(mirror / "config.json")

print(str(mirror))
PY
)

    echo "${mirror_path}"
}

mkdir -p "${THINKER_ONLY_MIRROR_DIR}"
EFFECTIVE_HF_MODEL_PATH=$(prepare_thinker_only_hf_path "${HF_MODEL_PATH}" "${THINKER_ONLY_MIRROR_DIR}" "${PYTHON_BIN}")

CMD=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nproc_per_node="${NUM_GPUS}"
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    "${ROOT_DIR}/scripts/training/run_recipe.py"
    --mode finetune
    --recipe "${RECIPE}"
    --step_func "${STEP_FUNC}"
    --hf_path "${EFFECTIVE_HF_MODEL_PATH}"
    checkpoint.pretrained_checkpoint="${EFFECTIVE_HF_MODEL_PATH}"
    checkpoint.save="${RESULTS_DIR}/${RUN_NAME}"
    checkpoint.save_interval="${SAVE_INTERVAL}"
    model.seq_length="${SEQ_LENGTH}"
    model.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE}"
    model.pipeline_model_parallel_size="${PIPELINE_PARALLEL_SIZE}"
    model.context_parallel_size="${CONTEXT_PARALLEL_SIZE}"
    model.expert_model_parallel_size="${EXPERT_MODEL_PARALLEL_SIZE}"
    model.expert_tensor_parallel_size="${EXPERT_TENSOR_PARALLEL_SIZE}"
    model.sequence_parallel="${SEQUENCE_PARALLEL}"
    model.freeze_language_model="${FREEZE_LANGUAGE_MODEL}"
    model.freeze_vision_model="${FREEZE_VISION_MODEL}"
    model.freeze_audio_model="${FREEZE_AUDIO_MODEL}"
    model.vit_gradient_checkpointing="${VIT_GRADIENT_CHECKPOINTING}"
    model.multimodal_attn_impl="${MULTIMODAL_ATTN_IMPL}"
    model.attention_backend="${ATTENTION_BACKEND}"
    train.train_iters="${TRAIN_ITERS}"
    train.global_batch_size="${GLOBAL_BATCH_SIZE}"
    train.micro_batch_size="${MICRO_BATCH_SIZE}"
    validation.eval_iters="${EVAL_ITERS}"
    validation.eval_interval="${EVAL_INTERVAL}"
    logger.log_interval="${LOG_INTERVAL}"
    logger.log_throughput=True
    logger.log_throughput_to_tensorboard=True
    logger.tensorboard_dir="${TENSORBOARD_DIR}"
    logger.log_timers_to_tensorboard=False
    logger.wandb_exp_name="${RUN_NAME}"
    dataset.seq_length="${SEQ_LENGTH}"
    dataset.hf_processor_path="${EFFECTIVE_HF_MODEL_PATH}"
    dataset.train_data_path="${TRAIN_JSONL}"
    dataset.valid_data_path="${VALID_JSONL}"
    dataset.test_data_path="${TEST_JSONL}"
    dataset.num_workers="${DATASET_NUM_WORKERS}"
    dataset.persistent_workers="${DATASET_PERSISTENT_WORKERS}"
    dataset.pack_sequences_in_batch=False
)

if [[ -n "${RECOMPUTE_GRANULARITY}" ]]; then
    CMD+=(model.recompute_granularity="${RECOMPUTE_GRANULARITY}")
fi
if [[ -n "${RECOMPUTE_METHOD}" ]]; then
    CMD+=(model.recompute_method="${RECOMPUTE_METHOD}")
fi
if [[ -n "${RECOMPUTE_NUM_LAYERS}" ]]; then
    CMD+=(model.recompute_num_layers="${RECOMPUTE_NUM_LAYERS}")
fi
if [[ -n "${RECOMPUTE_MODULES}" ]]; then
    CMD+=(model.recompute_modules="${RECOMPUTE_MODULES}")
fi
if [[ -n "${OPTIMIZER_CPU_OFFLOAD}" ]]; then
    CMD+=(optimizer.optimizer_cpu_offload="${OPTIMIZER_CPU_OFFLOAD}")
fi
if [[ -n "${OPTIMIZER_OFFLOAD_FRACTION}" ]]; then
    CMD+=(optimizer.optimizer_offload_fraction="${OPTIMIZER_OFFLOAD_FRACTION}")
fi
if [[ -n "${USE_PRECISION_AWARE_OPTIMIZER}" ]]; then
    CMD+=(optimizer.use_precision_aware_optimizer="${USE_PRECISION_AWARE_OPTIMIZER}")
fi
if [[ -n "${OVERLAP_GRAD_REDUCE}" ]]; then
    CMD+=(ddp.overlap_grad_reduce="${OVERLAP_GRAD_REDUCE}")
fi
if [[ -n "${OVERLAP_PARAM_GATHER}" ]]; then
    CMD+=(ddp.overlap_param_gather="${OVERLAP_PARAM_GATHER}")
fi
if [[ -n "${ALIGN_PARAM_GATHER}" ]]; then
    CMD+=(ddp.align_param_gather="${ALIGN_PARAM_GATHER}")
fi
if [[ -n "${USE_PYTORCH_PROFILER}" ]]; then
    CMD+=(profiling.use_pytorch_profiler="${USE_PYTORCH_PROFILER}")
fi
if [[ -n "${USE_NSYS_PROFILER}" ]]; then
    CMD+=(profiling.use_nsys_profiler="${USE_NSYS_PROFILER}")
fi
if [[ -n "${PROFILE_STEP_START}" ]]; then
    CMD+=(profiling.profile_step_start="${PROFILE_STEP_START}")
fi
if [[ -n "${PROFILE_STEP_END}" ]]; then
    CMD+=(profiling.profile_step_end="${PROFILE_STEP_END}")
fi
if [[ -n "${PROFILE_RANKS}" ]]; then
    CMD+=(profiling.profile_ranks="${PROFILE_RANKS}")
fi
if [[ -n "${PYTORCH_PROFILER_COLLECT_SHAPES}" ]]; then
    CMD+=(profiling.pytorch_profiler_collect_shapes="${PYTORCH_PROFILER_COLLECT_SHAPES}")
fi
if [[ -n "${PYTORCH_PROFILER_COLLECT_CALLSTACK}" ]]; then
    CMD+=(profiling.pytorch_profiler_collect_callstack="${PYTORCH_PROFILER_COLLECT_CALLSTACK}")
fi
if [[ -n "${PYTORCH_PROFILER_COLLECT_CHAKRA}" ]]; then
    CMD+=(profiling.pytorch_profiler_collect_chakra="${PYTORCH_PROFILER_COLLECT_CHAKRA}")
fi
if [[ -n "${RECORD_MEMORY_HISTORY}" ]]; then
    CMD+=(profiling.record_memory_history="${RECORD_MEMORY_HISTORY}")
fi
if [[ -n "${NVTX_RANGES}" ]]; then
    CMD+=(profiling.nvtx_ranges="${NVTX_RANGES}")
fi

echo "[info] ROOT_DIR=${ROOT_DIR}"
echo "[info] WORKSPACE=${WORKSPACE}"
echo "[info] HF_MODEL_PATH=${HF_MODEL_PATH}"
echo "[info] EFFECTIVE_HF_MODEL_PATH=${EFFECTIVE_HF_MODEL_PATH}"
echo "[info] TRAIN_JSONL=${TRAIN_JSONL}"
echo "[info] VALID_JSONL=${VALID_JSONL}"
echo "[info] TEST_JSONL=${TEST_JSONL}"
echo "[info] DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS}"
echo "[info] DATASET_PERSISTENT_WORKERS=${DATASET_PERSISTENT_WORKERS}"
echo "[info] NUM_GPUS=${NUM_GPUS}"
echo "[info] NNODES=${NNODES}"
echo "[info] TASK_ROLE=${TASK_ROLE:-unset}"
echo "[info] TASK_INDEX=${TASK_INDEX:-unset}"
echo "[info] NODE_RANK=${NODE_RANK}"
echo "[info] MASTER_ADDR=${MASTER_ADDR}"
echo "[info] MASTER_PORT=${MASTER_PORT}"
echo "[info] RUN_NAME=${RUN_NAME}"
echo "[info] TENSORBOARD_DIR=${TENSORBOARD_DIR}"
echo "[info] FREEZE_LANGUAGE_MODEL=${FREEZE_LANGUAGE_MODEL}"
echo "[info] FREEZE_VISION_MODEL=${FREEZE_VISION_MODEL}"
echo "[info] FREEZE_AUDIO_MODEL=${FREEZE_AUDIO_MODEL}"
echo "[info] VIT_GRADIENT_CHECKPOINTING=${VIT_GRADIENT_CHECKPOINTING}"
echo "[info] MULTIMODAL_ATTN_IMPL=${MULTIMODAL_ATTN_IMPL}"
echo "[info] ATTENTION_BACKEND=${ATTENTION_BACKEND}"
echo "[info] RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY}"
echo "[info] RECOMPUTE_METHOD=${RECOMPUTE_METHOD}"
echo "[info] RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS}"
echo "[info] RECOMPUTE_MODULES=${RECOMPUTE_MODULES}"
echo "[info] OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD}"
echo "[info] OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION}"
echo "[info] USE_PRECISION_AWARE_OPTIMIZER=${USE_PRECISION_AWARE_OPTIMIZER}"
echo "[info] OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE}"
echo "[info] OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER}"
echo "[info] ALIGN_PARAM_GATHER=${ALIGN_PARAM_GATHER}"
echo "[info] USE_PYTORCH_PROFILER=${USE_PYTORCH_PROFILER}"
echo "[info] USE_NSYS_PROFILER=${USE_NSYS_PROFILER}"
echo "[info] PROFILE_STEP_START=${PROFILE_STEP_START}"
echo "[info] PROFILE_STEP_END=${PROFILE_STEP_END}"
echo "[info] PROFILE_RANKS=${PROFILE_RANKS}"
echo "[info] PYTORCH_PROFILER_COLLECT_SHAPES=${PYTORCH_PROFILER_COLLECT_SHAPES}"
echo "[info] PYTORCH_PROFILER_COLLECT_CALLSTACK=${PYTORCH_PROFILER_COLLECT_CALLSTACK}"
echo "[info] PYTORCH_PROFILER_COLLECT_CHAKRA=${PYTORCH_PROFILER_COLLECT_CHAKRA}"
echo "[info] RECORD_MEMORY_HISTORY=${RECORD_MEMORY_HISTORY}"
echo "[info] NVTX_RANGES=${NVTX_RANGES}"
echo "[info] IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM}"
echo "[info] VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM}"
echo "[info] MAX_PIXELS=${MAX_PIXELS}"
echo "[info] VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS}"
echo "[info] FPS_MAX_FRAMES=${FPS_MAX_FRAMES}"
echo "[info] PYTHON_BIN=${PYTHON_BIN}"
echo "[info] PYTHONPATH=${PYTHONPATH}"
echo "[info] PYTHONNOUSERSITE=${PYTHONNOUSERSITE}"
echo "[info] LOG_PATH=${LOG_PATH}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] %q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

cd "${ROOT_DIR}"
SECONDS=0
set -o pipefail
"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
WALL_TIME=${SECONDS}

if [[ "${ENABLE_PERF_SUMMARY}" != "1" ]]; then
    exit 0
fi

python - "${LOG_PATH}" "${GLOBAL_BATCH_SIZE}" "${SEQ_LENGTH}" "${WALL_TIME}" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
global_batch = int(sys.argv[2])
seq_length = int(sys.argv[3])
wall_time = int(sys.argv[4])

elapsed = []
throughput = []
for line in log_path.read_text().splitlines():
    if m := re.search(r"elapsed time per iteration \(ms\):\s+([\d.]+)", line):
        elapsed.append(float(m.group(1)))
    if m := re.search(r"throughput per GPU \(TFLOP/s/GPU\):\s+([\d.]+)", line):
        throughput.append(float(m.group(1)))

print("[perf] wall_time_s=", wall_time, sep="")
if elapsed:
    last_ms = elapsed[-1]
    avg_ms = sum(elapsed) / len(elapsed)
    nominal_tokens_per_sec = global_batch * seq_length / (avg_ms / 1000.0)
    print(f"[perf] last_iter_ms={last_ms:.2f}")
    print(f"[perf] avg_iter_ms={avg_ms:.2f}")
    print(f"[perf] nominal_tokens_per_sec={nominal_tokens_per_sec:.2f}")
else:
    print("[perf] no iteration timing found in log")

if throughput:
    print(f"[perf] last_tflops_per_gpu={throughput[-1]:.2f}")
else:
    print("[perf] no throughput line found in log")
PY
