#!/bin/bash
# Run a small DeepSeek-V4 toy training matrix on HSG.
# Override CASES to run a subset, for example:
#   CASES="adam" sbatch examples/models/deepseek_v4/toy_training_hsg.sbatch

#SBATCH --job-name=coreai_dlalgo_llm-mbridge:dsv4-train
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/chcui/logs/dsv4_toy_train_%j.log
#SBATCH --exclusive

set -euo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/users/weijiac/sqsh/nemo_26.04.sqsh}
WORKDIR=${WORKDIR:-/lustre/fsw/portfolios/coreai/users/chcui/repos/Megatron-Bridge-dsv4-train}
MCORE_DIR=${MCORE_DIR:-/lustre/fsw/portfolios/coreai/users/chcui/repos/Megatron-LM-pr4518}
FHT_SITE=${FHT_SITE:-/lustre/fsw/portfolios/coreai/users/chcui/parity/dsv4_cudnn_dsa/fast_hadamard_site}
CUDNN_FE_SITE=${CUDNN_FE_SITE:-/lustre/fsw/portfolios/coreai/users/chcui/parity/dsv4_cudnn_dsa/cudnn_fe_site}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/lustre/fsw/portfolios/coreai/users/chcui/training/dsv4_transformers_5_8_1_nodeps}
HF_CONFIG=${HF_CONFIG:-/lustre/fsw/portfolios/coreai/users/weijiac/models/deepseek-ai/DeepSeek-V4-Flash}
OUTDIR=${OUTDIR:-/lustre/fsw/portfolios/coreai/users/chcui/training/dsv4_toy}
LOGDIR=${LOGDIR:-/lustre/fsw/portfolios/coreai/users/chcui/logs}
PY_DEPS=${PY_DEPS:-$OUTDIR/python_deps}
VENDOR_DEPS=${VENDOR_DEPS:-$WORKDIR/examples/models/deepseek_v4/vendor}
CSA_BACKEND=${CSA_BACKEND:-cudnn_dsa}
MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}
ADAM_MAX_LR=${ADAM_MAX_LR:-1e-4}
ADAM_MIN_LR=${ADAM_MIN_LR:-1e-5}
ADAM_WEIGHT_DECAY=${ADAM_WEIGHT_DECAY:-0.01}
MUON_MAX_LR=${MUON_MAX_LR:-1e-6}
MUON_MIN_LR=${MUON_MIN_LR:-1e-7}
MUON_WEIGHT_DECAY=${MUON_WEIGHT_DECAY:-0.0}
MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
MUON_SCALE_MODE=${MUON_SCALE_MODE:-unit_rms_norm}
MUON_FP32_MATMUL_PREC=${MUON_FP32_MATMUL_PREC:-highest}
MUON_NUM_NS_STEPS=${MUON_NUM_NS_STEPS:-5}
MUON_COEFFICIENT_TYPE=${MUON_COEFFICIENT_TYPE:-quintic}
MUON_OPTIMIZER_NAME=${MUON_OPTIMIZER_NAME:-muon}
MUON_EXTRA_SCALE_FACTOR=${MUON_EXTRA_SCALE_FACTOR:-0.2}
MUON_NESTEROV_ARG=${MUON_NESTEROV_ARG:---no-muon-use-nesterov}
EXTRA_ARGS=${EXTRA_ARGS:-}

CASES=${CASES:-"adam muon adam_mxfp8 muon_mxfp8"}
SEQ_LENGTH=${SEQ_LENGTH:-128}
TRAIN_ITERS=${TRAIN_ITERS:-3}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

mkdir -p "$LOGDIR" "$OUTDIR"

SUMMARY="$OUTDIR/summary_${SLURM_JOB_ID}.txt"
: > "$SUMMARY"

echo "DeepSeek-V4 toy training | job=$SLURM_JOB_ID"
echo "workdir=$WORKDIR"
echo "mcore=$MCORE_DIR"
echo "fht_site=$FHT_SITE"
echo "cudnn_fe_site=$CUDNN_FE_SITE"
echo "transformers_site=$TRANSFORMERS_SITE"
echo "hf_config=$HF_CONFIG"
echo "outdir=$OUTDIR"
echo "cases=$CASES"
echo "csa_backend=$CSA_BACKEND"
echo "moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER_TYPE"
echo "vendor_deps=$VENDOR_DEPS"
echo "seq_length=$SEQ_LENGTH train_iters=$TRAIN_ITERS nproc_per_node=$NPROC_PER_NODE"
echo "adam_lr=$ADAM_MAX_LR:$ADAM_MIN_LR adam_weight_decay=$ADAM_WEIGHT_DECAY"
echo "muon_lr=$MUON_MAX_LR:$MUON_MIN_LR muon_weight_decay=$MUON_WEIGHT_DECAY muon_momentum=$MUON_MOMENTUM"
echo "muon_optimizer_name=$MUON_OPTIMIZER_NAME muon_scale_mode=$MUON_SCALE_MODE muon_fp32_matmul_prec=$MUON_FP32_MATMUL_PREC"
echo "muon_num_ns_steps=$MUON_NUM_NS_STEPS muon_coefficient_type=$MUON_COEFFICIENT_TYPE muon_extra_scale_factor=$MUON_EXTRA_SCALE_FACTOR muon_nesterov_arg=$MUON_NESTEROV_ARG"
echo "extra_args=$EXTRA_ARGS"

case_index=0
overall_status=0
for CASE in $CASES; do
  case_index=$((case_index + 1))
  master_port=$((29540 + case_index))
  case_log="$OUTDIR/${CASE}_${SLURM_JOB_ID}.log"

  echo "START case=$CASE master_port=$master_port" | tee -a "$SUMMARY"

  set +e
  srun --ntasks=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts=/lustre:/lustre \
    --no-container-mount-home \
    bash -lc "
      set -euo pipefail
      export HF_HOME=/lustre/fsw/portfolios/coreai/users/chcui/hf_home
      export HF_HUB_OFFLINE=1
      export UV_CACHE_DIR=/tmp/uv_cache
      export NCCL_DEBUG=WARN
      export TORCH_NCCL_AVOID_RECORD_STREAMS=1
      export NCCL_NVLS_ENABLE=0
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export NCCL_TIMEOUT=1800000
      export CUDA_DEVICE_MAX_CONNECTIONS=1
      cd $WORKDIR
	      rm -f /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.pth \
	            /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.py
	      mkdir -p $PY_DEPS
	      export PYTHONPATH=$VENDOR_DEPS:$PY_DEPS:$TRANSFORMERS_SITE:$FHT_SITE:$CUDNN_FE_SITE:$WORKDIR/src:$MCORE_DIR:\${PYTHONPATH:-}
	      source $WORKDIR/examples/models/deepseek_v4/hsg_runtime.sh
	      ensure_transformers_site $TRANSFORMERS_SITE
	      /opt/venv/bin/python - <<'PY'
import importlib.util

from fast_hadamard_transform import hadamard_transform
from cudnn import DSA
from megatron.core.transformer.experimental_attention_variant.csa import _CudnnDSASparseAttention

fht_spec = importlib.util.find_spec('fast_hadamard_transform')
cudnn_spec = importlib.util.find_spec('cudnn')
assert fht_spec is not None, 'fast_hadamard_transform not importable'
assert cudnn_spec is not None, 'cudnn not importable'
assert hasattr(DSA, 'sparse_attention_backward_wrapper'), 'cudnn.DSA sparse attention backward missing'
assert _CudnnDSASparseAttention is not None
print('FHT origin:', fht_spec.origin)
print('cuDNN origin:', cudnn_spec.origin)
print('cuDNN DSA sparse attention backward wrapper: OK')
PY
	      if [[ $CASE == muon* ]]; then
	        /opt/venv/bin/python - <<'PY'
import importlib.metadata as md

version = md.version('emerging-optimizers')
assert tuple(int(x) for x in version.split('.')[:2]) >= (0, 2), version
import emerging_optimizers
print('emerging-optimizers:', version, emerging_optimizers.__file__)
PY
	      fi
      uv run --no-sync python -m torch.distributed.run \
        --nproc_per_node=$NPROC_PER_NODE \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --master_port=$master_port \
        examples/models/deepseek_v4/toy_training.py \
        --case $CASE \
        --hf-config-path $HF_CONFIG \
        --output-dir $OUTDIR \
        --seq-length $SEQ_LENGTH \
        --train-iters $TRAIN_ITERS \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --expert-parallel-size $EXPERT_PARALLEL_SIZE \
        --csa-backend $CSA_BACKEND \
        --moe-token-dispatcher-type $MOE_TOKEN_DISPATCHER_TYPE \
        --adam-max-lr $ADAM_MAX_LR \
        --adam-min-lr $ADAM_MIN_LR \
        --adam-weight-decay $ADAM_WEIGHT_DECAY \
        --muon-max-lr $MUON_MAX_LR \
        --muon-min-lr $MUON_MIN_LR \
        --muon-weight-decay $MUON_WEIGHT_DECAY \
        --muon-momentum $MUON_MOMENTUM \
        --muon-scale-mode $MUON_SCALE_MODE \
        --muon-fp32-matmul-prec $MUON_FP32_MATMUL_PREC \
        --muon-num-ns-steps $MUON_NUM_NS_STEPS \
        --muon-coefficient-type $MUON_COEFFICIENT_TYPE \
        --muon-optimizer-name $MUON_OPTIMIZER_NAME \
        --muon-extra-scale-factor $MUON_EXTRA_SCALE_FACTOR \
        $MUON_NESTEROV_ARG \
        $EXTRA_ARGS
    " 2>&1 | tee "$case_log"
  status=${PIPESTATUS[0]}
  set -e

  echo "END case=$CASE status=$status log=$case_log" | tee -a "$SUMMARY"
  if [ "$status" -ne 0 ]; then
    overall_status=$status
  fi
done

echo "Summary: $SUMMARY"
exit "$overall_status"
