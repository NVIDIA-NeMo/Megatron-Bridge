#!/usr/bin/env bash
# CW/Hopper DSv4 smoke launcher for Bridge main + Megatron-LM dev.
# Submit with MODE in:
#   adam_dsaoff, muon_dsaoff, adam_dsaon_idxon, muon_dsaon_idxon,
#   adam_fp8block_dsaoff

#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/weijiac/logs/dsv4_pretrain_cw/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/users/weijiac/logs/dsv4_pretrain_cw/%x_%j.err

set -euo pipefail

WKDIR=${WKDIR:-/lustre/fsw/portfolios/coreai/users/weijiac}
WORKSPACE=${WORKSPACE:-$WKDIR/nemo_workspace}
BRIDGE_DIR=${BRIDGE_DIR:-$WORKSPACE/Megatron-Bridge}
MCORE_DIR=${MCORE_DIR:-$WORKSPACE/Megatron-LM}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-$WKDIR/sqsh/nemo_26.06.rc2.sqsh}
LOGDIR=${LOGDIR:-$WKDIR/logs/dsv4_pretrain_cw}
OUTDIR=${OUTDIR:-$WKDIR/training/dsv4_pretrain_cw}
MODEL_PATH=${MODEL_PATH:-$WKDIR/models/deepseek-ai/DeepSeek-V4-Flash}
DCLM_DATA_DIR=${DCLM_DATA_DIR:-$WKDIR/data/dclm/preprocessed}
DCLM_CACHE=${DCLM_CACHE:-$WKDIR/.cache}

CUDNN_DSA_SITE=${CUDNN_DSA_SITE:-$WKDIR/training/dsv4_cudnn_frontend_pr263_latest_nodeps_site}
CUTLASS_DSL_SITE=${CUTLASS_DSL_SITE:-$WKDIR/training/dsv4_python_deps_cutlass_4_5_2}
CUTLASS_DSL_PYTHON_SITE=${CUTLASS_DSL_PYTHON_SITE:-$CUTLASS_DSL_SITE/nvidia_cutlass_dsl/python_packages}
NVRX_SITE=${NVRX_SITE:-$WKDIR/training/dsv4_nvidia_resiliency_ext_0_6_0}
FHT_SITE=${FHT_SITE:-$WKDIR/training/dsv4_fast_hadamard_transform_site}
EMERGING_SITE=${EMERGING_SITE:-$WKDIR/training/dsv4_emerging_optimizers_0_2_0}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-$WKDIR/training/dsv4_transformers_5_8_1_nodeps}
FLASH_MLA_SITE=${FLASH_MLA_SITE:-$WKDIR/training/flash_mla_nv_dev_b7643bd_site}
MCORE_COMPAT_SITE=${MCORE_COMPAT_SITE:-$WKDIR/training/dsv4_mcore_safe_world_size_overlay}
CW_COMPAT_SITE=${CW_COMPAT_SITE:-$WKDIR/training/dsv4_cw_recompute_compat_site}

MODE=${MODE:-muon_dsaoff}
TRAIN_ITERS=${TRAIN_ITERS:-20}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
TP=${TP:-1}
PP=${PP:-32}
EP=${EP:-8}
CP=${CP:-1}
RUN_TAG=${RUN_TAG:-}
PIPELINE_LAYOUT=${PIPELINE_LAYOUT:-Et*3|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t*2|t|t|t|t|t|t|t|t|t|t|t|t|t|t|t|t|t|t|t|t|t|tmL}
APPLY_ROPE_FUSION=${APPLY_ROPE_FUSION:-true}
USE_FUSED_MHC=${USE_FUSED_MHC:-false}
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-selective}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-null}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-null}
RECOMPUTE_MODULES=${RECOMPUTE_MODULES:-[moe_act,mhc]}
MIXED_PRECISION_OVERRIDES=${MIXED_PRECISION_OVERRIDES:-}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
NNODES=${SLURM_JOB_NUM_NODES:-32}
MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
MASTER_PORT=${MASTER_PORT:-$((29000 + SLURM_JOB_ID % 1000))}

mkdir -p "$LOGDIR" "$OUTDIR"

case "$MODE" in
    adam_dsaoff)
        RECIPE=deepseek_v4_flash_pretrain_config
        DSA_OVERRIDES="model.apply_dsa_kernel_fusion=false model.dsa_indexer_loss_coeff=0.0 model.dsa_indexer_use_sparse_loss=false"
        ;;
    adam_fp8block_dsaoff)
        RECIPE=deepseek_v4_flash_pretrain_config
        DSA_OVERRIDES="model.apply_dsa_kernel_fusion=false model.dsa_indexer_loss_coeff=0.0 model.dsa_indexer_use_sparse_loss=false"
        MIXED_PRECISION_OVERRIDES="${MIXED_PRECISION_OVERRIDES:-mixed_precision=bf16_with_fp8_subchannel_scaling_mixed model.moe_router_padding_for_fp8=true}"
        ;;
    muon_dsaoff)
        RECIPE=deepseek_v4_flash_pretrain_muon_config
        DSA_OVERRIDES="model.apply_dsa_kernel_fusion=false model.dsa_indexer_loss_coeff=0.0 model.dsa_indexer_use_sparse_loss=false"
        ;;
    adam_dsaon_idxon)
        RECIPE=deepseek_v4_flash_pretrain_config
        DSA_OVERRIDES="model.apply_dsa_kernel_fusion=true model.dsa_indexer_loss_coeff=1e-2 model.dsa_indexer_use_sparse_loss=true"
        ;;
    muon_dsaon_idxon)
        RECIPE=deepseek_v4_flash_pretrain_muon_config
        DSA_OVERRIDES="model.apply_dsa_kernel_fusion=true model.dsa_indexer_loss_coeff=1e-2 model.dsa_indexer_use_sparse_loss=true"
        ;;
    *)
        echo "ERROR: unsupported MODE=$MODE"
        exit 2
        ;;
esac

RUN_TAG_SUFFIX=""
if [ -n "$RUN_TAG" ]; then
    RUN_TAG_SUFFIX="-${RUN_TAG}"
fi
RUN_NAME=dsv4-h100-${MODE}${RUN_TAG_SUFFIX}-pp${PP}-ep${EP}-cp${CP}-rope${APPLY_ROPE_FUSION}-2606-${SLURM_JOB_ID:-manual}
CHECKPOINT_DIR=$OUTDIR/$RUN_NAME/checkpoints
WANDB_DIR=$OUTDIR/wandb
mkdir -p "$CHECKPOINT_DIR" "$WANDB_DIR"

DATA_OVERRIDES=""
if [ -d "$DCLM_DATA_DIR" ]; then
    BLEND=""
    for n in $(seq 1 6); do
        i=$(printf "%02d" "$n")
        prefix="$DCLM_DATA_DIR/dclm_01_${i}_text_document"
        if [ -f "${prefix}.bin" ] || [ -f "${prefix}.idx" ]; then
            if [ -n "$BLEND" ]; then
                BLEND="$BLEND,"
            fi
            BLEND="$BLEND$prefix"
        fi
    done
    if [ -n "$BLEND" ]; then
        DATA_OVERRIDES="dataset.blend=[[$BLEND],null] dataset.path_to_cache=$DCLM_CACHE"
    fi
fi

if [ -z "$DATA_OVERRIDES" ]; then
    echo "ERROR: no DCLM data found under $DCLM_DATA_DIR"
    exit 3
fi

PYTHONPATH_PARTS="$MCORE_DIR:$BRIDGE_DIR/src"
if [ -d "$MCORE_COMPAT_SITE" ]; then
    PYTHONPATH_PARTS="$MCORE_COMPAT_SITE:$PYTHONPATH_PARTS"
fi
if [ -d "$CW_COMPAT_SITE" ]; then
    PYTHONPATH_PARTS="$CW_COMPAT_SITE:$PYTHONPATH_PARTS"
fi
for site in "$NVRX_SITE" "$CUDNN_DSA_SITE" "$CUTLASS_DSL_PYTHON_SITE" "$CUTLASS_DSL_SITE" "$FHT_SITE" "$EMERGING_SITE" "$TRANSFORMERS_SITE" "$FLASH_MLA_SITE"; do
    if [ -d "$site" ]; then
        PYTHONPATH_PARTS="$site:$PYTHONPATH_PARTS"
    fi
done

COMMON_OVERRIDES="
train.train_iters=$TRAIN_ITERS
train.global_batch_size=$GLOBAL_BATCH_SIZE
train.micro_batch_size=$MICRO_BATCH_SIZE
dataset.sequence_length=$SEQ_LENGTH
model.seq_length=$SEQ_LENGTH
model.tensor_model_parallel_size=$TP
model.pipeline_model_parallel_size=$PP
model.pipeline_model_parallel_layout='$PIPELINE_LAYOUT'
model.context_parallel_size=$CP
model.expert_model_parallel_size=$EP
model.expert_tensor_parallel_size=1
model.sequence_parallel=false
model.apply_rope_fusion=$APPLY_ROPE_FUSION
model.use_fused_mhc=$USE_FUSED_MHC
model.moe_token_dispatcher_type=alltoall
model.recompute_granularity=$RECOMPUTE_GRANULARITY
model.recompute_method=$RECOMPUTE_METHOD
model.recompute_num_layers=$RECOMPUTE_NUM_LAYERS
model.recompute_modules=$RECOMPUTE_MODULES
validation.eval_interval=$TRAIN_ITERS
validation.eval_iters=2
scheduler.lr_warmup_iters=10
scheduler.lr_decay_iters=$TRAIN_ITERS
checkpoint.load=null
checkpoint.save=$CHECKPOINT_DIR
checkpoint.save_interval=1000
checkpoint.save_optim=false
checkpoint.fully_parallel_save=false
checkpoint.exit_on_missing_checkpoint=false
logger.wandb_project=megatron-bridge-dsv4
logger.wandb_entity=nvidia-nemo-fw-public
logger.wandb_exp_name=$RUN_NAME
logger.wandb_save_dir=$WANDB_DIR
logger.log_interval=1
dist.distributed_timeout_minutes=60
$DSA_OVERRIDES
$DATA_OVERRIDES
$MIXED_PRECISION_OVERRIDES
$EXTRA_OVERRIDES
"
COMMON_OVERRIDES_ONE_LINE=$(printf "%s" "$COMMON_OVERRIDES" | tr "\n" " ")

echo "Run name: $RUN_NAME"
echo "Mode: $MODE"
echo "Recipe: $RECIPE"
echo "Bridge dir: $BRIDGE_DIR"
git -C "$BRIDGE_DIR" status --short --branch || true
git -C "$BRIDGE_DIR" log -1 --oneline --decorate || true
echo "MCore dir: $MCORE_DIR"
git -C "$MCORE_DIR" status --short --branch || true
git -C "$MCORE_DIR" log -1 --oneline --decorate || true
echo "Container: $CONTAINER_IMAGE"
echo "Nodes: $NNODES GPUs/node: $NPROC_PER_NODE"
echo "Parallelism: TP=$TP PP=$PP EP=$EP CP=$CP"
echo "Recompute: granularity=$RECOMPUTE_GRANULARITY method=$RECOMPUTE_METHOD num_layers=$RECOMPUTE_NUM_LAYERS modules=$RECOMPUTE_MODULES"
echo "Mixed precision overrides: ${MIXED_PRECISION_OVERRIDES:-<none>}"
echo "Extra overrides: ${EXTRA_OVERRIDES:-<none>}"
echo "PP layout: $PIPELINE_LAYOUT"
echo "DSA overrides: $DSA_OVERRIDES"
echo "PYTHONPATH parts: $PYTHONPATH_PARTS"
echo "Checkpoint dir: $CHECKPOINT_DIR"

srun --ntasks="$NNODES" --ntasks-per-node=1 --gpus-per-node="$NPROC_PER_NODE" \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="/lustre:/lustre${EXTRA_CONTAINER_MOUNTS:+,$EXTRA_CONTAINER_MOUNTS}" \
    bash -lc "
        set -euo pipefail
        export HF_HOME=/tmp/weijiac_hf_${SLURM_JOB_ID}_\${SLURM_PROCID:-0}
        export TRANSFORMERS_CACHE=\$HF_HOME/transformers
        export HF_HUB_CACHE=\$HF_HOME/hub
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export MEGATRON_CONFIG_LOCK_DIR=\$HF_HOME/locks
        export HF_SNAPSHOT=0000000000000000000000000000000000000000
        export HF_MODEL_CACHE=\$HF_HUB_CACHE/models--deepseek-ai--DeepSeek-V4-Flash
        mkdir -p \$TRANSFORMERS_CACHE \$HF_HUB_CACHE \$MEGATRON_CONFIG_LOCK_DIR \$HF_MODEL_CACHE/refs \$HF_MODEL_CACHE/snapshots/\$HF_SNAPSHOT
        rm -f \$MEGATRON_CONFIG_LOCK_DIR/.megatron_config_lock_* 2>/dev/null || true
        printf '%s' \$HF_SNAPSHOT > \$HF_MODEL_CACHE/refs/main
        for f in config.json generation_config.json tokenizer_config.json tokenizer.json model.safetensors.index.json; do
            if [ -f $MODEL_PATH/\$f ]; then
                cp -f $MODEL_PATH/\$f \$HF_MODEL_CACHE/snapshots/\$HF_SNAPSHOT/\$f
            fi
        done
        export WANDB_API_KEY=\${WANDB_API_KEY:-}
        export NVTE_NORM_FWD_USE_CUDNN=1
        export NVTE_NORM_BWD_USE_CUDNN=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PYTHONPATH=$PYTHONPATH_PARTS:\${PYTHONPATH:-}
        cd $BRIDGE_DIR
        python - <<'PY'
import importlib, importlib.metadata as md
for dist in ['nvidia-cutlass-dsl', 'nvidia-cudnn-frontend']:
    try:
        print(f'{dist}={md.version(dist)}')
    except Exception as exc:
        print(f'{dist}=missing ({type(exc).__name__})')
for mod in ['cutlass', 'cutlass.cute', 'cudnn']:
    try:
        m = importlib.import_module(mod)
        print(f'{mod}: {getattr(m, \"__file__\", None)}')
    except Exception as exc:
        print(f'{mod}: import failed {type(exc).__name__}: {exc}')
PY
        uv run --no-sync python -m torch.distributed.run \
            --nproc_per_node=$NPROC_PER_NODE \
            --nnodes=$NNODES \
            --node_rank=\$SLURM_PROCID \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            scripts/training/run_recipe.py \
            --recipe $RECIPE \
            --dataset llm-pretrain \
            --step_func gpt_step \
            --hf_path $MODEL_PATH \
            $COMMON_OVERRIDES_ONE_LINE
    "
