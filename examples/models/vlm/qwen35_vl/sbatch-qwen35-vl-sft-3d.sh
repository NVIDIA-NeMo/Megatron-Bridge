#!/bin/bash
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

# =============================================================================
# Qwen3.5-VL 35B-A3B SFT with 3D Parallelism (TP/PP/EP, no FSDP)
#
# This script launches Qwen3.5-VL SFT training with standard 3D parallelism
# (no Megatron FSDP). Uses mcore torch_dist checkpoint format.
#
# Prerequisites:
#   1. Convert HF checkpoint to mcore torch_dist format:
#        bash download-convert-qwen35-vl.sh 35B-A3B
#   2. Set WORKSPACE, CONTAINER_IMAGE, and WANDB_API_KEY below.
#
# Usage:
#   sbatch sbatch-qwen35-vl-sft-3d.sh
# =============================================================================

#SBATCH -A <your_account>
#SBATCH -p <your_partition>
#SBATCH -J megatron-bridge.qwen35vl-sft-3d
#SBATCH -t 00:20:00
#SBATCH -N8
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=<your_log_dir>/qwen35vl-sft_%j.out
#SBATCH --error=<your_log_dir>/qwen35vl-sft_%j.err

# ---- Model (hardcoded: Qwen3.5-35B-A3B) ------------------------------------
MODEL_SIZE="35B-A3B"
HF_MODEL_NAME="Qwen3.5-35B-A3B"
RECIPE="qwen35_vl_35b_a3b_sft_config"

# ---- Paths ------------------------------------------------------------------
WORKSPACE="/path/to/your/workspace"
BRIDGE_DIR="${WORKSPACE}/Megatron-Bridge"

# Pretrained checkpoint: FSDP requires fsdp_dtensor format (converted offline).
# Non-FSDP uses standard mcore torch_dist format.
# To convert: bash download-convert-qwen35-vl.sh 35B-A3B  (auto-skips done steps)
#PRETRAINED_CHECKPOINT=${WORKSPACE}/Qwen-Models/mcore/${HF_MODEL_NAME}-fsdp-dtensor
PRETRAINED_CHECKPOINT=${WORKSPACE}/Qwen-Models/mcore/${HF_MODEL_NAME}
HF_MODEL_DIR=${WORKSPACE}/Qwen-Models/hf/${HF_MODEL_NAME}
# HF dataset cache dir (auto-downloaded from HuggingFace Hub on first run)
DATASET_DIR="${WORKSPACE}/dataset"
RESULTS_DIR=${WORKSPACE}/Qwen35-VL/results/${RECIPE}_sft

# ---- Custom TransformerEngine ------------------------------------------------
# USE_CUSTOM_TE=1 : use offline-built TE at TE_DIR (prepend to PYTHONPATH)
# USE_CUSTOM_TE=0 : use container's default TE
# Offline build:  srun -N1 --ntasks-per-node=1 --container-image=$CONTAINER_IMAGE \
#                   --container-mounts=$WORKSPACE:$WORKSPACE \
#                   bash -c "cd $TE_DIR && pip install --no-deps --no-build-isolation -e ."
USE_CUSTOM_TE=0
TE_DIR="${WORKSPACE}/TransformerEngine"

# ---- Container --------------------------------------------------------------
CONTAINER_IMAGE="/path/to/your/container.sqsh"

CONTAINER_MOUNTS="${WORKSPACE}:${WORKSPACE}"

# ---- Training config --------------------------------------------------------
DATASET_NAME=cord_v2
SEQ_LENGTH=4096
GLOBAL_BATCH_SIZE=256
MICRO_BATCH_SIZE=4
PROFILE=0
TRAIN_ITERS=500
EVAL_ITERS=10
LOG_INTERVAL=1

# ---- Parallelism -----------------------------------------------------------
TP=1; PP=1; CP=1; EP=8
#TP=1; PP=1; CP=1; EP=1
SP=$( [ "$TP" -gt 1 ] && echo "true" || echo "false" )

# ---- Recomputation ---------------------------------------------------------
RECOMPUTE_GRANULARITY="none" # selective to enable recomputation
RECOMPUTE_MODULES="layernorm,moe_act"

# ---- CUDA Graph ------------------------------------------------------------
# CG_IMPL selects the implementation — change this one variable to switch:
#   "transformer_engine" = use it for fine-grained cuda graph
#   "local"              = use it for full iteration graph
#   "none"               = baseline (no CUDA graph, for throughput comparison)
#
# CG_SCOPE controls which parts of each layer are captured:
#   attn           - attention sub-layer (fixed shapes, safest)
#   moe_router     - MoE router + shared expert (NOT compatible with moe_router_force_load_balancing)
#   moe_preprocess - MoE preprocessing/dispatch (requires moe_router)
#   moe            - full MoE layer (requires drop-and-pad / moe_pad_expert_input_to_capacity)
#   mlp            - dense MLP (not applicable for pure MoE model)
#   full_iteration - captures entire training iteration (local impl only, conflicts with selective recompute)
#   <empty>        - captures the whole TransformerLayer (most aggressive, may hit dynamic shape issues)
#
# This recipe has moe_router_force_load_balancing=False, so moe_router scope is safe.
CG_IMPL="none"
CG_SCOPE="attn,moe_router,moe_preprocess"
CG_WARMUP=3

# ---- MoE dispatcher --------------------------------------------------------
# "flex" + "hybridep": uses DeepEP HybridEP (requires NVL72 or compatible topology)
# "alltoall": standard MoE All-to-All (works on any topology, well-tested)
# "allgather": standard MoE All-Gather (simpler, less scalable)
MOE_DISPATCHER="alltoall"
MOE_DISPATCHER_BACKEND=""

# ---- Communication Overlap --------------------------------------------------
# DDP overlap (requires use_distributed_optimizer=True, DP>1)
DDP_OVERLAP_GRAD_REDUCE="true"       # overlap grad reduce-scatter (RS) with backward
DDP_OVERLAP_PARAM_GATHER="true"      # overlap param all-gather (AG) with forward
DDP_OVERLAP_PARAM_GATHER_OPT="false" # overlap param gather with optimizer step (requires PP>1, VP>1)
DDP_ALIGN_PARAM_GATHER="false"       # all PP stages issue param all-gather simultaneously (requires PP>1, VP>1, DP>1)
DDP_BUCKET_SIZE=$((128 * 1024 * 1024))

# Megatron FSDP (enables AG↔RS overlap via separate streams + pipeline scheduling)
# To toggle FSDP:
#   ON:  set both to "true"; use fsdp-dtensor PRETRAINED_CHECKPOINT above;
#        keep CLI overrides: checkpoint.ckpt_format=fsdp_dtensor
#   OFF: set both to "false"; switch PRETRAINED_CHECKPOINT to mcore torch_dist;
#        remove checkpoint.ckpt_format override
DDP_USE_MEGATRON_FSDP="false"
DDP_FSDP_DOUBLE_BUFFER="false"
#DDP_USE_MEGATRON_FSDP="true"
#DDP_FSDP_DOUBLE_BUFFER="true"

# AG/RS communication enhancements
# NOTE: NCCL UB not effective for Qwen3.5-VL + FSDP — MoE+VLM architecture produces
#   heterogeneous FSDP units (vision layers, MoE expert buckets, dense buckets, embeddings)
#   with different sizes; FixedPoolAllocator only pools the largest uniform group,
#   all other units fallback to dynamic alloc where UB is unsupported.
DDP_NCCL_UB="true"                  # disabled: MoE+VLM FSDP units all fallback, UB causes hang
DDP_PAD_BUCKETS="false"              # pad bucket size to 2^16 for high NCCL bus bandwidth at large DP
DDP_RS_FP32_ACCUM="false"            # reduce-scatter with FP32 local accumulation (low-prec wire + FP32 accum)

# TP overlap (requires TP>=2, sequence_parallel=True, Transformer Engine)
TP_COMM_OVERLAP="false"              # not applicable when TP=1

# PP overlap (requires PP>1)
PP_OVERLAP_P2P_COMM="false"          # overlap P2P send/recv with compute (requires PP>1)
PP_BATCH_P2P_COMM="true"             # use batch_isend_irecv (requires PP>1, mutually exclusive with overlap_p2p)
PP_DEFER_EMBEDDING_WGRAD="false"     # defer embedding wgrad to pipeline flush (requires PP>1, grad_accum_fusion)

# MoE EP overlap (requires EP>1, flex/alltoall, torch>=2.6, selective recompute)
# NOTE: EP A2A overlap (combined_1f1b) requires GPTModel only, NOT supported for VLM
# (Qwen3VLModel wraps GPTModel as language_model, get_attr_wrapped_model can't traverse it)
MOE_EP_OVERLAP="false"               # overlap EP all-to-all with other micro-batches
MOE_DELAY_WGRAD="false"              # delay wgrad compute for better overlap
MOE_SHARED_EXPERT_OVERLAP="false"    # overlap shared expert with EP comm (requires alltoall dispatcher). Enable it on B200/300, disable it for GB200/300

# ---- Performance / Fusions --------------------------------------------------
GRAD_REDUCE_IN_FP32="true"           # gradient reduce in fp32 precision
GRAD_ACCUM_FUSION="true"             # fused gradient accumulation (needs fused_weight_gradient_mlp_cuda)
CROSS_ENTROPY_FUSION="true"          # fused cross entropy loss
CROSS_ENTROPY_FUSION_IMPL="native"   # "native" (MCore) or "te" (Transformer Engine)
MOE_GROUPED_GEMM="true"              # grouped GEMM for MoE experts
MOE_PERMUTE_FUSION="true"            # fused MoE token permutation (needs TE>=2.1)
MOE_ROUTER_FUSION="true"            # fused TopK routing + aux-loss (needs TE>=2.7)

# ---- WandB naming (auto-generated from config above) -----------------------
CG_TAG=$( [ "$CG_IMPL" != "none" ] && echo "_cg-${CG_IMPL}" || echo "" )
RECOMP_TAG=$( [ "$RECOMPUTE_GRANULARITY" = "selective" ] && echo "_recomp-${RECOMPUTE_MODULES//,/-}" || echo "" )
FSDP_TAG=$( [ "$DDP_USE_MEGATRON_FSDP" = "true" ] && echo "_fsdp" || echo "" )
WANDB_PROJECT="qwen35_vl_35b_a3b_sft"
WANDB_EXP_NAME="qwen35vl_${MODEL_SIZE}_mbs${MICRO_BATCH_SIZE}_tp${TP}cp${CP}pp${PP}ep${EP}${FSDP_TAG}${RECOMP_TAG}${CG_TAG}"

# ---- Environment (exported to srun, inherited by container) -----------------
export CUDA_HOME=/usr/local/cuda
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=1
export HTTPX_LOG_LEVEL=WARNING
export PYTHONWARNINGS="ignore::FutureWarning:torch.cuda,ignore::UserWarning:modelopt.torch"

TE_PYTHONPATH=""
if [ "$USE_CUSTOM_TE" = "1" ]; then
    TE_PYTHONPATH="${TE_DIR}:"
fi
export PYTHONPATH="${TE_PYTHONPATH}${BRIDGE_DIR}/3rdparty/Megatron-LM:${BRIDGE_DIR}/src:${PYTHONPATH:-}"

# export WANDB_API_KEY="<your_wandb_api_key>"
export WANDB_MODE="online"

export HF_HOME="${DATASET_DIR}"

# ---- Build override blocks from config variables above ----------------------
PARALLEL_OVERRIDES="\
    model.tensor_model_parallel_size=$TP \
    model.pipeline_model_parallel_size=$PP \
    model.context_parallel_size=$CP \
    model.expert_model_parallel_size=$EP \
    model.sequence_parallel=$SP"

MEMORY_OVERRIDES=""
if [ "$RECOMPUTE_GRANULARITY" != "none" ]; then
    MEMORY_OVERRIDES="\
        model.recompute_granularity=$RECOMPUTE_GRANULARITY \
        model.recompute_modules=[$RECOMPUTE_MODULES]"
fi

CUDAGRAPH_OVERRIDES=""
if [ "$CG_IMPL" != "none" ]; then
    CUDAGRAPH_OVERRIDES="\
        model.cuda_graph_impl=$CG_IMPL \
        model.cuda_graph_scope=[$CG_SCOPE] \
        model.cuda_graph_warmup_steps=$CG_WARMUP \
        model.use_te_rng_tracker=true"
fi

MOE_OVERRIDES="\
    model.moe_token_dispatcher_type=$MOE_DISPATCHER"
if [ -n "$MOE_DISPATCHER_BACKEND" ]; then
    MOE_OVERRIDES="$MOE_OVERRIDES model.moe_flex_dispatcher_backend=$MOE_DISPATCHER_BACKEND"
fi

OVERLAP_OVERRIDES="\
    ddp.use_megatron_fsdp=$DDP_USE_MEGATRON_FSDP \
    ddp.fsdp_double_buffer=$DDP_FSDP_DOUBLE_BUFFER \
    ddp.overlap_grad_reduce=$DDP_OVERLAP_GRAD_REDUCE \
    ddp.overlap_param_gather=$DDP_OVERLAP_PARAM_GATHER \
    optimizer.overlap_param_gather_with_optimizer_step=$DDP_OVERLAP_PARAM_GATHER_OPT \
    ddp.align_param_gather=$DDP_ALIGN_PARAM_GATHER \
    ddp.bucket_size=$DDP_BUCKET_SIZE \
    ddp.nccl_ub=$DDP_NCCL_UB \
    ddp.pad_buckets_for_high_nccl_busbw=$DDP_PAD_BUCKETS \
    ddp.reduce_scatter_with_fp32_accumulation=$DDP_RS_FP32_ACCUM \
    model.tp_comm_overlap=$TP_COMM_OVERLAP \
    model.overlap_p2p_comm=$PP_OVERLAP_P2P_COMM \
    model.batch_p2p_comm=$PP_BATCH_P2P_COMM \
    model.defer_embedding_wgrad_compute=$PP_DEFER_EMBEDDING_WGRAD \
    model.overlap_moe_expert_parallel_comm=$MOE_EP_OVERLAP \
    model.delay_wgrad_compute=$MOE_DELAY_WGRAD \
    ddp.delay_wgrad_compute=$MOE_DELAY_WGRAD \
    model.moe_shared_expert_overlap=$MOE_SHARED_EXPERT_OVERLAP"

PERF_OVERRIDES="\
    ddp.grad_reduce_in_fp32=$GRAD_REDUCE_IN_FP32 \
    model.gradient_accumulation_fusion=$GRAD_ACCUM_FUSION \
    model.cross_entropy_loss_fusion=$CROSS_ENTROPY_FUSION \
    model.cross_entropy_fusion_impl=$CROSS_ENTROPY_FUSION_IMPL \
    model.moe_grouped_gemm=$MOE_GROUPED_GEMM \
    model.moe_permute_fusion=$MOE_PERMUTE_FUSION \
    model.moe_router_fusion=$MOE_ROUTER_FUSION"

# ---- Profiling overrides (only when PROFILE=1) ------------------------------
# Pure observation — does not alter CUDA Graph, overlap, or any other training
# settings. With CUDA Graph on, trace shows cudaGraphLaunch instead of individual
# kernels; for kernel detail use: nsys profile --cuda-graph-trace=node
# View trace: download .json.gz → https://ui.perfetto.dev/
PROFILING_OVERRIDES=""
if [ "$PROFILE" = "1" ]; then
    PROFILE_STEP_START=${PROFILE_STEP_START:-25}
    PROFILE_STEP_END=${PROFILE_STEP_END:-26}
    PROFILE_RANKS=${PROFILE_RANKS:-"[0]"}
    TB_DIR="${RESULTS_DIR}/tensorboard"

    PROFILING_OVERRIDES="\
        profiling.use_pytorch_profiler=true \
        profiling.profile_step_start=$PROFILE_STEP_START \
        profiling.profile_step_end=$PROFILE_STEP_END \
        profiling.pytorch_profiler_collect_shapes=true \
        profiling.pytorch_profiler_collect_callstack=true \
        profiling.profile_ranks=$PROFILE_RANKS \
        logger.tensorboard_dir=$TB_DIR"

    export TORCH_SHOW_CPP_STACKTRACES=1

    echo ""
    echo ">>> PROFILING MODE ENABLED <<<"
    echo "  Profile steps: $PROFILE_STEP_START ~ $PROFILE_STEP_END"
    echo "  Profile ranks: $PROFILE_RANKS"
    echo "  Chrome Trace:  ${RESULTS_DIR}/torch_profile/rank-*.json.gz"
    echo "  TensorBoard:   $TB_DIR"
    echo "  View trace:    https://ui.perfetto.dev/"
    echo ""
fi

# ---- Build CLI overrides ----------------------------------------------------
CLI_OVERRIDES="\
    checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    model.seq_length=$SEQ_LENGTH \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    train.eval_iters=$EVAL_ITERS \
    train.manual_gc_interval=20 \
    checkpoint.save_interval=999999 \
    checkpoint.fully_parallel_load=true \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=${WANDB_EXP_NAME} \
    dataset.maker_name=make_${DATASET_NAME}_dataset \
    dataset.seq_length=$SEQ_LENGTH \
    $PARALLEL_OVERRIDES \
    $MEMORY_OVERRIDES \
    $CUDAGRAPH_OVERRIDES \
    $MOE_OVERRIDES \
    $OVERLAP_OVERRIDES \
    $PERF_OVERRIDES \
    $PROFILING_OVERRIDES"

# ---- Launch -----------------------------------------------------------------
srun \
    --mpi=pmix \
    -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --container-workdir=${BRIDGE_DIR} \
    bash -c "\
        cd ${BRIDGE_DIR} && \
        python3 -c 'import os; exec(\"try:\\n import wandb; wandb.login()\\nexcept Exception:\\n pass\") if os.environ.get(\"SLURM_LOCALID\",\"0\")==\"0\" else None' && \
        python3 scripts/training/run_recipe.py \
            --recipe $RECIPE \
            --step_func vlm_step \
            --hf_path $HF_MODEL_DIR \
            $CLI_OVERRIDES
    "
