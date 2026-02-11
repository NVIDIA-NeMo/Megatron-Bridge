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

export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTHONWARNINGS=ignore
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_NODEID
export NVLINK_DOMAIN_SIZE=8
unset CUDA_DEVICE_MAX_CONNECTIONS

# Configuration: Set these variables before running the script
export MEGATRON_BRIDGE_PATH=YOUR_MEGATRON_BRIDGE_PATH # Path to Megatron-Bridge repository
export MEGATRON_LM_PATH=${MEGATRON_BRIDGE_PATH}/3rdparty/Megatron-LM # Path to Megatron-LM repository
export OUTPUT_PATH=${MEGATRON_BRIDGE_PATH}/../qwen3_vl_30b_a3b_output # Path for output logs and checkpoints
export HF_HOME=${MEGATRON_BRIDGE_PATH}/../hf_home # Path to Hugging Face cache
export CONTAINER_IMAGE=YOUR_CONTAINER_IMAGE # Path to .sqsh or docker image url
export WANDB=${WANDB:-1}
export PROFILE=${PROFILE:-0}

TP=${TP:-1}
EP=${EP:-8}
MBS=${MBS:-1}
GBS=${GBS:-32}
COMMENT=${COMMENT:-"alltoall-full-recompute"}

PRETRAIN_ARGS=(
    mixed_precision=null
    model.bf16=true
    model.params_dtype=torch.bfloat16
    model.pipeline_dtype=torch.bfloat16
    model.tensor_model_parallel_size=${TP}
    model.expert_model_parallel_size=${EP}
    model.freeze_language_model=false
    model.freeze_vision_model=false
    model.freeze_vision_projection=false
    model.init_model_with_meta_device=true
    model.seq_length=4096
    model.gradient_accumulation_fusion=false
    model.calculate_per_token_loss=true
    model.moe_token_dispatcher_type=alltoall
    model.recompute_granularity=full
    model.recompute_method=uniform
    model.recompute_num_layers=1
    train.train_iters=200
    train.global_batch_size=${GBS}
    train.micro_batch_size=${MBS}
    train.eval_iters=5
    train.eval_interval=100
    checkpoint.save=${OUTPUT_PATH}/checkpoints
    checkpoint.ckpt_format=fsdp_dtensor
    dist.use_megatron_fsdp=true
    dist.use_torch_fsdp2=false
    logger.log_interval=1
    logger.log_throughput=true
    logger.log_throughput_to_tensorboard=true
    logger.log_l2_norm_grad_to_tensorboard=false
    ddp.grad_reduce_in_fp32=false
    ddp.use_megatron_fsdp=true
    ddp.use_distributed_optimizer=true
    ddp.data_parallel_sharding_strategy=optim_grads_params
)

if [ "${WANDB}" = 1 ]; then
    export WANDB_API_KEY=YOUR_WANDB_API_KEY # Replace with your own Wandb API key
    PRETRAIN_ARGS=(
        "${PRETRAIN_ARGS[@]}"
        logger.wandb_project=YOUR_WANDB_PROJECT # Replace with your own Wandb project name
        logger.wandb_exp_name=Qwen3-VL-30B-A3B-EP${EP}-MBS${MBS}GBS${GBS}-${COMMENT}
    )
fi

# Profiling command
if [ "${PROFILE}" = 1 ]; then
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx,cublas,cudnn \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-graph-trace=node \
        --cuda-memory-usage=true \
        -f true -x true \
        -o ${OUTPUT_PATH}/nsys/Megatron-FSDP-Qwen3-VL-30B-A3B-EP${EP}-MBS${MBS}GBS${GBS}-${COMMENT}"
    PRETRAIN_ARGS=(
        "${PRETRAIN_ARGS[@]}"
        profiling.use_nsys_profiler=true
        profiling.profile_step_start=10
        profiling.profile_step_end=12
        profiling.profile_ranks=[0]
    )
    echo "PROFILE_CMD="
    echo $PROFILE_CMD
else
    PROFILE_CMD=""
fi

TRAINING_CMD="
export PYTHONPATH=${MEGATRON_BRIDGE_PATH}/src:${MEGATRON_LM_PATH}:${PYTHONPATH}
cd ${MEGATRON_BRIDGE_PATH}/examples/models/vlm/qwen_vl;
python finetune_qwen_vl.py \
    --recipe qwen3_vl_30b_a3b_finetune_config \
    ${PRETRAIN_ARGS[@]}"

# SLURM settings
SLURM_LOGS="${OUTPUT_PATH}/slurm_logs"
mkdir -p ${SLURM_LOGS} || {
    echo "Error: Failed to create SLURM logs directory ${SLURM_LOGS}"
    exit 1
}

# Submit SLURM job
# Note: Update SBATCH parameters and container mounts below according to your cluster configuration
set +e
sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=YOUR_JOB_NAME
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=YOUR_NUM_TASKS_PER_NODE
#SBATCH --time=00:15:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --segment=2

srun \
    --mpi=pmix -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=YOUR_CONTAINER_MOUNTS \
    --container-workdir=${MEGATRON_BRIDGE_PATH} \
    bash -x -c "${TRAINING_CMD}" 2>&1 | tee ${SLURM_LOGS}/\${SLURM_JOB_ID}.log

EOF
set -e