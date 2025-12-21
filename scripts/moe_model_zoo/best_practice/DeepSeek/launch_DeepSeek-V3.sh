# Model config
export MODEL=DeepSeek-V3
export DATASET=Slimpajama

# Sbatch config
export ACCOUNT=
export PARTITION=
export RUN_TIME=
export CONTAINER_IMAGE=
export CONTAINER_MOUNTS=
export RUN_NAME=

# Cluster config
export CLUSTER=
export CLUSTER_CONF_PATH=
export PRETRAIN=1

# Env config
export MBRIDGE_PATH=
export MEGATRON_PATH=
export PYTHONPATH=${MBRIDGE_PATH}/src/:${MEGATRON_PATH}:${PYTHONPATH:-}

# WanDB and log config
export WANDB_PROJECT=
export OUTPUT_PATH=
export MBRIDGE_RELEASE_VERSION=
export COMMENT=


# HuggingFace config
export HF_TOKEN=

# Training parameters
export PROFILE=0
export MBS=1
export SEQ_LEN=4096
export MOE_GROUPED_GEMM=true

# Interacitve mode
SEQ_LEN=1024 PR=fp8 A2A_OVERLAP=1 CP=1 PP=2 VPP=2 TP=1 EP=4 GBS=1024 bash ${MBRIDGE_PATH}/scripts/moe_model_zoo/interactive_benchmarking.sh \
  model.recompute_granularity=selective model.recompute_modules=mla_up_proj,mlp \
  model.pipeline_model_parallel_layout="E(tt|)*3mL" \
  model.sequence_parallel=false \
  model.num_layers=6 \
  model.num_moe_experts=16 \
  model.hidden_size=512 \
  model.moe_layer_freq="'[0]*2+[1]*4'" \
  model.moe_router_force_load_balancing=true

# Launch tests
# H100 config with 1024 GPUs
PR=fp8 A2A_OVERLAP=1 PP=8 VPP=4 TP=2 EP=64 NNODES=128 GBS=8192 bash ${MBRIDGE_PATH}/scripts/moe_model_zoo/sbatch_benchmarking.sh \
  model.recompute_granularity=selective model.recompute_modules=mla_up_proj,mlp \
  model.pipeline_model_parallel_layout="Et*3|(tt|)*29m|L" \
  model.moe_router_force_load_balancing=true

# B200 config with 256 GPUs
PR=mxfp8 A2A_OVERLAP=1 TP=1 PP=8 EP=32 NNODES=32 GBS=2048 bash ${MBRIDGE_PATH}/scripts/moe_model_zoo/sbatch_benchmarking.sh \
  model.recompute_granularity=selective \
  model.recompute_modules=mla_up_proj,mlp \
  model.pipeline_model_parallel_layout="Et*3|(tt|)*29m|L" \
  model.moe_router_force_load_balancing=true
