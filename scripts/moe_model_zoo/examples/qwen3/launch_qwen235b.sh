# Sbatch config
export ACCOUNT=
export PARTITION=
export RUN_NAME=
export RUN_TIME=
export CONTAINER_IMAGE=
export CONTAINER_MOUNTS=

# Cluster config
export CLUSTER=
export CLUSTER_CONF_PATH=

# Env config
export MBRIDGE_PATH=
export MEGATRON_PATH=
export PYTHONPATH=

# WanDB and log config
export WANDB_PROJECT=
export OUTPUT_PATH=
export MBRIDGE_RELEASE_VERSION=
export COMMENT=

# Model config
export MODEL=
export DATASET=
export PRETRAIN=

# HuggingFace config
export HF_TOKEN=

# Training parameters
export PROFILE=0
export MBS=1
export SEQ_LEN=4096
export MOE_GROUPED_GEMM=true

# Interacitve mode
# TP=1 PP=1 CP=1 VPP=null EP=2 GBS=2048 /lustre/fsw/coreai_devtech_all/pingtianl/mbridge/megatron-bridge/scripts/moe_model_zoo/interactive_benchmarking.sh \
#   model.moe_router_force_load_balancing=true model.sequence_parallel=false model.num_layers=1 model.num_moe_experts=4 \
#   model.hidden_size=512 model.moe_router_topk=2

# Launch tests
A2A_OVERLAP=1 TP=2 PP=8 VPP=4 EP=32 NNODES=32 GBS=2048 bash ${MBRIDGE_PATH}/scripts/moe_model_zoo/sbatch_benchmarking.sh model.recompute_granularity=selective model.recompute_modules=moe_act,layernorm model.moe_router_force_load_balancing=true 
