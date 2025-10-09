export ACCOUNT=""
export PARTITION=""
export RUN_NAME=""
export RUN_TIME=""
export CONTAINER_IMAGE=""
export CONTAINER_MOUNTS=""

export OUTPUT_PATH=""
export MODEL=""
export MBRIDGE_PATH=""
export MEGATRON_PATH=""
export PYTHONPATH=""
export WANDB_API_KEY=""
export MBRIDGE_RELEASE_VERSION=""

# Training parameters
export PROFILE=0 # whether to profile the model with nsys profile
export MBS=1
export SEQ_LEN=4096
export MOE_GROUPED_GEMM=true

A2A_OVERLAP=1 TP=2 PP=8 VPP=4 EP=32 NNODES=32 GBS=2048 bash ./sbatch_benchmarking.sh config_container.model.recompute_granularity=selective config_container.model.recompute_modules="moe_act layernorm" config_container.model.moe_router_force_load_balancing=true
