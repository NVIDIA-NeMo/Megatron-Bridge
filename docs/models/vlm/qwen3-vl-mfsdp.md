# Qwen3-VL Training with Megatron-FSDP

## Step 1: Download Source Code

```bash
# Clone Megatron-Bridge
git clone git@github.com:NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge/
git remote add -f xuwchen git@github.com:xuwchen/Megatron-Bridge.git
git checkout -b qwen3_vl xuwchen/qwen3_vl_mfsdp

# Clone Megatron-LM
cd 3rdparty/
git clone git@github.com:NVIDIA/Megatron-LM.git
cd Megatron-LM/
git remote add -f xuwchen git@github.com:xuwchen/Megatron-LM.git
git checkout -b qwen3_vl_dev xuwchen/qwen3_vl_mfsdp_dev
cd ../../
```

## Step 2: Update the Config File

Update the model config YAML file for your use case with Megatron-FSDP required arguments. Example configuration files are provided for:
- [Qwen3-VL-30B-A3B-Instruct](../../../examples/recipes/qwen_vl/conf/qwen3_vl_30b_a3b_pretrain_mfsdp_override_example.yaml)
- [Qwen3-VL-235B-A22B-Instruct](../../../examples/recipes/qwen_vl/conf/qwen3_vl_235b_a22b_pretrain_mfsdp_override_example.yaml)

## Step 3: Run Training

### Interactive Node Training

Taking Qwen3-VL-30B-A3B as an example, follow these steps to train on an interactive node:

**1. Start the Docker container:**

```bash
# Start docker container
srun -A coreai_dlalgo_mcore -p interactive --time=04:00:00 --gpus-per-node=8 --container-image=nvcr.io/nvidia/nemo:25.11 --container-mounts=/lustre:/lustre -J coreai_dlalgo_mcore:qwen3_vl --pty bash
```

**2. Set up the environment:**

```bash
export MEGATRON_BRIDGE_PATH=<your_megatron_bridge_path>
export MEGATRON_LM_PATH=${MEGATRON_BRIDGE_PATH}/3rdparty/Megatron-LM
export HF_HOME=${MEGATRON_BRIDGE_PATH}/../hf_home
unset CUDA_DEVICE_MAX_CONNECTIONS
```

**3. Launch training:**

```bash
export PYTHONPATH=${MEGATRON_BRIDGE_PATH}/src:${MEGATRON_LM_PATH}:${PYTHONPATH}
cd ${MEGATRON_BRIDGE_PATH}/examples/recipes/qwen_vl
python -m torch.distributed.run --nproc_per_node=8 \
    finetune_qwen_vl.py \
    --recipe qwen3_vl_30b_a3b_finetune_config \
    --config-file conf/qwen3_vl_30b_a3b_pretrain_mfsdp_override_example.yaml
```

> **Note:** If you want to track experiments using Wandb, use the command below to ensure detailed model configuration recorded in your experiment overview. Otherwise, Wandb will only display the config file path (`--config-file conf/qwen3_vl_30b_a3b_pretrain_mfsdp_override_example.yaml`) instead of the actual configuration values.

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    finetune_qwen_vl.py \
    --recipe qwen3_vl_3b_active_30b_moe_finetune_config \
    mixed_precision=null \
    model.bf16=true \
    model.params_dtype=torch.bfloat16 \
    model.pipeline_dtype=torch.bfloat16 \
    model.tensor_model_parallel_size=1 \
    model.expert_model_parallel_size=8 \
    model.freeze_language_model=false \
    model.freeze_vision_model=false \
    model.freeze_vision_projection=false \
    model.init_model_with_meta_device=true \
    model.seq_length=4096 \
    model.gradient_accumulation_fusion=false \
    model.calculate_per_token_loss=true \
    model.moe_token_dispatcher_type=alltoall \
    model.recompute_granularity=full \
    model.recompute_method=uniform \
    model.recompute_num_layers=1 \
    train.train_iters=20 \
    train.global_batch_size=32 \
    train.micro_batch_size=1 \
    train.eval_iters=5 \
    checkpoint.save=<your_cehckpoint_path> \
    checkpoint.ckpt_format=fsdp_dtensor \
    dist.use_megatron_fsdp=true \
    dist.use_torch_fsdp2=false \
    logger.log_interval=1 \
    logger.log_throughput=true \
    logger.log_throughput_to_tensorboard=true \
    logger.wandb_project=<your_wandb_project> \
    logger.wandb_exp_name=<your_wandb_exp_name> \
    ddp.grad_reduce_in_fp32=false \
    ddp.use_megatron_fsdp=true \
    ddp.use_distributed_optimizer=true \
    ddp.data_parallel_sharding_strategy=optim_grads_params
```

### Multi-Node Training

For multi-node training on a SLURM cluster, refer to the example scripts：
- [Qwen3-VL-30B-A3B-Instruct](../../../examples/recipes/qwen_vl/scripts/sbatch_qwen3_vl_30b_a3b_mfsdp.sh).
- Qwen3-VL-235B-A22B-Instruct: [B200](../../../examples/recipes/qwen_vl/scripts/sbatch_qwen3_vl_235b_a22b_mfsdp_b200.sh) | [GB200](../../../examples/recipes/qwen_vl/scripts/sbatch_qwen3_vl_235b_a22b_mfsdp_gb200.sh).