# Kimi-K2.5-VL Tiny-Model Validation

Step-by-step guide to validate the full Kimi VL pipeline (creation, conversion,
inference, comparison, training) using a small toy model — no full-size
checkpoint download required.

## Prerequisites

```bash
export WORKSPACE=/your/custom/path
```

## Step 1: Create Toy Model

```bash
python examples/models/vlm/kimi_vl/create_hf_toy_model.py \
    --hf-model-id moonshotai/Kimi-K2.5 \
    --output-dir ../kimi/kimi_toy \
    --num-experts 16 \
    --num-hidden-layers 2
```

## Step 2: Checkpoint Conversion (HF → Megatron → HF)

**Import** the toy HF checkpoint into Megatron format:

```bash
python examples/conversion/convert_checkpoints.py import \
    --hf-model ../kimi/kimi_toy \
    --megatron-path ${WORKSPACE}/models/Kimi-K2.5 \
    --trust-remote-code
```

**Export** back to HF format for round-trip verification:

```bash
python examples/conversion/convert_checkpoints.py export \
    --hf-model ../kimi/kimi_toy \
    --megatron-path ${WORKSPACE}/models/Kimi-K2.5/iter_0000000 \
    --hf-path ${WORKSPACE}/models/Kimi-K2.5-hf-export
```

## Step 3: HF vs Megatron Comparison

Compare 1-step forward-pass outputs between HuggingFace and Megatron:

```bash
torchrun --nproc_per_node=2 examples/models/vlm/kimi_vl/compare.py \
    --hf_model_path ../kimi/kimi_toy \
    --trust_remote_code \
    --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "Describe this image." \
    --tp 2 --ep 2
```

## Step 4: Inference (HF-to-Megatron Generation)

Run greedy auto-regressive generation through the Megatron model:

```bash
torchrun --nproc_per_node=2 examples/models/vlm/kimi_vl/hf_to_megatron_generate_vlm.py \
    --hf_model_path ../kimi/kimi_toy \
    --trust_remote_code \
    --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "Describe this image." \
    --tp 1 --ep 2 --pp 1
```

## Step 5: SFT Training (Minimal)

Quick smoke test with default recipe settings:

```bash
torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe kimi_k25_vl_sft_config \
    --step_func vlm_step \
    --hf_path moonshotai/Kimi-K2.5 \
    model.hf_model_path=../kimi/kimi_toy \
    dataset.maker_name=make_cord_v2_dataset
```

## Step 6: SFT Training (Full Configuration)

Full training run with explicit parallelism, logging, and checkpoint settings:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe kimi_k25_vl_sft_config \
    --step_func vlm_step \
    --hf_path moonshotai/Kimi-K2.5 \
    model.seq_length=2048 \
    model.tensor_model_parallel_size=2 \
    model.sequence_parallel=true \
    model.pipeline_model_parallel_size=1 \
    model.expert_model_parallel_size=2 \
    model.pipeline_model_parallel_layout=null \
    model.hf_model_path=moonshotai/Kimi-K2.5 \
    model.freeze_vision_model=true \
    model.freeze_vision_projection=true \
    model.calculate_per_token_loss=true \
    model.cross_entropy_loss_fusion=false \
    model.hidden_size=7168 \
    model.ffn_hidden_size=1024 \
    model.num_moe_experts=16 \
    model.moe_ffn_hidden_size=64 \
    train.train_iters=5000 \
    train.micro_batch_size=2 \
    train.global_batch_size=8 \
    dataset.maker_name=make_cord_v2_dataset \
    dataset.pack_sequences_in_batch=false \
    dataset.seq_length=2048 \
    checkpoint.save=../kimi-test \
    checkpoint.load=../kimi-test \
    checkpoint.load_rng=false \
    checkpoint.save_interval=1000 \
    checkpoint.ckpt_step=100 \
    checkpoint.load_optim=false \
    ddp.average_in_collective=false \
    logger.log_interval=1 \
    logger.log_throughput=true \
    logger.log_params_norm=true \
    logger.log_throughput_to_tensorboard=true \
    logger.log_timers_to_tensorboard=true \
    logger.log_validation_ppl_to_tensorboard=true \
    logger.log_l2_norm_grad_to_tensorboard=false \
    logger.wandb_project=mbridge-kimi \
    logger.wandb_save_dir=./outputs/kimi-k25-vl/ \
    logger.wandb_exp_name=kimi-tp2-pp1-ep2-tiny \
```

Note: The toy model overrides (`hidden_size`, `ffn_hidden_size`, `num_moe_experts`,
`moe_ffn_hidden_size`) to ensure that model can run in one node.
