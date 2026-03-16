#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe kimi_k25_vl_sft_config \
    --dataset_type hf \
    --step_func vlm_step \
    --hf_path moonshotai/Kimi-K2.5 \
    model.seq_length=4096 \
    model.num_layers=2 \
    model.hidden_size=7168 \
    model.ffn_hidden_size=1024 \
    model.num_moe_experts=16 \
    model.moe_ffn_hidden_size=64 \
    model.tensor_model_parallel_size=2 \
    model.pipeline_model_parallel_size=2 \
    model.expert_model_parallel_size=2 \
    model.pipeline_model_parallel_layout=null \
    model.hf_model_path=moonshotai/Kimi-K2.5 \
    model.moe_layer_freq=1 \
    train.train_iters=5000 \
    model.cross_entropy_loss_fusion=false \
    model.sequence_parallel=true \
    train.micro_batch_size=1 \
    train.global_batch_size=4 \
    dataset.maker_name=make_cord_v2_dataset \
    dataset.pack_sequences_in_batch=false \
