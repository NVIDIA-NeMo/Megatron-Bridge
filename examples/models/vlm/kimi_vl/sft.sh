#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe kimi_k25_vl_pretrain_config \
    --dataset_type hf \
    --step_func vlm_step \
    --hf_path moonshotai/Kimi-K2.5 \
    model.seq_length=4096 \
    model.num_layers=2 \
    model.tensor_model_parallel_size=2 \
    model.pipeline_model_parallel_size=1 \
    model.expert_model_parallel_size=4 \
    model.pipeline_model_parallel_layout=null \
    model.hf_model_path=moonshotai/Kimi-K2.5 \
    model.moe_layer_freq=1 \
    train.train_iters=5000 \
    model.cross_entropy_loss_fusion=false \
    model.sequence_parallel=true \
    train.micro_batch_size=2 \
    train.global_batch_size=8 \
    model.num_moe_experts=96 \
    # +dataset.maker_name=make_cord_v2_dataset \
    # +dataset.pack_sequences_in_batch=true \