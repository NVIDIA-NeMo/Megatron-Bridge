#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe kimi_k25_vl_sft_config \
    --step_func vlm_step \
    --hf_path moonshotai/Kimi-K2.5 \
    model.seq_length=2048 \
    model.num_layers=2 \
    model.hidden_size=7168 \
    model.ffn_hidden_size=1024 \
    model.num_moe_experts=16 \
    model.moe_ffn_hidden_size=64 \
    model.tensor_model_parallel_size=1 \
    model.sequence_parallel=false \
    model.pipeline_model_parallel_size=2 \
    model.expert_model_parallel_size=1 \
    model.pipeline_model_parallel_layout=null \
    model.hf_model_path=moonshotai/Kimi-K2.5 \
    model.freeze_vision_model=true \
    model.freeze_vision_projection=true \
    model.moe_layer_freq=1 \
    train.train_iters=5000 \
    model.cross_entropy_loss_fusion=false \
    train.micro_batch_size=1 \
    train.global_batch_size=4 \
    dataset.maker_name=make_cord_v2_dataset \
    dataset.pack_sequences_in_batch=false \
    dataset.seq_length=2048 \
    checkpoint.save=../mbridge-kimi/ \
    checkpoint.load=../mbridge-kimi/ \
    checkpoint.save_interval=1000 \
    checkpoint.ckpt_step=100 \
    checkpoint.load_optim=false \
    logger.log_interval=1 \
    logger.log_throughput=true \
    logger.log_params_norm=true \
    logger.log_throughput_to_tensorboard=true \
    logger.log_timers_to_tensorboard=true \
    logger.log_validation_ppl_to_tensorboard=true \
    logger.log_l2_norm_grad_to_tensorboard=false \
    logger.wandb_project=mbridge-kimi \
    logger.wandb_save_dir=./outputs/kimi-k25-vl/ \
    logger.wandb_exp_name=kimi-pp2 \
    logger.wandb_entity=wplf \

