# Full finetuning with TP=2, PP=1
torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe gemma3_vl_4b_finetune_config \
    --step_func vlm_step \
    checkpoint.pretrained_checkpoint=/models/gemma-3-4b-it \
    model.seq_length=4096 \
    train.train_iters=20 \
    train.global_batch_size=8 \
    train.micro_batch_size=1 \
    train.eval_iters=5 \
    optimizer.lr=0.00025 \
    optimizer.min_lr=0.000025 \
    scheduler.lr_warmup_iters=10 \
    checkpoint.save=/result/gemma3_vl_4b_sft_tp2 \
    logger.log_interval=1 \
    logger.wandb_project=mbridge_gemma3_vl \
    logger.wandb_exp_name=full_finetune_tp2 \
    logger.wandb_entity=joc \
    dataset.seq_length=4096 \
    rng.seed=42 \
    ddp.grad_reduce_in_fp32=true \
    model.tensor_model_parallel_size=2 \
    model.pipeline_model_parallel_size=1

# Full finetuning with TP=1, PP=2
torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe gemma3_vl_4b_finetune_config \
    --step_func vlm_step \
    checkpoint.pretrained_checkpoint=/models/gemma-3-4b-it \
    model.seq_length=4096 \
    train.train_iters=20 \
    train.global_batch_size=8 \
    train.micro_batch_size=1 \
    train.eval_iters=5 \
    optimizer.lr=0.00025 \
    optimizer.min_lr=0.000025 \
    scheduler.lr_warmup_iters=10 \
    checkpoint.save=/result/gemma3_vl_4b_sft_pp2 \
    logger.log_interval=1 \
    logger.wandb_project=mbridge_gemma3_vl \
    logger.wandb_exp_name=full_finetune_pp2 \
    logger.wandb_entity=joc \
    dataset.seq_length=4096 \
    rng.seed=42 \
    ddp.grad_reduce_in_fp32=true \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=2

# LoRA finetuning with TP=2, PP=1
torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe gemma3_vl_4b_finetune_config \
    --step_func vlm_step \
    --peft_scheme lora \
    checkpoint.pretrained_checkpoint=/models/gemma-3-4b-it \
    model.seq_length=4096 \
    train.train_iters=20 \
    train.global_batch_size=8 \
    train.micro_batch_size=1 \
    train.eval_iters=5 \
    optimizer.lr=0.0001 \
    optimizer.min_lr=0.00001 \
    scheduler.lr_warmup_iters=10 \
    checkpoint.save=/result/gemma3_vl_4b_lora_tp2 \
    logger.log_interval=1 \
    logger.wandb_project=mbridge_gemma3_vl \
    logger.wandb_exp_name=lora_finetune_tp2 \
    logger.wandb_entity=joc \
    dataset.seq_length=4096 \
    rng.seed=42 \
    ddp.grad_reduce_in_fp32=true \
    model.tensor_model_parallel_size=2 \
    model.pipeline_model_parallel_size=1

# LoRA finetuning with TP=1, PP=2
torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe gemma3_vl_4b_finetune_config \
    --step_func vlm_step \
    --peft_scheme lora \
    checkpoint.pretrained_checkpoint=/models/gemma-3-4b-it \
    model.seq_length=4096 \
    train.train_iters=20 \
    train.global_batch_size=8 \
    train.micro_batch_size=1 \
    train.eval_iters=5 \
    optimizer.lr=0.0001 \
    optimizer.min_lr=0.00001 \
    scheduler.lr_warmup_iters=10 \
    checkpoint.save=/result/gemma3_vl_4b_lora_pp2 \
    logger.log_interval=1 \
    logger.wandb_project=mbridge_gemma3_vl \
    logger.wandb_exp_name=lora_finetune_pp2 \
    logger.wandb_entity=joc \
    dataset.seq_length=4096 \
    rng.seed=42 \
    ddp.grad_reduce_in_fp32=true \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=2
