# Common configurations
PRETRAINED_CHECKPOINT=/models/gemma-3-4b-it
MODEL_NAME=gemma3_vl_4b
DATASET_NAME=raven
SEQ_LENGTH=4096
TRAIN_ITERS=200
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=1
EVAL_ITERS=0  # Raven dataset only supports train split
LR=0.0001
MIN_LR=0.00001
LR_WARMUP_ITERS=10
LOG_INTERVAL=1
WANDB_PROJECT=mbridge_gemma3_vl
SEED=42

# TP/PP combinations: "TP,PP"
PARALLELISM_CONFIGS=("2,1" "1,2")

for config in "${PARALLELISM_CONFIGS[@]}"; do
    IFS=',' read -r TP PP <<< "$config"
    
    echo "Running full finetuning with TP=$TP, PP=$PP"
    uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
        --recipe ${MODEL_NAME}_finetune_config \
        --step_func vlm_step \
        checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
        model.seq_length=$SEQ_LENGTH \
        train.train_iters=$TRAIN_ITERS \
        train.global_batch_size=$GLOBAL_BATCH_SIZE \
        train.micro_batch_size=$MICRO_BATCH_SIZE \
        train.eval_iters=$EVAL_ITERS \
        optimizer.lr=$LR \
        optimizer.min_lr=$MIN_LR \
        scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
        checkpoint.save=/result/${MODEL_NAME}_sft_tp${TP}_pp${PP} \
        logger.log_interval=$LOG_INTERVAL \
        logger.wandb_project=$WANDB_PROJECT \
        logger.wandb_exp_name=${MODEL_NAME}_${DATASET_NAME}_sft_tp${TP}_pp${PP} \
        dataset.maker_name=make_${DATASET_NAME}_dataset \
        dataset.seq_length=$SEQ_LENGTH \
        rng.seed=$SEED \
        ddp.grad_reduce_in_fp32=true \
        model.tensor_model_parallel_size=$TP \
        model.pipeline_model_parallel_size=$PP
done
