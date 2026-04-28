#!/bin/bash
# Heterogeneous MIMO LLaVA training — LLM on ranks 0-3, CLIP on ranks 4-7.

GPUS_PER_NODE=8
NUM_NODES=1

uv run torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes "$NUM_NODES" \
    examples/models/megatron_mimo/megatron_mimo_training_llava.py \
    --micro-batch-size 4 \
    --global-batch-size 128 \
    --train-iters 1000 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --log-interval 1 \
    --lr 1e-3 \
    --lr-warmup-iters 60 \
    --min-lr 2.0e-5 \
    --weight-decay 0.0 \
    --wandb-project "Megatron-Bridge-MIMO" \
    --wandb-exp-name "mimo-llava-hetero-e2e-test" \
    --wandb-save-dir "/tmp/wandb" \
    --vision-encoder-checkpoint /path/to/clip_checkpoint \
    --language-model-checkpoint /path/to/llm_checkpoint \
    --dataset-root /path/to/llava/pretrain/dataset
