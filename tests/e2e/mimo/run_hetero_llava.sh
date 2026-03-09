#!/bin/bash
# Heterogeneous MIMO LLaVA training — LLM on ranks 0-3, CLIP on ranks 4-7.

GPUS_PER_NODE=8
NUM_NODES=1

uv run torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes "$NUM_NODES" \
    tests/e2e/mimo/test_mimo_training_llava.py \
    --micro-batch-size 2 \
    --global-batch-size 32 \
    --train-iters 500 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --log-interval 1 \
    --lr 1e-4 \
    --lr-warmup-iters 20 \
    --min-lr 2.0e-5 \
    --weight-decay 0.01 \
    --wandb-project "Megatron-Bridge-MIMO" \
    --wandb-exp-name "mimo-llava-e2e-test" \
    --wandb-save-dir "/tmp/wandb" \
    --dataset-root /path/to/llava/pretrain/dataset
