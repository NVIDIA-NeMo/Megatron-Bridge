#!/bin/bash
# Heterogeneous MIMO LLaVA smoke test against the mini audio-augmented dataset.
# LLM on ranks 0-3 (TP=4), CLIP on ranks 4-5 (TP=2), Whisper on ranks 6-7 (TP=2).
#
# Assumes ./prepare_llava_pretrain_audio.sh has been run.

set -euo pipefail

GPUS_PER_NODE=8
NUM_NODES=1

uv run torchrun \
    --nproc_per_node "$GPUS_PER_NODE" \
    --nnodes "$NUM_NODES" \
    examples/models/megatron_mimo/megatron_mimo_training_llava_audio.py \
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
    --wandb-exp-name "mimo-llava-audio-hetero-e2e-test" \
    --wandb-save-dir "/tmp/wandb" \
    --dataset-root /path/to/LLaVA-Pretrain-Audio-Augmented \
    --hf-data-files "blip_laion_cc_sbu_558k_with_audio.json" \
    --audio-column audio \
    --vision-encoder-checkpoint /path/to/clip_checkpoint \
    --language-model-checkpoint /path/to/llm_checkpoint \
    --audio-encoder-checkpoint /path/to/whisper_checkpoint
