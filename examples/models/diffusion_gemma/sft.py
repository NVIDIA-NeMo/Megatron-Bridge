# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Run offline native DiffusionGemma SFT (TP=EP=PP=CP=1).

Launch with ``uv run python -m torch.distributed.run --nproc_per_node=1``.
Provide local dataset/tokenizer paths and HF or native DiffusionGemma weights.
HF initialization maps text weights through AutoBridge; training checkpoints remain native.
"""

import argparse
from pathlib import Path

from megatron.bridge.diffusion.models.diffusion_gemma.step import DiffusionGemmaStep
from megatron.bridge.diffusion.recipes.diffusion_gemma.sft import diffusion_gemma_sft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.pretrain import pretrain


def main() -> None:
    """Configure local artifacts and launch the existing Bridge trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Local train/validation JSONL directory")
    parser.add_argument("--tokenizer-model", required=True, help="Existing local tokenizer directory")
    parser.add_argument("--hf-model", help="HF DiffusionGemma model ID or local checkpoint for initial weights")
    parser.add_argument("--pretrained-checkpoint", help="Native Megatron DiffusionGemma checkpoint")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--train-iters", type=int, default=1000)
    parser.add_argument("--encoder-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Fixture-only random initialization; does not validate pretrained checkpoint loading or model quality",
    )
    args = parser.parse_args()
    for name in ("dataset_root", "tokenizer_model"):
        if not Path(getattr(args, name)).is_dir():
            option = name.replace("_", "-")
            parser.error(f"--{option} must name an existing local directory")
    if args.pretrained_checkpoint and not Path(args.pretrained_checkpoint).is_dir():
        parser.error("--pretrained-checkpoint must name an existing native checkpoint directory")
    if not 0 < args.canvas_length <= args.seq_length:
        parser.error("canvas-length must be positive and no larger than seq-length")
    if sum(bool(value) for value in (args.hf_model, args.pretrained_checkpoint, args.allow_random_init)) != 1:
        parser.error("select exactly one of --hf-model, --pretrained-checkpoint, or --allow-random-init")
    cfg = diffusion_gemma_sft_config(
        dataset_root=args.dataset_root,
        tokenizer_model=args.tokenizer_model,
        pretrained_checkpoint=args.pretrained_checkpoint,
        experiment_dir=args.experiment_dir,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        train_iters=args.train_iters,
        allow_random_init=args.allow_random_init,
        hf_model=args.hf_model,
    )
    step = DiffusionGemmaStep(canvas_length=args.canvas_length, encoder_loss_weight=args.encoder_loss_weight)
    train = finetune if args.pretrained_checkpoint else pretrain
    train(config=cfg, forward_step_func=step)


if __name__ == "__main__":
    main()
