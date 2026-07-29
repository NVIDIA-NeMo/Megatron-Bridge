# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run BAGEL training from a validated official checkpoint."""

import argparse
import os
from pathlib import Path

import torch

from megatron.bridge.models.bagel.bagel_step import BagelForwardStep
from megatron.bridge.recipes.bagel.h100.bagel import (
    bagel_7b_finetune_8gpu_h100_bf16_config,
    bagel_7b_pretrain_8gpu_h100_bf16_config,
)
from megatron.bridge.training.pretrain import pretrain


def parse_args() -> argparse.Namespace:
    """Parse BAGEL training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=("pretrain", "finetune"), default="pretrain")
    parser.add_argument("--bagel-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--native-model-checkpoint", type=Path, required=True)
    parser.add_argument("--official-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-iters", type=int, default=1)
    parser.add_argument("--tensorboard-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    """Configure and run the first supported BAGEL training topology."""
    args = parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    cfg = (
        bagel_7b_finetune_8gpu_h100_bf16_config()
        if args.recipe == "finetune"
        else bagel_7b_pretrain_8gpu_h100_bf16_config()
    )
    cfg.model.bagel_repo = str(args.bagel_repo.resolve())
    cfg.model.model_path = str(args.model_path.resolve())
    cfg.model.vae_path = str((args.model_path / "ae.safetensors").resolve())
    cfg.model.native_model_checkpoint = str(args.native_model_checkpoint.resolve())
    cfg.model.native_model_seed = args.seed * world_size
    cfg.model.native_world_size = world_size
    cfg.model.validate_native_checkpoint_metadata = not args.official_ema
    cfg.model.reset_reference_training_rng = True
    cfg.dataset.dataset_root = str(args.dataset_root.resolve())
    cfg.dataset.bagel_repo = str(args.bagel_repo.resolve())
    cfg.dataset.tokenizer_model = str(args.tokenizer_model.resolve())
    cfg.dataset.seed = args.seed
    cfg.dataset.data_seed = args.seed
    cfg.train.train_iters = args.train_iters
    cfg.train.global_batch_size = world_size
    cfg.rng.seed = args.seed
    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = str(args.tensorboard_dir.resolve()) if args.tensorboard_dir else None
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = 0
    cfg.checkpoint.async_save = False
    pretrain(config=cfg, forward_step_func=BagelForwardStep())
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
