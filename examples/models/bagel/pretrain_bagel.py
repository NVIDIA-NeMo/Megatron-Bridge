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

"""Run BAGEL training from a native or Bridge checkpoint."""

import argparse
import json
import os
from pathlib import Path

import torch
from megatron.core.optimizer_param_scheduler import get_canonical_lr_for_logging

from megatron.bridge.models.bagel.bagel_step import BagelForwardStep
from megatron.bridge.recipes.bagel.h100.bagel import (
    bagel_7b_finetune_8gpu_h100_bf16_config,
    bagel_7b_pretrain_8gpu_h100_bf16_config,
    bagel_7b_pretrain_32gpu_h100_bf16_config,
)
from megatron.bridge.training.callbacks import CallbackContext, CallbackManager
from megatron.bridge.training.pretrain import pretrain


def parse_args() -> argparse.Namespace:
    """Parse BAGEL training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=("pretrain", "pretrain-32gpu", "finetune"), default="pretrain")
    parser.add_argument("--bagel-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--native-model-checkpoint", type=Path)
    checkpoint.add_argument("--mcore-checkpoint", type=Path)
    parser.add_argument("--official-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-iters", type=int, default=1)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--loss-output", type=Path)
    parser.add_argument("--forward-output-prefix", type=Path)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--exit-interval", type=int)
    return parser.parse_args()


def _record_loss(output: Path, context: CallbackContext) -> None:
    if torch.distributed.get_rank() != 0:
        return
    if context.loss_dict is None or context.grad_norm is None or context.optimizer is None:
        raise RuntimeError("BAGEL loss callback did not receive training metrics")
    learning_rate = get_canonical_lr_for_logging(context.optimizer.param_groups)
    if learning_rate is None:
        raise RuntimeError("BAGEL optimizer has no canonical learning rate")
    row = {
        "step": context.state.train_state.step,
        "grad_norm": context.grad_norm,
        "lr": float(learning_rate),
        "losses": {name: value.detach().float().cpu().tolist() for name, value in sorted(context.loss_dict.items())},
    }
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _capture_forward(prefix: Path, context: CallbackContext) -> None:
    output_path = Path(f"{prefix}.rank{torch.distributed.get_rank()}.pt")
    if output_path.exists():
        raise ValueError(f"Forward output already exists: {output_path}")
    captured = False

    def capture(_module, _inputs, output):
        nonlocal captured
        if captured:
            return
        if not isinstance(output, tuple) or len(output) != 4:
            raise RuntimeError("BAGEL forward capture requires the PP=1 output tuple")
        ce, mse, mse_mask, _loss_mask = output
        torch.save(
            {
                "ce": ce.detach().float().cpu(),
                "mse": mse.detach().float().cpu(),
                "mse_mask": mse_mask.detach().cpu(),
            },
            output_path,
        )
        captured = True

    context.model[0].register_forward_hook(capture)


def main() -> None:
    """Configure and run the first supported BAGEL training topology."""
    args = parse_args()
    if args.official_ema and args.native_model_checkpoint is None:
        raise ValueError("--official-ema requires --native-model-checkpoint")
    if (args.checkpoint_output is None and args.save_interval != 0) or (
        args.checkpoint_output is not None and args.save_interval <= 0
    ):
        raise ValueError("--checkpoint-output and a positive --save-interval are required together")
    if args.loss_output is not None:
        args.loss_output = args.loss_output.resolve()
        if args.loss_output.exists():
            raise ValueError(f"Loss output already exists: {args.loss_output}")
        args.loss_output.parent.mkdir(parents=True, exist_ok=True)
    forward_prefix = None
    if args.forward_output_prefix is not None:
        forward_prefix = args.forward_output_prefix.resolve()
        forward_prefix.parent.mkdir(parents=True, exist_ok=True)
    world_size = int(os.environ["WORLD_SIZE"])
    recipes = {
        "pretrain": bagel_7b_pretrain_8gpu_h100_bf16_config,
        "pretrain-32gpu": bagel_7b_pretrain_32gpu_h100_bf16_config,
        "finetune": bagel_7b_finetune_8gpu_h100_bf16_config,
    }
    cfg = recipes[args.recipe]()
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.model.bagel_repo = str(args.bagel_repo.resolve())
    cfg.model.model_path = str(args.model_path.resolve())
    cfg.model.vae_path = str((args.model_path / "ae.safetensors").resolve())
    if args.native_model_checkpoint is not None:
        cfg.model.native_model_checkpoint = str(args.native_model_checkpoint.resolve())
        cfg.model.native_model_seed = args.seed * world_size
        cfg.model.native_world_size = world_size
        cfg.model.validate_native_checkpoint_metadata = not args.official_ema
    else:
        cfg.checkpoint.load = str(args.mcore_checkpoint.resolve())
        cfg.checkpoint.load_optim = True
        cfg.checkpoint.load_rng = True
        cfg.dataset.dataloader_load = cfg.checkpoint.load
    cfg.model.reference_training_seed = args.seed * world_size
    cfg.model.reference_training_world_size = world_size
    cfg.model.reset_reference_training_rng = True
    cfg.dataset.dataset_root = str(args.dataset_root.resolve())
    cfg.dataset.bagel_repo = str(args.bagel_repo.resolve())
    cfg.dataset.tokenizer_model = str(args.tokenizer_model.resolve())
    cfg.dataset.seed = args.seed
    cfg.dataset.data_seed = args.seed
    if args.checkpoint_output is not None:
        cfg.dataset.dataloader_save = str(args.checkpoint_output.resolve())
    cfg.train.train_iters = args.train_iters
    cfg.train.exit_interval = args.exit_interval
    cfg.train.global_batch_size = world_size
    cfg.rng.seed = args.seed
    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = str(args.tensorboard_dir.resolve()) if args.tensorboard_dir else None
    cfg.checkpoint.save = str(args.checkpoint_output.resolve()) if args.checkpoint_output else None
    cfg.checkpoint.save_interval = args.save_interval
    cfg.checkpoint.async_save = False
    callbacks = CallbackManager() if args.loss_output is not None or forward_prefix is not None else None
    if args.loss_output is not None:
        callbacks.register("on_train_step_end", lambda context: _record_loss(args.loss_output, context))
    if forward_prefix is not None:
        callbacks.register("on_data_init_start", lambda context: _capture_forward(forward_prefix, context))
    pretrain(config=cfg, forward_step_func=BagelForwardStep(), callbacks=callbacks)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
