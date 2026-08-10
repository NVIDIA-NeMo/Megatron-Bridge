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

"""Export native BAGEL initialization as an iteration-0 Bridge checkpoint."""

import argparse
import logging
import os
from pathlib import Path

import torch
from megatron.core import parallel_state

from megatron.bridge.models.bagel.bagel_step import BagelForwardStep
from megatron.bridge.recipes.bagel.h100.bagel import (
    bagel_7b_finetune_8gpu_h100_bf16_config,
    bagel_7b_pretrain_8gpu_h100_bf16_config,
)
from megatron.bridge.training.callbacks import CallbackContext, CallbackManager
from megatron.bridge.training.checkpointing import save_checkpoint
from megatron.bridge.training.pretrain import pretrain


logger = logging.getLogger(__name__)


class _ExportComplete(Exception):
    """Stop setup after the initialization checkpoint is durable."""


def parse_args() -> argparse.Namespace:
    """Parse checkpoint export arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=("pretrain", "finetune"), default="pretrain")
    parser.add_argument("--bagel-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--native-model-checkpoint", type=Path, required=True)
    parser.add_argument("--official-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _save_initialization(output: Path, context: CallbackContext) -> None:
    if context.optimizer is None or context.scheduler is None:
        raise RuntimeError("BAGEL initialization export requires optimizer and scheduler state")
    model_config = context.state.cfg.model
    model_config.native_model_checkpoint = None
    model_config.native_model_seed = None
    model_config.native_world_size = None
    model_config.validate_native_checkpoint_metadata = True
    save_checkpoint(
        state=context.state,
        model=context.model,
        optimizer=context.optimizer,
        opt_param_scheduler=context.scheduler,
        num_floating_point_operations_so_far=0,
    )
    logger.info("Saved BAGEL iteration-0 Bridge checkpoint to %s", output)
    raise _ExportComplete


def main() -> None:
    """Build native BAGEL model and optimizer, then save before opening data."""
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
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
    cfg.model.reference_training_seed = args.seed * world_size
    cfg.model.reference_training_world_size = world_size
    cfg.model.reset_reference_training_rng = True
    cfg.dataset.dataset_root = str(args.dataset_root.resolve())
    cfg.dataset.bagel_repo = str(args.bagel_repo.resolve())
    cfg.dataset.tokenizer_model = str(args.tokenizer_model.resolve())
    cfg.dataset.seed = args.seed
    cfg.dataset.data_seed = args.seed
    cfg.train.train_iters = 1
    cfg.train.global_batch_size = world_size
    cfg.rng.seed = args.seed
    cfg.checkpoint.save = str(output)
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = 0
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.save_rng = True
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.checkpoint.async_save = False
    cfg.checkpoint.fully_parallel_save = False
    callbacks = CallbackManager()
    callbacks.register("on_data_init_start", lambda context: _save_initialization(output, context))
    try:
        pretrain(config=cfg, forward_step_func=BagelForwardStep(), callbacks=callbacks)
    except _ExportComplete:
        pass
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
