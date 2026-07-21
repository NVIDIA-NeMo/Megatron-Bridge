#!/usr/bin/env python3
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

"""Pure forward-only evaluation with the existing Bridge eval helper.

Run on one or more GPUs:

    uv run python -m torch.distributed.run --nproc_per_node=2 examples/evaluation/pure_eval.py --tp-size 2

The script builds a tiny GPT model and mock GPT dataset, creates Bridge runtime
state with the standard setup path, and calls ``evaluate_and_print_results()``
directly. For a ModelOpt calibration workflow that already has a Megatron model,
the important public call is the same ``evaluate_and_print_results()`` call near
the end of ``main``.
"""

import argparse

import torch

from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.recipes.gpt.vanilla_gpt import vanilla_gpt_pretrain_config
from megatron.bridge.training.config import ConfigContainer, MockGPTDatasetConfig, runtime_config_update
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.setup import setup
from megatron.bridge.training.state import GlobalState
from megatron.bridge.utils.common_utils import print_rank_0


def parse_args() -> argparse.Namespace:
    """Parse example arguments."""
    parser = argparse.ArgumentParser(description="Run a pure Bridge evaluation smoke test.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GPT layers.")
    parser.add_argument("--hidden-size", type=int, default=128, help="GPT hidden size.")
    parser.add_argument("--num-attention-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length.")
    parser.add_argument("--eval-iters", type=int, default=2, help="Number of evaluation iterations.")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size.")
    parser.add_argument("--global-batch-size", type=int, default=4, help="Global batch size.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ConfigContainer:
    """Build a small mock-data config for pure eval."""
    cfg = vanilla_gpt_pretrain_config()

    cfg.model.tensor_model_parallel_size = args.tp_size
    cfg.model.num_layers = args.num_layers
    cfg.model.hidden_size = args.hidden_size
    cfg.model.num_attention_heads = args.num_attention_heads
    cfg.model.seq_length = args.seq_length
    cfg.model.apply_rope_fusion = False

    cfg.dataset = MockGPTDatasetConfig(
        seq_length=args.seq_length,
        random_seed=1234,
        split="9999,8,2",
        dataloader_type="single",
        num_workers=0,
        persistent_workers=False,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    cfg.dataset.seq_length = args.seq_length
    cfg.dataset.dataloader_type = "single"
    cfg.dataset.num_workers = 0
    cfg.dataset.persistent_workers = False

    cfg.train.train_iters = 1
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.train.global_batch_size = args.global_batch_size

    cfg.validation.skip_train = True
    cfg.validation.eval_interval = 1
    cfg.validation.eval_iters = args.eval_iters
    cfg.validation.eval_micro_batch_size = args.micro_batch_size
    cfg.validation.eval_global_batch_size = args.global_batch_size

    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_decay_iters = 1
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.logger.tensorboard_dir = None

    return cfg


def main() -> None:
    """Run pure evaluation and print the returned loss dictionary."""
    cfg = build_config(parse_args())
    runtime_config_update(cfg)

    state = GlobalState()
    state.cfg = cfg

    setup_output = setup(state, get_dataset_provider(cfg.dataset))
    data_iterator = setup_output.valid_data_iterator or setup_output.train_data_iterator

    losses = evaluate_and_print_results(
        state=setup_output.state,
        prefix="pure eval",
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=setup_output.model,
        config=cfg.model,
        verbose=True,
        write_to_tensorboard=False,
        pg_collection=setup_output.pg_collection,
    )
    if losses is not None:
        loss_values = {name: loss.item() for name, loss in losses.items()}
        print_rank_0(f"pure eval returned losses: {loss_values}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
