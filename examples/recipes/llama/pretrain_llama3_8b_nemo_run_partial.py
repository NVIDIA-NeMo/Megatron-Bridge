#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import logging

import nemo_run as run

from megatron.bridge.recipes.llama import llama3_8b_pretrain_config as pretrain_config
from megatron.bridge.recipes.utils.nemo_run_utils import get_partial_fn
from megatron.bridge.training.config import ConfigContainer, ProfilingConfig
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


logger: logging.Logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """
    Example launcher for Llama3 8B pretraining using nemo_run.Partial.
    """
    logger.info("Nemo Run Launcher for Llama3 8B using run.Partial")
    logger.info("=================================================")

    # Get the base ConfigContainer from the recipe
    cfg: ConfigContainer = pretrain_config()

    # Example of applying programmatic overrides
    cfg.train.train_iters = 20
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.eval_iters = 0

    cfg.scheduler.lr_warmup_iters = 5

    cfg.logger.log_interval = 1

    cfg.dataset.sequence_length = 4096
    cfg.checkpoint.save = None

    if cfg.profiling is None:
        cfg.profiling = ProfilingConfig()
    cfg.profiling.use_nsys_profiler = False
    cfg.profiling.use_pytorch_profiler = True
    cfg.profiling.record_shapes = True

    # Create a run.Partial object for the pretrain function
    fn = get_partial_fn(pretrain, cfg, forward_step)

    logger.info(f"Launching locally with TorchRun with nproc_per_node={args.nproc_per_node}")
    executor = run.LocalExecutor(ntasks_per_node=args.nproc_per_node, launcher="torchrun")

    run.run(fn, executor=executor, dryrun=args.dryrun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example launcher for Llama3 8B pretraining using nemo_run.Partial.")
    parser.add_argument(
        "--nproc-per-node", type=int, default=2, help="Number of processes per node (typically number of GPUs)."
    )
    parser.add_argument("--dryrun", action="store_true", help="Dry run the script.")

    cmd_args = parser.parse_args()
    main(cmd_args)
