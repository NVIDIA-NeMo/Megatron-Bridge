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
from pathlib import Path

import nemo_run as run

from nemo_lm.models.utils import forward_step
from nemo_lm.recipes.llm.llama3_8b import pretrain_config
from nemo_lm.training.config import ConfigContainer, ProfilingConfig
from nemo_lm.training.pretrain import megatron_pretrain
from nemo_lm.utils.nemo_run_utils import prepare_config_for_nemo_run

logger: logging.Logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    logger.info(f"Nemo Run Launcher for Llama3 8B using run.Partial")
    logger.info(f"=================================================")

    # Get the base ConfigContainer from the recipe
    cfg: ConfigContainer = pretrain_config()

    # Example of applying programmatic overrides
    cfg.train_config.train_iters = 10
    cfg.logger_config.log_interval = 50
    if cfg.profiling_config is None:
        cfg.profiling_config = ProfilingConfig()
    cfg.profiling_config.use_nsys_profiler = False
    cfg.profiling_config.use_pytorch_profiler = True
    cfg.profiling_config.record_shapes = True

    # Prepare the configuration for NeMo Run using the utility function
    cfg = prepare_config_for_nemo_run(cfg)

    # Create a run.Partial object for the main training function
    train_fn = run.Partial(megatron_pretrain, config=cfg, forward_step_func=forward_step)

    logger.info(f"Launching locally with TorchRun with nproc_per_node={args.nproc_per_node}")
    executor = run.LocalExecutor(ntasks_per_node=args.nproc_per_node, launcher="torchrun")

    run.run(train_fn, executor=executor, dryrun=args.dryrun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example launcher for Llama3 8B pretraining using nemo_run.Partial.")
    parser.add_argument(
        "--nproc-per-node", type=int, default=1, help="Number of processes per node (typically number of GPUs)."
    )
    parser.add_argument("--dryrun", action="store_true", help="Dry run the script.")

    cmd_args = parser.parse_args()
    main(cmd_args)
