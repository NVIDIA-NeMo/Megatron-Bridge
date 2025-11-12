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

"""
Launch Training Locally with NeMo-Run

This script demonstrates how to launch training scripts (pretrain or finetune)
using NeMo-Run's LocalExecutor with torchrun. This provides better job management
and logging compared to running torchrun directly.

Prerequisites: Install nemo-run

Usage:
    # Launch pretrain script
    python 02_launch_pretrain_local.py --script 00_quickstart_pretrain.py --devices 2

    # Launch finetune script
    python 02_launch_pretrain_local.py --script 00_quickstart_finetune.py --devices 1

    # Launch with YAML config
    python 02_launch_pretrain_local.py \
        --script 01_pretrain_with_yaml.py \
        --devices 2 \
        --config-file conf/llama32_1b_pretrain.yaml

    # Pass CLI overrides to the training script
    python 02_launch_pretrain_local.py \
        --script 01_finetune_with_yaml.py \
        --devices 2 \
        --script-args "train.train_iters=500 peft.dim=16"

    # Dry run (see what would be executed)
    python 02_launch_pretrain_local.py --script 00_quickstart_pretrain.py --dry-run
"""

import argparse
import logging
from pathlib import Path

import nemo_run as run


logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch training (pretrain/finetune) locally using NeMo-Run",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Training script to run (e.g., 00_quickstart_pretrain.py, 00_quickstart_finetune.py)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="YAML config file to pass to the training script (optional)",
    )
    parser.add_argument(
        "--script-args",
        type=str,
        default="",
        help='Additional arguments to pass to the training script (space-separated, e.g., "train.train_iters=100")',
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="megatron_bridge_training",
        help="Name for the experiment (default: megatron_bridge_training)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed without running",
    )

    return parser.parse_args()


def main() -> None:
    """Launch training (pretrain/finetune) using NeMo-Run LocalExecutor."""
    args = parse_args()

    # Resolve script path
    script_path = SCRIPT_DIR / args.script
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    # Build arguments for the training script
    script_args = []
    if args.config_file:
        script_args.extend(["--config-file", args.config_file])

    if args.script_args:
        # Split the script args string and add each arg
        script_args.extend(args.script_args.split())

    logger.info("Launching training with NeMo-Run LocalExecutor")
    logger.info(f"Script: {script_path.name}")
    logger.info(f"GPUs: {args.devices}")
    if args.config_file:
        logger.info(f"Config: {args.config_file}")
    if script_args:
        logger.info(f"Script args: {' '.join(script_args)}")
    logger.info("")

    # Create the training task
    task = run.Script(
        path=str(script_path),
        entrypoint="python",
        args=script_args,
    )

    # Create the local executor with torchrun
    executor = run.LocalExecutor(
        ntasks_per_node=args.devices,
        launcher="torchrun",
    )

    # Run the experiment
    with run.Experiment(args.experiment_name) as exp:
        exp.add(task, executor=executor, name="training")
        exp.run(detach=False, dryrun=args.dry_run)

    if not args.dry_run:
        logger.info("Training completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
