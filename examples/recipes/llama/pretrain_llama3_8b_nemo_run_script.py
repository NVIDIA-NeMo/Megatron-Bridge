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
NeMo Run Launcher for Llama3 8B Pretraining.

This script launches the pretrain_llama3_8b.py script using NeMo Run with TorchRun,
while forwarding any additional command line arguments to the target script.

Examples:
    Basic usage with default config:
        $ python pretrain_llama3_8b_nemo_run_script.py --nproc-per-node=8

    Using a custom config file:
        $ python pretrain_llama3_8b_nemo_run_script.py --nproc-per-node=8 --config-file=my_config.yaml

    Passing additional overrides to the target script:
        $ python pretrain_llama3_8b_nemo_run_script.py --nproc-per-node=8 \
            model.tensor_model_parallel_size=4 \
            train.train_iters=100000

    Using both custom config and CLI overrides:
        $ python pretrain_llama3_8b_nemo_run_script.py --nproc-per-node=8 \
            --config-file=conf/my_custom_config.yaml \
            optimizerg.lr=0.0002 \
            train.global_batch_size=512

    Dry run to see what would be executed:
        $ python pretrain_llama3_8b_nemo_run_script.py --nproc-per-node=8 --dryrun \
            model.pipeline_dtype=torch.float16

Argument Forwarding:
    Any arguments not recognized by this launcher script will be forwarded
    to the target pretrain_llama3_8b.py script as Hydra-style overrides.
"""

import argparse
import logging
import sys
from pathlib import Path

import nemo_run as run


logger: logging.Logger = logging.getLogger(__name__)

# Define paths relative to this script's location
# Assumes this script (pretrain_llama3_8b_nemo_run_script.py) is in Megatron-Bridge/examples/recipes/llama/
# and pretrain_llama3_8b.py is in the same directory,
# and the config is in a 'conf' subdirectory.
SCRIPT_DIR: Path = Path(__file__).parent.resolve()
PRETRAIN_SCRIPT_FILENAME: str = "pretrain_llama3_8b.py"
PRETRAIN_SCRIPT_PATH: Path = SCRIPT_DIR / PRETRAIN_SCRIPT_FILENAME
DEFAULT_CONFIG_FILENAME: str = "llama3_8b_pretrain_override_example.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def parse_cli_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating launcher args from target script args."""
    parser = argparse.ArgumentParser(
        description="Launcher for Llama3 8B pretraining using nemo_run and TorchRun. "
        "Additional arguments will be forwarded to pretrain_llama3_8b.py",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=2,
        help="Number of processes per node for TorchRun (typically number of GPUs).",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_FILE_PATH),
        help="Path to the YAML override config file for the pretrain_llama3_8b.py script.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dry run the script without actually running it.",
    )

    # Parse known args for the launcher, remaining will be forwarded to target script
    args, forwarded_args = parser.parse_known_args()
    return args, forwarded_args


def main() -> None:
    """
    Main function for script demonstrating how to use the NeMo Run executor.
    """
    args, forwarded_args = parse_cli_args()

    logger.info("Nemo Run Launcher for Llama3 8B Pretraining")
    logger.info("===========================================")

    if not PRETRAIN_SCRIPT_PATH.is_file():
        logger.error(f"Target pretraining script not found: {PRETRAIN_SCRIPT_PATH}")
        logger.error(f"Please ensure '{PRETRAIN_SCRIPT_FILENAME}' exists in the same directory as this launcher.")
        sys.exit(1)

    config_file_to_use = Path(args.config_file).resolve()
    if not config_file_to_use.is_file():
        logger.error(f"Specified YAML config file not found: {config_file_to_use}")
        logger.error("Ensure the path passed to --config_file is correct.")
        sys.exit(1)

    # Build the arguments list for the target script
    target_script_args = [
        "--config-file",
        str(config_file_to_use),
    ]

    # Add any forwarded arguments (Hydra-style overrides and other target script args)
    if forwarded_args:
        target_script_args.extend(forwarded_args)
        logger.info(f"Forwarding additional arguments to target script: {forwarded_args}")

    logger.info(f"Target script: {PRETRAIN_SCRIPT_PATH}")
    logger.info(f"Target script arguments: {target_script_args}")

    train_script = run.Script(
        path=str(PRETRAIN_SCRIPT_PATH),
        entrypoint="python",
        args=target_script_args,
    )

    # Define the executor
    logger.info(f"Launching locally with TorchRun with nproc_per_node={args.nproc_per_node}")
    executor = run.LocalExecutor(ntasks_per_node=args.nproc_per_node, launcher="torchrun")

    # Execute the run
    run.run(train_script, executor=executor, dryrun=args.dryrun)


if __name__ == "__main__":
    main()
