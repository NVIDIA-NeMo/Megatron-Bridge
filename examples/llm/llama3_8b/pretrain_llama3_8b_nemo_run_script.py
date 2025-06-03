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
import os
from pathlib import Path

import nemo_run as run

logger: logging.Logger = logging.getLogger(__name__)

# Define paths relative to this script's location
# Assumes this script (pretrain_llama3_8b_nemo_run_script.py) is in NeMo-LM/examples/llm/llama3_8b/
# and pretrain_llama3_8b.py is in the same directory,
# and the config is in a 'conf' subdirectory.
SCRIPT_DIR: Path = Path(__file__).parent.resolve()
PRETRAIN_SCRIPT_FILENAME: str = "pretrain_llama3_8b.py"
PRETRAIN_SCRIPT_PATH: Path = SCRIPT_DIR / PRETRAIN_SCRIPT_FILENAME
DEFAULT_CONFIG_FILENAME: str = "llama3_8b_pretrain_override_example.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def main(args: argparse.Namespace) -> None:
    """
    Main function for script demonstrating how to use the NeMo Run executor.
    """
    logger.info(f"Nemo Run Launcher for Llama3 8B Pretraining")
    logger.info(f"===========================================")

    if not PRETRAIN_SCRIPT_PATH.is_file():
        logger.error(f"Target pretraining script not found: {PRETRAIN_SCRIPT_PATH}")
        logger.error(f"Please ensure '{PRETRAIN_SCRIPT_FILENAME}' exists in the same directory as this launcher.")
        sys.exit(1)

    config_file_to_use = Path(args.config_file).resolve()
    if not config_file_to_use.is_file():
        logger.error(f"Specified YAML config file not found: {config_file_to_use}")
        logger.error(f"Ensure the path passed to --config_file is correct.")
        sys.exit(1)

    train_script = run.Script(
        path=str(PRETRAIN_SCRIPT_PATH),
        entrypoint="python",
        args=[
            "--config-file",
            str(config_file_to_use),
        ],
    )

    # Define the executor
    logger.info(f"Launching locally with TorchRun with nproc_per_node={args.nproc_per_node}")
    executor = run.LocalExecutor(ntasks_per_node=args.nproc_per_node, launcher="torchrun")

    # Execute the run
    run.run(train_script, executor=executor, dryrun=args.dryrun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launcher for Llama3 8B pretraining using nemo_run and TorchRun.")
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
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

    cmd_args = parser.parse_args()
    main(cmd_args)
