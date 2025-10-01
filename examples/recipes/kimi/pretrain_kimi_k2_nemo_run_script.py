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
NeMo Run Launcher for Kimi-K2 (1T) Pretraining.

This script launches the pretrain_kimi_k2.py script using NeMo Run with TorchRun,
while forwarding any additional command line arguments to the target script.

Examples:
    Basic usage with default config:
        $ python pretrain_kimi_k2_nemo_run_script.py --nproc-per-node=8

    Using a custom config file:
        $ python pretrain_kimi_k2_nemo_run_script.py --nproc-per-node=8 --config-file=my_config.yaml

    Passing additional overrides to the target script:
        $ python pretrain_kimi_k2_nemo_run_script.py --nproc-per-node=8 \
            model.tensor_model_parallel_size=2 \
            train.train_iters=100000

    Using both custom config and CLI overrides:
        $ python pretrain_kimi_k2_nemo_run_script.py --nproc-per-node=8 \
            --config-file=conf/my_custom_config.yaml \
            optimizer.lr=0.0003 \
            train.global_batch_size=4096

    Dry run to see what would be executed:
        $ python pretrain_kimi_k2_nemo_run_script.py --nproc-per-node=8 --dryrun \
            model.pipeline_dtype=torch.bfloat16

Argument Forwarding:
    Any arguments not recognized by this launcher script will be forwarded
    to the target pretrain_kimi_k2.py script as Hydra-style overrides.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple
from typing import Optional
import nemo_run as run


logger: logging.Logger = logging.getLogger(__name__)

# Define paths relative to this script's location
# Assumes this script (pretrain_kimi_k2_nemo_run_script.py) is in Megatron-Bridge/examples/recipes/kimi/
# and pretrain_kimi_k2.py is in the same directory,
# and the config is in a 'conf' subdirectory.
SCRIPT_DIR: Path = Path(__file__).parent.resolve()
PRETRAIN_SCRIPT_FILENAME: str = "pretrain_kimi_k2.py"
PRETRAIN_SCRIPT_PATH: Path = SCRIPT_DIR / PRETRAIN_SCRIPT_FILENAME
DEFAULT_CONFIG_FILENAME: str = "kimi_k2_pretrain_override_example.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def slurm_executor_aot(
    user: str = "aot",
    host: str = "login-eos",
    remote_job_dir: str = "/lustre/fsw/coreai_dlalgo_llm/aot/exp/nemorun",
    account: str = "coreai_dlalgo_ci",
    partition: str = "batch",
    nodes: int = 1,
    devices: int = 8,
    time: str = "00:30:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = None,
    retries: int = 0,
    nemo_home_dir: str = "/lustre/fsw/coreai_dlalgo_llm/nemo_home",
    hf_home_dir: str = "/lustre/fsw/coreai_dlalgo_llm/aot/.cache/",
    ssh_identity: str = "/home/aot/.ssh/id_ed25519",
    dependency_type: str = "afterok",
) -> run.SlurmExecutor:
    if not (user and host and remote_job_dir and account and partition and nodes and devices):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this function."
        )

    # CHANGE ME
    mounts = []
    # Custom mounts are defined here.
    if custom_mounts:
        mounts.extend(custom_mounts)

    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NEMO_LOG_MEMORY_USAGE": "1",
        "TORCH_LOGS": "recompiles",
        "HF_HOME": hf_home_dir,
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    # This defines the slurm executor.
    # We connect to the executor via the tunnel defined by user, host and remote_job_dir.
    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir, # This is where the results of the run will be stored by default.
            identity=ssh_identity # OPTIONAL: Provide path to the private key that can be used to establish the SSH connection without entering your password.
        ),
        nodes=nodes,
        ntasks_per_node=devices,
        # gpus_per_node=devices,  # eos can't have this
        mem="0",
        exclusive=True,
        # gres="gpu:8",  # eos can't have this
        packager=run.Packager(),
        dependency_type=dependency_type,
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time
    executor.srun_args = ["--mpi=pmix", "--container-writable"]
    if "cw-dfw" in host:
        executor.gres = "gpu:8"
    return executor



def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating launcher args from target script args."""
    parser = argparse.ArgumentParser(
        description="Launcher for Kimi-K2 (1T) pretraining using nemo_run and TorchRun. "
        "Additional arguments will be forwarded to pretrain_kimi_k2.py",
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
        help="Path to the YAML override config file for the pretrain_kimi_k2.py script.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dry run the script without actually running it.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=64,
        help="Number of nodes to use for the run.",
    )
    parser.add_argument(
        "--optimizer-type",
        type=str,
        default="muon",
        help="Optimizer type, [muon, adam].",
    )
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="Use proxy model.",
    )

    # Parse known args for the launcher, remaining will be forwarded to target script
    args, forwarded_args = parser.parse_known_args()
    return args, forwarded_args


def main() -> None:
    """
    Main function for script demonstrating how to use the NeMo Run executor.
    """
    args, forwarded_args = parse_cli_args()

    logger.info("Nemo Run Launcher for Kimi-K2 (1T) Pretraining")
    logger.info("=============================================")

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
        "--optimizer-type",
        args.optimizer_type,
    ]
    if args.proxy:
        target_script_args.append("--proxy")

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
    executor = slurm_executor_aot(
        nodes=args.nodes,
        host='login-eos01.eos.clusters.nvidia.com',
        remote_job_dir="/lustre/fsw/coreai_dlalgo_llm/aot/exp/mbridge/kimi_k2_pretrain",
        partition="batch",
        time="01:00:00",
        container_image="/lustre/fsw/coreai_dlalgo_llm/aot/sqsh/mbridge-kimi.sqsh",
        custom_mounts=[
            "/lustre:/lustre",
            "/lustre/fsw/coreai_dlalgo_llm/aot:/aot",
            "/lustre/fsw/coreai_dlalgo_llm/aot/codebases/Megatron-Bridge:/opt/mbridge",
        ],
    )

    # Execute the run
    run.run(train_script, executor=executor, dryrun=args.dryrun)


if __name__ == "__main__":
    main()
