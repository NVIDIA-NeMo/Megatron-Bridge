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
Launch Training on Slurm with NeMo-Run

This script demonstrates how to launch training scripts (pretrain or finetune)
on a Slurm cluster using NeMo-Run. This enables easy multi-node training with
proper job management.

Prerequisites: Install nemo-run

Usage:
    # From the Slurm cluster (uses LocalTunnel)
    python 03_launch_pretrain_slurm.py \
        --script 00_quickstart_pretrain.py \
        --nodes 2 \
        --partition gpu \
        --account my_account

    # From your local machine (uses SSHTunnel)
    python 03_launch_pretrain_slurm.py \
        --script 00_quickstart_pretrain.py \
        --nodes 2 \
        --partition gpu \
        --account my_account \
        --ssh-tunnel \
        --host my-cluster.example.com \
        --user myusername \
        --remote-job-dir /home/myusername/nemo-runs

    # With custom SSH key
    python 03_launch_pretrain_slurm.py \
        --script 00_quickstart_pretrain.py \
        --nodes 2 \
        --partition gpu \
        --account my_account \
        --ssh-tunnel \
        --host my-cluster.example.com \
        --user myusername \
        --remote-job-dir /home/myusername/nemo-runs \
        --identity ~/.ssh/id_rsa

    # Launch with custom config
    python 03_launch_pretrain_slurm.py \
        --script 01_finetune_with_yaml.py \
        --nodes 1 \
        --partition gpu \
        --account my_account \
        --config-file conf/llama32_1b_finetune.yaml

    # With container and custom mounts
    python 03_launch_pretrain_slurm.py \
        --script 00_quickstart_pretrain.py \
        --nodes 2 \
        --partition gpu \
        --account my_account \
        --container-image /path/to/container.sqsh \
        --mount /data:/data

Note:
- Use --ssh-tunnel when launching from your local machine
- Omit --ssh-tunnel when already on the Slurm cluster (uses LocalTunnel)
- Adjust cluster-specific settings (account, partition, container paths)
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
        description="Launch training (pretrain/finetune) on Slurm using NeMo-Run",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Training script to run (e.g., 00_quickstart_pretrain.py, 00_quickstart_finetune.py)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to use (default: 1)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=8,
        help="GPUs per node (default: 8)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        required=True,
        help="Slurm partition name",
    )
    parser.add_argument(
        "--account",
        type=str,
        required=True,
        help="Slurm account name",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="04:00:00",
        help="Job time limit (default: 04:00:00)",
    )
    parser.add_argument(
        "--ssh-tunnel",
        action="store_true",
        help="Use SSH tunnel (for launching from local machine). Requires --host, --user, --remote-job-dir",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="SSH host for tunnel (required if --ssh-tunnel is set)",
    )
    parser.add_argument(
        "--user",
        type=str,
        help="SSH user for tunnel (required if --ssh-tunnel is set)",
    )
    parser.add_argument(
        "--remote-job-dir",
        type=str,
        help="Remote directory to store job files (required if --ssh-tunnel is set)",
    )
    parser.add_argument(
        "--identity",
        type=str,
        default=None,
        help="Path to SSH private key for authentication (optional)",
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
        help="Additional arguments for the training script (space-separated)",
    )
    parser.add_argument(
        "--container-image",
        type=str,
        default=None,
        help="Container image path (optional)",
    )
    parser.add_argument(
        "--mount",
        type=str,
        action="append",
        default=[],
        help="Container mounts in format host:container (can be specified multiple times)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="megatron_bridge_training",
        help="Name for the experiment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed without submitting the job",
    )

    return parser.parse_args()


def main() -> None:
    """Launch training (pretrain/finetune) using NeMo-Run SlurmExecutor."""
    args = parse_args()

    # Validate SSH tunnel arguments
    if args.ssh_tunnel:
        if not all([args.host, args.user, args.remote_job_dir]):
            raise ValueError("--ssh-tunnel requires --host, --user, and --remote-job-dir to be specified")

    # Resolve script path
    script_path = SCRIPT_DIR / args.script
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    # Build arguments for the training script
    script_args = []
    if args.config_file:
        script_args.extend(["--config-file", args.config_file])

    if args.script_args:
        script_args.extend(args.script_args.split())

    # Create the training task
    task = run.Script(
        path=str(script_path),
        entrypoint="python",
        args=script_args,
    )

    # Configure tunnel (SSH for remote, Local if already on cluster)
    tunnel = None
    if args.ssh_tunnel:
        tunnel = run.SSHTunnel(
            host=args.host,
            user=args.user,
            job_dir=args.remote_job_dir,
            identity=args.identity,
        )
        logger.info(f"Using SSH tunnel to {args.user}@{args.host}")
    else:
        tunnel = run.LocalTunnel()
        logger.info("Using LocalTunnel (running on cluster)")

    # Create the Slurm executor
    executor = run.SlurmExecutor(
        account=args.account,
        partition=args.partition,
        nodes=args.nodes,
        ntasks_per_node=args.devices,
        gpus_per_node=args.devices,
        mem="0",
        exclusive=True,
        time=args.time,
        tunnel=tunnel,
    )

    # Configure container if specified
    if args.container_image:
        executor.container_image = args.container_image

    # Configure mounts if specified
    if args.mount:
        executor.container_mounts = args.mount

    # Set common environment variables
    executor.env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
    }

    # Run the experiment
    with run.Experiment(args.experiment_name) as exp:
        exp.add(task, executor=executor, name="training")
        exp.run(detach=True, dryrun=args.dry_run)

    if args.dry_run:
        logger.info("Dry run completed - no job was submitted")
    else:
        logger.info("Job submitted to Slurm!")
        logger.info("Use 'squeue' to check job status")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
