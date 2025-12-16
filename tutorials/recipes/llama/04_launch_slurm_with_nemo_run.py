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
    python 04_launch_slurm_with_nemo_run.py \
        --script 00_quickstart_pretrain.py \
        --nodes 2 \
        --partition gpu \
        --account my_account

    # From your local machine (uses SSHTunnel)
    python 04_launch_slurm_with_nemo_run.py \
        --script 00_quickstart_pretrain.py \
        --nodes 2 \
        --partition gpu \
        --account my_account \
        --ssh-tunnel \
        --host my-cluster.example.com \
        --user myusername \
        --remote-job-dir /home/myusername/nemo-runs

    # With custom SSH key
    python 04_launch_slurm_with_nemo_run.py \
        --script 00_quickstart_pretrain.py \
        --nodes 2 \
        --partition gpu \
        --account my_account \
        --ssh-tunnel \
        --host my-cluster.example.com \
        --user myusername \
        --remote-job-dir /home/myusername/nemo-runs \
        --identity ~/.ssh/id_rsa

    # Launch with custom config (pass arguments to training script)
    python 04_launch_slurm_with_nemo_run.py \
        --script 03_finetune_with_yaml.py \
        --nodes 1 \
        --partition gpu \
        --account my_account \
        --config-file conf/llama32_1b_finetune.yaml

    # Pass CLI overrides to training script
    python 04_launch_slurm_with_nemo_run.py \
        --script 02_pretrain_with_yaml.py \
        --nodes 2 \
        --partition gpu \
        --account my_account \
        train.train_iters=5000 \
        optimizer.lr=0.0002

    # With container and custom mounts
    python 04_launch_slurm_with_nemo_run.py \
        --script 00_quickstart_pretrain.py \
        --nodes 2 \
        --partition gpu \
        --account my_account \
        --container-image /path/to/container.sqsh \
        --mount /data:/data

    # Wait for job completion and tail logs
    python 04_launch_slurm_with_nemo_run.py \
        --script 00_quickstart_pretrain.py \
        --nodes 2 \
        --partition gpu \
        --account my_account \
        --no-detach \
        --tail-logs

Note:
- Use --ssh-tunnel when launching from your local machine
- Omit --ssh-tunnel when already on the Slurm cluster (uses LocalTunnel)
- By default, jobs are submitted and detached (--detach)
- Use --no-detach --tail-logs to wait and monitor job output
- Any unknown arguments are forwarded to the training script
- Adjust cluster-specific settings (account, partition, container paths)
"""

import argparse
import logging
from pathlib import Path

import nemo_run as run


logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch training (pretrain/finetune) on Slurm using NeMo-Run",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Training script to run (e.g., 00_quickstart_pretrain.py, 01_quickstart_finetune.py)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to use",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=8,
        help="GPUs per node",
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
        help="Job time limit",
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
        help="Path to SSH private key for authentication",
    )
    parser.add_argument(
        "--container-image",
        type=str,
        default=None,
        help="Container image path",
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
    parser.add_argument(
        "--detach",
        action="store_true",
        default=True,
        help="Detach from the experiment after submission",
    )
    parser.add_argument(
        "--tail-logs",
        action="store_true",
        help="Tail logs after submission (only works with --no-detach)",
    )

    # Use parse_known_args to capture forwarded arguments for the training script
    args, forwarded_args = parser.parse_known_args()
    return args, forwarded_args


def main() -> None:
    """Launch training (pretrain/finetune) using NeMo-Run SlurmExecutor."""
    args, forwarded_args = parse_args()

    # Validate SSH tunnel arguments
    if args.ssh_tunnel:
        if not all([args.host, args.user, args.remote_job_dir]):
            raise ValueError("--ssh-tunnel requires --host, --user, and --remote-job-dir to be specified")

    # Resolve script path
    script_path = SCRIPT_DIR / args.script
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    # Build arguments for the training script from forwarded args
    script_args = forwarded_args if forwarded_args else []

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

    # Run the experiment
    with run.Experiment(args.experiment_name) as exp:
        exp.add(task, executor=executor, name="training")

        if args.dry_run:
            exp.dryrun()
        else:
            exp.run(detach=args.detach, tail_logs=args.tail_logs)

            if args.detach:
                logger.info("Job submitted to Slurm!")
                logger.info("Use 'squeue' to check job status")
            else:
                logger.info("Job completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
