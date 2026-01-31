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
Launch Training with NeMo-Run

Generic launcher for training scripts. Supports local execution and Slurm clusters.

Prerequisites: Install nemo-run

Usage:
    # Test locally (single node)
    python launch_with_nemo_run.py \
        --local \
        --script run_recipe.py \
        --recipe llama32_1b_pretrain_config \
        --devices 2

    # Launch on Slurm from the cluster (LocalTunnel)
    python launch_with_nemo_run.py \
        --script run_recipe.py \
        --recipe llama32_1b_pretrain_config \
        --nodes 2 \
        --partition gpu \
        --account my_account

    # Launch on Slurm from your local machine (SSHTunnel)
    python launch_with_nemo_run.py \
        --script run_recipe.py \
        --recipe llama32_1b_finetune_config \
        --nodes 1 \
        --partition gpu \
        --account my_account \
        --ssh-tunnel \
        --host my-cluster.example.com \
        --user myusername \
        --remote-job-dir /home/myusername/nemo-runs

    # With CLI overrides
    python launch_with_nemo_run.py \
        --script run_recipe.py \
        --recipe gemma3_1b_pretrain_config \
        --nodes 1 \
        --partition gpu \
        --account my_account \
        train.train_iters=5000 \
        optimizer.lr=0.0002

    # With containers (uses PatternPackager by default)
    python launch_with_nemo_run.py \
        --script run_recipe.py \
        --recipe qwen3_8b_pretrain_config \
        --nodes 1 \
        --partition gpu \
        --account my_account \
        --container-image /path/to/container.sqsh \
        --mount /data:/data

    # With custom packager (git archive)
    python launch_with_nemo_run.py \
        --script run_recipe.py \
        --recipe llama3_8b_pretrain_config \
        --nodes 2 \
        --partition gpu \
        --account my_account \
        --container-image /path/to/container.sqsh \
        --packager git

    # With environment variables (HF token, W&B key, etc.)
    python launch_with_nemo_run.py \
        --script /opt/Megatron-Bridge/scripts/training/run_recipe.py \
        --recipe llama32_1b_pretrain_config \
        --nodes 1 \
        --partition gpu \
        --account my_account \
        --container-image /path/to/container.sqsh \
        --mount /path/to/Megatron-Bridge:/opt/Megatron-Bridge \
        --env HF_TOKEN=your_token \
        --env WANDB_API_KEY=your_key

    # With fault-tolerant launcher
    python launch_with_nemo_run.py \
        --script run_recipe.py \
        --recipe llama32_1b_pretrain_config \
        --launcher ft \
        --nodes 2 \
        --partition gpu \
        --account my_account

    # Wait for completion and tail logs
    python launch_with_nemo_run.py \
        --script run_recipe.py \
        --recipe llama32_1b_pretrain_config \
        --nodes 1 \
        --partition gpu \
        --account my_account \
        --no-detach \
        --tail-logs

Note:
- Use --local for single-node testing with LocalExecutor
- Use --ssh-tunnel when launching to Slurm from your local machine
- Omit --ssh-tunnel when already on the Slurm cluster (uses LocalTunnel)
- By default, jobs are submitted and detached (use --no-detach --tail-logs to monitor)
- With containers, scripts are auto-packaged using PatternPackager (or use --packager git)
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
        description="Launch training with NeMo-Run (local or Slurm)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally with LocalExecutor (single node). Omit for Slurm execution.",
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Training script to run (e.g., run_recipe.py, pretrain_vlm.py, finetune_vlm.py)",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        required=True,
        help="Recipe name (e.g., llama32_1b_pretrain_config)",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default="torchrun",
        choices=["torchrun", "ft", "default"],
        help="Launcher to use: 'torchrun', 'ft' (fault-tolerant), or 'default' (no launcher)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="GPUs per node. Required for --local. For Slurm, omit if cluster auto-allocates whole nodes.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to use (Slurm only, ignored for --local)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="Slurm partition name (required for Slurm execution)",
    )
    parser.add_argument(
        "--account",
        type=str,
        help="Slurm account name (required for Slurm execution)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="04:00:00",
        help="Job time limit",
    )
    parser.add_argument(
        "--gres",
        type=str,
        default=None,
        help="Slurm GRES (e.g., 'gpu:8').",
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
        help="Container image path (Slurm only)",
    )
    parser.add_argument(
        "--mount",
        type=str,
        action="append",
        default=[],
        help="Container mounts in format host:container (can be specified multiple times)",
    )
    parser.add_argument(
        "--packager",
        type=str,
        default="none",
        choices=["pattern", "git", "none"],
        help="Code packaging method: 'none' (passthrough, use mounted/accessible code), "
        "'pattern' (package *.py files), or 'git' (git archive).",
    )
    parser.add_argument(
        "--env",
        type=str,
        action="append",
        default=[],
        help="Environment variables in format KEY=VALUE (can be specified multiple times)",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach from the experiment after submission (use --no-detach to wait)",
    )
    parser.add_argument(
        "--tail-logs",
        action="store_true",
        help="Tail logs after submission (only works with --no-detach)",
    )

    args, forwarded_args = parser.parse_known_args()
    return args, forwarded_args


def main() -> None:
    """Launch training using NeMo-Run."""
    args, forwarded_args = parse_args()

    # Validate arguments based on execution mode
    if args.local:
        # Local execution - SSH tunnel args are not used
        if args.ssh_tunnel:
            raise ValueError("--ssh-tunnel cannot be used with --local")
        if args.devices is None:
            raise ValueError("--devices is required for --local execution")
    else:
        # Slurm execution - require partition and account
        if not args.partition or not args.account:
            raise ValueError("--partition and --account are required for Slurm execution (omit --local)")

        if args.ssh_tunnel:
            if not all([args.host, args.user, args.remote_job_dir]):
                raise ValueError("--ssh-tunnel requires --host, --user, and --remote-job-dir to be specified")

    # Validate script path (skip validation for absolute paths, assuming they're container paths)
    if Path(args.script).is_absolute():
        # Absolute path - assume it's a container path or cluster path
        script_path = Path(args.script)
        task_script_path = str(script_path)
        logger.info(f"Using absolute script path (container/cluster): {task_script_path}")
    else:
        # Relative path - resolve from SCRIPT_DIR and validate
        script_path = SCRIPT_DIR / args.script
        if not script_path.exists():
            raise FileNotFoundError(f"Training script not found: {script_path}")

    script_args = ["--recipe", args.recipe]
    if forwarded_args:
        script_args.extend(forwarded_args)

    # Determine packager
    if args.packager == "pattern":
        packager = run.PatternPackager(include_pattern="*.py", relative_path=str(SCRIPT_DIR))
        logger.info("Using PatternPackager")
        # For pattern packager, use relative path
        if not Path(args.script).is_absolute():
            task_script_path = args.script
    elif args.packager == "git":
        packager = run.GitArchivePackager(subpath="scripts/training")
        logger.info("Using GitArchivePackager")
        # For git packager, use relative path
        if not Path(args.script).is_absolute():
            task_script_path = args.script
    else:  # none
        packager = run.Packager()
        logger.info("Using passthrough packager (no packaging)")

    task = run.Script(
        path=task_script_path,
        entrypoint="python",
        args=script_args,
    )

    # Parse environment variables
    env_vars = {}
    for env_str in args.env:
        if "=" not in env_str:
            raise ValueError(f"Invalid env format: {env_str}. Expected KEY=VALUE")
        key, value = env_str.split("=", 1)
        env_vars[key] = value

    if env_vars:
        logger.info(f"Setting environment variables: {list(env_vars.keys())}")

    launcher = None
    if args.launcher == "torchrun":
        launcher = "torchrun"
    elif args.launcher == "ft":
        launcher = "ft"
        logger.debug("Using fault-tolerant launcher")
    elif args.launcher == "default":
        launcher = None

    if args.local:
        logger.debug("Using LocalExecutor")
        executor = run.LocalExecutor(
            ntasks_per_node=args.devices,
            launcher=launcher,
        )
        if env_vars:
            executor.env_vars = env_vars
    else:
        # Configure tunnel (SSH for remote, Local if already on cluster)
        tunnel = None
        if args.ssh_tunnel:
            tunnel = run.SSHTunnel(
                host=args.host,
                user=args.user,
                job_dir=args.remote_job_dir,
                identity=args.identity,
            )
            logger.debug(f"Using SSH tunnel to {args.user}@{args.host}")
        else:
            tunnel = run.LocalTunnel()
            logger.debug("Using LocalTunnel (running on cluster)")

        # Create the Slurm executor
        executor_kwargs = {
            "account": args.account,
            "partition": args.partition,
            "nodes": args.nodes,
            "mem": "0",
            "exclusive": True,
            "time": args.time,
            "tunnel": tunnel,
            "packager": packager,
        }

        # Add devices only if specified
        if args.devices is not None:
            executor_kwargs["ntasks_per_node"] = args.devices
            executor_kwargs["gpus_per_node"] = args.devices

        # Add gres only if explicitly specified
        if args.gres:
            executor_kwargs["gres"] = args.gres

        executor = run.SlurmExecutor(**executor_kwargs)

        # Configure container if specified
        if args.container_image:
            executor.container_image = args.container_image

        # Configure mounts if specified
        if args.mount:
            executor.container_mounts = args.mount

        # Set environment variables
        if env_vars:
            executor.env_vars = env_vars

    # Run the experiment
    with run.Experiment(args.experiment_name) as exp:
        exp.add(task, executor=executor, name="training")

        if args.dry_run:
            exp.dryrun()
        else:
            exp.run(detach=args.detach, tail_logs=args.tail_logs)

            if args.detach:
                if args.local:
                    logger.info("Job started locally!")
                else:
                    logger.info("Job submitted to Slurm!")
                    logger.info("Use 'squeue' to check job status")
            else:
                logger.info("Job completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
