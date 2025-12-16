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

import logging
import os
import subprocess

import torch
from argument_parser import parse_cli_args
from utils.overrides import set_post_overrides, set_user_overrides
from utils.utils import get_perf_optimized_recipe

from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


logger = logging.getLogger(__name__)


def _maybe_print_numa_debug() -> None:
    """Opt-in debug to validate CPU/memory pinning from inside the launched rank process.

    Enable via: --custom_env_vars=MBRIDGE_NUMA_DEBUG=1
    """
    if os.environ.get("MBRIDGE_NUMA_DEBUG", "0") not in ("1", "true", "True", "yes", "YES"):
        return

    # Keep this lightweight and robust (works even if numactl is absent).
    env_keys = [
        "SLURM_JOB_ID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "CUDA_VISIBLE_DEVICES",
    ]
    env_dump = " ".join(f"{k}={os.environ.get(k, '')}" for k in env_keys)
    print(f"[MBRIDGE_NUMA_DEBUG] env: {env_dump}", flush=True)

    try:
        affinity = sorted(os.sched_getaffinity(0))
        print(f"[MBRIDGE_NUMA_DEBUG] sched_getaffinity: {affinity}", flush=True)
    except Exception as e:
        print(f"[MBRIDGE_NUMA_DEBUG] sched_getaffinity: <error: {e}>", flush=True)

    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith(("Cpus_allowed_list:", "Mems_allowed_list:")):
                    print(f"[MBRIDGE_NUMA_DEBUG] {line.strip()}", flush=True)
    except Exception as e:
        print(f"[MBRIDGE_NUMA_DEBUG] /proc/self/status: <error: {e}>", flush=True)

    # If numactl is available, show the effective policy from inside the process.
    try:
        subprocess.run(["numactl", "--show"], check=False)
    except Exception as e:
        print(f"[MBRIDGE_NUMA_DEBUG] numactl --show: <error: {e}>", flush=True)


def main():
    """Main function to run the pretraining/finetuning script."""
    _maybe_print_numa_debug()
    parser = parse_cli_args()
    args, _ = parser.parse_known_args()

    recipe = get_perf_optimized_recipe(
        model_family_name=args.model_family_name,
        model_recipe_name=args.model_recipe_name,
        train_task=args.task,
        gpu=args.gpu,
        compute_dtype=args.compute_dtype,
        mock=args.data == "mock",
    )

    recipe = set_user_overrides(recipe, args)

    recipe = set_post_overrides(
        recipe,
        args.model_family_name,
        args.model_recipe_name,
        args.gpu,
        args.num_gpus,
        args.compute_dtype,
        args.task,
        user_gbs=args.global_batch_size,
    )

    pretrain(config=recipe, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
