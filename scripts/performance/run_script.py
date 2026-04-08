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
import re
import sys

import torch
from argument_parser import parse_cli_args
from utils.overrides import set_cli_overrides, set_post_overrides, set_user_overrides
from utils.utils import get_perf_optimized_recipe

from megatron.bridge.diffusion.models.wan.wan_step import WanForwardStep
from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step as qwen3_vl_forward_step
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step


logger = logging.getLogger(__name__)
SENSITIVE_ENV_VAR_PATTERN = re.compile(
    r"(^|_)(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|SECRET_KEY|PRIVATE_KEY|AUTHORIZATION)(_|$)",
    re.IGNORECASE,
)


def _dump_env_rank0() -> None:
    """Capture the container environment to /nemo_run/env_<SLURM_JOB_ID>.log on rank 0.

    The file lands alongside log*.out and configs/ inside the per-run nemo_run
    directory for easy post-run debugging.
    """
    if os.environ.get("SLURM_JOB_ID") is None:
        return
    if int(os.environ.get("SLURM_PROCID", "-1")) != 0:
        return
    job_id = os.environ["SLURM_JOB_ID"]
    env_path = f"/nemo_run/env_{job_id}.log"
    try:
        fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            for k, v in sorted(os.environ.items()):
                if SENSITIVE_ENV_VAR_PATTERN.search(k):
                    f.write(f"{k}=[REDACTED]\n")
                else:
                    safe_v = v.replace("\r", "\\r").replace("\n", "\\n")
                    f.write(f"{k}={safe_v}\n")
        logger.info(f"Environment dump written to {env_path} (mode 600)")
    except OSError as e:
        logger.warning(f"Failed to write environment dump to {env_path}: {e}")


def main():
    """Main function to run the pretraining/finetuning script."""
    # Parse known args and treat any unknown args as Hydra-style config overrides.
    # `argparse.parse_known_args()` returns the unknown args as a `list[str]`.
    parser = parse_cli_args()
    args, cli_overrides = parser.parse_known_args()

    if args.dump_env:
        _dump_env_rank0()

    recipe = get_perf_optimized_recipe(
        model_family_name=args.model_family_name,
        model_recipe_name=args.model_recipe_name,
        train_task=args.task,
        gpu=args.gpu,
        compute_dtype=args.compute_dtype,
        mock=args.data == "mock",
        config_variant=args.config_variant,
        optimizer_type=getattr(args, "optimizer_type", None),
    )

    recipe = set_cli_overrides(recipe, cli_overrides)
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
        config_variant=args.config_variant,
    )

    if args.dryrun:
        save_path = args.save_config_filepath or "ConfigContainer.yaml"
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        recipe.to_yaml(save_path)
        logger.info(f"ConfigContainer saved to: {os.path.abspath(save_path)}")
        recipe.print_yaml()
        sys.exit(0)

    # Select forward step function based on the model family name.
    if args.domain == "vlm":
        forward_step_func = vlm_forward_step
    elif args.domain == "qwen3vl":
        forward_step_func = qwen3_vl_forward_step
    elif args.domain == "diffusion":
        forward_step_func = WanForwardStep(mode=args.task)
    else:
        forward_step_func = forward_step

    pretrain(config=recipe, forward_step_func=forward_step_func)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
