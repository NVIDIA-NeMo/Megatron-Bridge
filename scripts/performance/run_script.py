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

"""Run flat performance training after process environment bootstrap."""

import logging
import os
import re
import sys

from argument_parser import parse_cli_args
from utils.utils import get_perf_recipe_by_name


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


def _apply_perf_recipe_overrides(recipe, cli_overrides: list[str], args):
    """Apply Hydra and argparse overrides to a flat performance recipe."""
    from utils.overrides import _apply_flat_cli_environment_compatibility, set_cli_overrides, set_user_overrides
    from utils.utils import explicit_environment_override_names

    base_env_vars = dict(recipe.env_vars)
    base_dispatcher_backend = getattr(recipe.model, "moe_flex_dispatcher_backend", None)
    comm_overlap = getattr(recipe, "comm_overlap", None)
    base_moe_a2a_overlap = bool(
        comm_overlap is not None and getattr(comm_overlap, "overlap_moe_expert_parallel_comm", False)
    )
    recipe = set_cli_overrides(recipe, cli_overrides)
    protected_env_names = explicit_environment_override_names(cli_overrides, base_env_vars, recipe.env_vars)
    hydra_env_vars = {name: recipe.env_vars[name] for name in protected_env_names if name in recipe.env_vars}
    recipe = set_user_overrides(recipe, args)
    # Argparse normally has higher precedence than Hydra in the flat runner,
    # but explicit Hydra env selections remain final. Restore changed, added,
    # and removed env keys after argparse features such as --deterministic.
    for name in protected_env_names:
        if name in hydra_env_vars:
            recipe.env_vars[name] = hydra_env_vars[name]
        else:
            recipe.env_vars.pop(name, None)
    return _apply_flat_cli_environment_compatibility(
        recipe,
        args,
        base_dispatcher_backend=base_dispatcher_backend,
        base_moe_a2a_overlap=base_moe_a2a_overlap,
        protected_env_names=protected_env_names,
    )


def _prepare_perf_recipe(args, cli_overrides: list[str]):
    """Build a flat performance recipe with all user overrides applied."""
    recipe = get_perf_recipe_by_name(
        model_recipe_name=args.model_recipe_name,
        task=args.task,
        num_gpus=args.num_gpus,
        gpu=args.gpu,
        precision=args.compute_dtype,
        config_variant=getattr(args, "config_variant", None),
    )
    return _apply_perf_recipe_overrides(recipe, cli_overrides, args)


def _run_training(args, cli_overrides: list[str]) -> None:
    """Import the training stack after env bootstrap and run the workload."""
    from megatron.bridge.diffusion.models.wan.wan_step import WanForwardStep
    from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step as qwen3_vl_forward_step
    from megatron.bridge.training.config import runtime_config_update
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step

    recipe = _prepare_perf_recipe(args, cli_overrides)

    if args.dump_env:
        _dump_env_rank0()

    # Preserve BF16 Adam precision-aware behavior from the previous script path. Parallelism-dependent
    # optimizer-step overlap is encoded directly in the flat perf recipes.
    if args.compute_dtype == "bf16" and recipe.optimizer.optimizer == "adam":
        recipe.optimizer.use_precision_aware_optimizer = True

    if args.dryrun:
        save_path = args.save_config_filepath or "ConfigContainer.yaml"
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" not in os.environ:
            os.environ["WORLD_SIZE"] = str(args.num_gpus)
        if "RANK" not in os.environ and "SLURM_PROCID" not in os.environ:
            os.environ["RANK"] = "0"
        runtime_config_update(recipe)
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


def main() -> None:
    """Parse the final training arguments and run the workload once."""
    parser = parse_cli_args()
    args, cli_overrides = parser.parse_known_args()
    _run_training(args, cli_overrides)


if __name__ == "__main__":
    main()
