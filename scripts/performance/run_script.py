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

import functools
import importlib
import logging
import os
import pkgutil
import re
import sys
from collections.abc import Callable
from typing import cast

import torch
from argument_parser import parse_cli_args
from utils.overrides import set_cli_overrides, set_user_overrides

from megatron.bridge.diffusion.models.wan.wan_step import WanForwardStep
from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step as qwen3_vl_forward_step
from megatron.bridge.training.config import ConfigContainer, runtime_config_update
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


@functools.lru_cache(maxsize=1)
def _perf_recipe_family_modules() -> tuple[str, ...]:
    """Return import paths for perf recipe family packages."""
    import megatron.bridge.perf_recipes as perf_recipes

    module_names = [
        f"{perf_recipes.__name__}.{module_info.name}"
        for module_info in pkgutil.iter_modules(perf_recipes.__path__)
        if module_info.ispkg and not module_info.name.startswith("_")
    ]
    return tuple(sorted(module_names))


def _find_perf_recipe(name: str) -> Callable[[], ConfigContainer] | None:
    """Find a flat perf recipe function exported by any perf recipe family package."""
    for module_name in _perf_recipe_family_modules():
        recipe_fn = getattr(importlib.import_module(module_name), name, None)
        if callable(recipe_fn):
            return cast(Callable[[], ConfigContainer], recipe_fn)
    return None


def _flat_recipe_variant_suffix(config_variant: str | None) -> str:
    """Return the suffix used in flat perf recipe function names."""
    return f"_{config_variant}" if config_variant and config_variant not in {"v1", "v2", "v3"} else ""


def get_perf_recipe_by_name(
    model_recipe_name: str,
    task: str,
    num_gpus: int,
    gpu: str,
    precision: str,
    config_variant: str | None = None,
) -> ConfigContainer:
    """Load a flat perf recipe from megatron.bridge.perf_recipes by convention name.

    Non-canonical ``config_variant`` values are appended to the function name.
    E.g. ``config_variant="large_scale"`` resolves to
    ``{model}_{task}_{N}gpu_{gpu}_{prec}_large_scale_config``.
    """
    precision_map = {
        "bf16": "bf16",
        "fp8_cs": "fp8cs",
        "fp8_mx": "fp8mx",
        "fp8_sc": "fp8sc",
        "nvfp4": "nvfp4",
    }
    prec = precision_map.get(precision.lower(), precision.lower().replace("_", ""))

    variant_suffix = _flat_recipe_variant_suffix(config_variant)
    name = f"{model_recipe_name}_{task}_{num_gpus}gpu_{gpu}_{prec}{variant_suffix}_config"

    recipe_fn = _find_perf_recipe(name)
    if recipe_fn is None:
        searched_modules = ", ".join(_perf_recipe_family_modules()) or "none"
        raise ValueError(f"No perf recipe {name!r} found in perf recipe packages: {searched_modules}.")
    return recipe_fn()


def main():
    """Main function to run the pretraining/finetuning script."""
    # Parse known args and treat any unknown args as Hydra-style config overrides.
    # `argparse.parse_known_args()` returns the unknown args as a `list[str]`.
    parser = parse_cli_args()
    args, cli_overrides = parser.parse_known_args()

    if args.dump_env:
        _dump_env_rank0()

    recipe = get_perf_recipe_by_name(
        model_recipe_name=args.model_recipe_name,
        task=args.task,
        num_gpus=args.num_gpus,
        gpu=args.gpu,
        precision=args.compute_dtype,
        config_variant=getattr(args, "config_variant", None),
    )

    recipe = set_cli_overrides(recipe, cli_overrides)
    recipe = set_user_overrides(recipe, args)

    # Preserve legacy BF16 Adam precision-aware behavior. Parallelism-dependent
    # optimizer-step overlap is encoded directly in the flat perf recipes.
    if args.compute_dtype == "bf16" and recipe.optimizer.optimizer == "adam":
        recipe.optimizer.use_precision_aware_optimizer = True

    # Set NCCL env vars for nccl_ub enabled via recipe config (not just CLI).
    if getattr(recipe.ddp, "nccl_ub", False):
        os.environ["NCCL_NVLS_ENABLE"] = "1"
        os.environ["NCCL_CTA_POLICY"] = "1"

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

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
