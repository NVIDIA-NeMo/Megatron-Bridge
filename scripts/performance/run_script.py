# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Bootstrap the recipe environment and run performance training."""

import logging
import os
import sys
from pathlib import Path

from argument_parser import parse_cli_args
from utils.utils import get_perf_recipe_by_name as get_perf_recipe_for_environment


logger = logging.getLogger(__name__)
ENV_BOOTSTRAP_MARKER = "_MB_PERF_ENV_BOOTSTRAPPED"
_TRAINING_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "training"


def _load_recipe_runner():
    """Import the shared runner after the perf environment bootstrap."""
    if str(_TRAINING_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_TRAINING_SCRIPT_DIR))

    import recipe_runner

    return recipe_runner


def _step_function_name(domain: str) -> str:
    """Map the perf domain selector to the shared forward-step registry name."""
    if domain == "vlm":
        return "vlm_step"
    if domain == "qwen3vl":
        return "qwen3_vl_step"
    if domain == "diffusion":
        return "wan_step"
    return "llm_step"


def _apply_perf_recipe_overrides(recipe, cli_overrides: list[str], args):
    """Apply the same CLI and argparse overrides in both self-exec passes."""
    from utils.overrides import apply_flat_cli_environment_compatibility, set_cli_overrides, set_user_overrides
    from utils.utils import explicit_environment_override_names

    base_env_vars = dict(recipe.env_vars)
    base_dispatcher_backend = getattr(recipe.model, "moe_flex_dispatcher_backend", None)
    comm_overlap = getattr(recipe, "comm_overlap", None)
    base_moe_a2a_overlap = bool(
        comm_overlap is not None and getattr(comm_overlap, "overlap_moe_expert_parallel_comm", False)
    )
    recipe = set_cli_overrides(recipe, cli_overrides)
    protected_env_names = explicit_environment_override_names(cli_overrides, base_env_vars, recipe.env_vars)
    recipe = set_user_overrides(recipe, args)
    return apply_flat_cli_environment_compatibility(
        recipe,
        args,
        base_dispatcher_backend=base_dispatcher_backend,
        base_moe_a2a_overlap=base_moe_a2a_overlap,
        protected_env_names=protected_env_names,
    )


def _apply_recipe_environment(recipe) -> None:
    """Install recipe environment defaults without importing the training stack."""
    for name, value in recipe.env_vars.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Environment variable names must be non-empty strings.")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"Environment variable {name!r} must have a scalar value, got {type(value).__name__}.")
        os.environ.setdefault(name, str(value))


def _bootstrap_recipe_environment(args, cli_overrides: list[str]) -> None:
    """Install recipe env and re-exec this script in a clean interpreter."""
    recipe = get_perf_recipe_for_environment(
        model_recipe_name=args.model_recipe_name,
        task=args.task,
        num_gpus=args.num_gpus,
        gpu=args.gpu,
        precision=args.compute_dtype,
        config_variant=args.config_variant,
    )
    recipe = _apply_perf_recipe_overrides(recipe, cli_overrides, args)
    _apply_recipe_environment(recipe)

    environment = dict(os.environ)
    # exec preserves the PID. Binding the marker to it prevents an inherited
    # or stale variable from skipping environment setup in a new process.
    environment[ENV_BOOTSTRAP_MARKER] = str(os.getpid())
    os.execvpe(sys.executable, [sys.executable, __file__, *sys.argv[1:]], environment)


def _run_training(args, cli_overrides: list[str]) -> None:
    """Import the training stack after env bootstrap and run the workload."""
    recipe_runner = _load_recipe_runner()
    recipe = recipe_runner.load_perf_recipe_by_name(
        model_recipe_name=args.model_recipe_name,
        task=args.task,
        num_gpus=args.num_gpus,
        gpu=args.gpu,
        precision=args.compute_dtype,
        config_variant=getattr(args, "config_variant", None),
    )

    recipe = _apply_perf_recipe_overrides(recipe, cli_overrides, args)
    forward_step_func = recipe_runner.load_forward_step(_step_function_name(args.domain), mode="pretrain")
    recipe_runner.run_config(
        config=recipe,
        mode="pretrain",
        step_func=forward_step_func,
        dryrun=args.dryrun,
        save_config_filepath=args.save_config_filepath,
        barrier_before_destroy=True,
        dryrun_num_gpus=args.num_gpus,
        dump_environment=args.dump_env,
    )


def main() -> None:
    """Resolve recipe env on the first pass and train in the self-exec process."""
    parser = parse_cli_args()
    args, cli_overrides = parser.parse_known_args()

    if os.environ.get(ENV_BOOTSTRAP_MARKER) != str(os.getpid()):
        _bootstrap_recipe_environment(args, cli_overrides)
    else:
        _run_training(args, cli_overrides)


if __name__ == "__main__":
    main()
