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

"""Resolve recipe-dependent environment variables, then exec the recipe runner.

This entrypoint runs inside the training container.  It keeps recipe imports off
the dependency-light login node while ensuring environment variables are present
before the training Python process imports Torch, Transformer Engine, or Megatron.
"""

import os
import sys
from pathlib import Path

from argument_parser import parse_cli_args
from perf_plugins import PerfEnvPlugin
from utils.utils import (
    _workload_base_config_from_recipe,
    add_library_recipe_environment_variables,
    apply_library_environment_overrides,
    explicit_environment_override_names,
    finalize_library_environment_overrides,
    get_library_recipe,
    get_perf_recipe_by_name,
)


class _EnvironmentExecutor:
    """Minimal executor adapter used by PerfEnvPlugin's environment helpers."""

    def __init__(self) -> None:
        self.env_vars = os.environ


def _process_library_hydra_overrides(recipe, cli_overrides: list[str]):
    """Apply library Hydra overrides while keeping Bridge imports rank-local."""
    from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

    return process_config_with_overrides(recipe, cli_overrides=cli_overrides)


def _apply_library_recipe_overrides(recipe, cli_overrides: list[str], args):
    """Mirror env-relevant user overrides before applying Hydra overrides."""
    recipe = apply_library_environment_overrides(recipe, args)
    if cli_overrides:
        recipe = _process_library_hydra_overrides(recipe, cli_overrides)
    return finalize_library_environment_overrides(recipe)


def _apply_perf_recipe_overrides(recipe, cli_overrides: list[str], args):
    """Apply the same perf overrides used by the final training entry point."""
    from utils.overrides import set_cli_overrides, set_user_overrides

    recipe = set_cli_overrides(recipe, cli_overrides)
    return set_user_overrides(recipe, args)


def main() -> None:
    """Apply the selected recipe's environment and replace this process with its runner."""
    parser = parse_cli_args()
    args, cli_overrides = parser.parse_known_args()

    if args.use_recipes:
        recipe = get_library_recipe(
            model_family_name=args.model_family_name,
            model_recipe_name=args.model_recipe_name,
            train_task=args.task,
            wandb_experiment_name=args.wandb_experiment_name,
        )
        base_env_vars = dict(recipe.env_vars)
        recipe = _apply_library_recipe_overrides(recipe, cli_overrides, args)
        protected_env_names = explicit_environment_override_names(cli_overrides, base_env_vars, recipe.env_vars)
        add_library_recipe_environment_variables(
            custom_env_vars=os.environ,
            config=recipe,
            gpu=args.gpu,
            protected_env_names=protected_env_names,
        )
        runner_name = "run_recipe.py"
    else:
        recipe = get_perf_recipe_by_name(
            model_recipe_name=args.model_recipe_name,
            task=args.task,
            num_gpus=args.num_gpus,
            gpu=args.gpu,
            precision=args.compute_dtype,
            config_variant=args.config_variant,
        )
        recipe = _apply_perf_recipe_overrides(recipe, cli_overrides, args)
        workload_base_config = _workload_base_config_from_recipe(recipe, num_gpus=args.num_gpus)

        plugin = PerfEnvPlugin(
            deterministic=args.deterministic,
        )
        plugin.setup_recipe_environment(
            None,
            _EnvironmentExecutor(),
            workload_base_config,
        )
        runner_name = "run_script.py"

    run_script = Path(__file__).with_name(runner_name)
    os.execvpe(sys.executable, [sys.executable, str(run_script), *sys.argv[1:]], dict(os.environ))


if __name__ == "__main__":
    main()
