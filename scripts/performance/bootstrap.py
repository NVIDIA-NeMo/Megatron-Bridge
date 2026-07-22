#!/usr/bin/env python3
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

"""Apply recipe process settings before executing performance training."""

import os
import sys
from pathlib import Path

from argument_parser import parse_cli_args


ENTRYPOINT_PERFORMANCE = "run_script.py"
ENTRYPOINT_RECIPE = "run_recipe.py"


def _prepare_recipe_and_target(args, cli_overrides: list[str]):
    """Resolve the effective recipe environment and selected training script."""
    if args.use_recipes:
        from run_recipe import _prepare_recipe

        recipe = _prepare_recipe(args, cli_overrides, environment_only=True)
        return recipe, ENTRYPOINT_RECIPE

    from run_script import _prepare_perf_recipe

    recipe = _prepare_perf_recipe(args, cli_overrides)
    return recipe, ENTRYPOINT_PERFORMANCE


def _apply_recipe_environment(recipe) -> None:
    """Install recipe defaults while preserving explicit process values."""
    for name, value in recipe.env_vars.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Environment variable names must be non-empty strings.")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"Environment variable {name!r} must have a scalar value, got {type(value).__name__}.")
        os.environ.setdefault(name, str(value))


def _exec_training(target_name: str) -> None:
    """Replace the bootstrap process with the selected training entrypoint."""
    target_path = Path(__file__).resolve().parent / target_name
    os.execvpe(
        sys.executable,
        [sys.executable, str(target_path), *sys.argv[1:]],
        dict(os.environ),
    )


def main() -> None:
    """Prepare process settings, then execute one training entrypoint."""
    parser = parse_cli_args()
    args, cli_overrides = parser.parse_known_args()
    recipe, target_name = _prepare_recipe_and_target(args, cli_overrides)
    _apply_recipe_environment(recipe)
    _exec_training(target_name)


if __name__ == "__main__":
    main()
