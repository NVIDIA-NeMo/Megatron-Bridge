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

"""Dependency-free discovery of exact flat performance recipes."""

from __future__ import annotations

import argparse
import ast
import functools
import re
from dataclasses import dataclass
from pathlib import Path


PERFORMANCE_RECIPE_PATTERN = re.compile(
    r"_(?P<num_gpus>[1-9][0-9]*)gpu_"
    r"(?P<hardware>[a-z0-9]+)_"
    r"(?:bf16|fp8cs|fp8mx|fp8sc|nvfp4)"
    r"(?:_[a-z0-9_]+)?_config$"
)

PERFORMANCE_RECIPE_ROOT = Path(__file__).resolve().parents[2] / "src" / "megatron" / "bridge" / "perf_recipes"

# These two names are also exported as functional library recipes. Preserve
# their existing library behavior until unified performance finetuning lands.
LIBRARY_RECIPE_PRECEDENCE_COLLISIONS = frozenset(
    {
        "llama3_70b_peft_8gpu_h100_bf16_config",
        "llama3_70b_sft_32gpu_h100_bf16_config",
    }
)


@dataclass(frozen=True)
class PerformanceRecipeMetadata:
    """Dimensions encoded in an exact performance recipe name."""

    recipe_name: str
    num_gpus: int
    hardware: str


def performance_recipe_metadata(recipe_name: str) -> PerformanceRecipeMetadata:
    """Parse allocation metadata from an exact flat recipe name."""
    match = PERFORMANCE_RECIPE_PATTERN.search(recipe_name)
    if match is None:
        raise ValueError(
            f"Performance recipe '{recipe_name}' is not a canonical flat recipe name; expected "
            "..._<N>gpu_<hardware>_<precision>_config."
        )

    return PerformanceRecipeMetadata(
        recipe_name=recipe_name,
        num_gpus=int(match.group("num_gpus")),
        hardware=match.group("hardware"),
    )


@functools.lru_cache(maxsize=1)
def performance_recipe_names() -> frozenset[str]:
    """Return exact names exported anywhere in the flat performance package.

    The source-only index keeps Slurm submission lightweight and avoids
    importing the GPU training stack on a login node.
    """
    recipe_names: set[str] = set()
    for init_path in PERFORMANCE_RECIPE_ROOT.rglob("__init__.py"):
        module = ast.parse(init_path.read_text(), filename=str(init_path))
        for node in ast.walk(module):
            if isinstance(node, ast.alias):
                candidate = node.asname or node.name
                if candidate.endswith("_config"):
                    recipe_names.add(candidate)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith("_config"):
                recipe_names.add(node.value)
    return frozenset(recipe_names)


def resolved_performance_recipe_metadata(recipe_name: str) -> PerformanceRecipeMetadata | None:
    """Return encoded metadata when an exact exported performance recipe is selected."""
    if recipe_name in LIBRARY_RECIPE_PRECEDENCE_COLLISIONS or recipe_name not in performance_recipe_names():
        return None
    return performance_recipe_metadata(recipe_name)


def selected_performance_recipe(argv: list[str]) -> PerformanceRecipeMetadata | None:
    """Return metadata when forwarded runner arguments select an exact performance recipe."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--recipe")
    args, _ = parser.parse_known_args(argv)
    if args.recipe is None:
        return None
    return resolved_performance_recipe_metadata(args.recipe)
