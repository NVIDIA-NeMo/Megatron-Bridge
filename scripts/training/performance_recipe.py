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

"""Lightweight metadata parsing for canonical flat performance recipes."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass


PERFORMANCE_GPUS_PER_NODE = {
    "h100": 8,
    "b200": 8,
    "b300": 8,
    "gb200": 4,
    "gb300": 4,
    "vr200": 4,
    "r100": 1,
}

PERFORMANCE_RECIPE_FAMILY_PREFIXES = (
    ("qwen3_vl_", "qwen_vl"),
    ("qwen35_vl_", "qwen_vl"),
    ("deepseek_", "deepseek"),
    ("gpt_oss_", "gpt_oss"),
    ("nemotron", "nemotronh"),
    ("llama", "llama"),
    ("qwen", "qwen"),
    ("kimi_", "kimi"),
    ("glm5", "glm_moe_dsa"),
    ("wan_", "wan"),
)

PERFORMANCE_RECIPE_PATTERN = re.compile(
    r"_(?P<num_gpus>[1-9][0-9]*)gpu_"
    r"(?P<hardware>[a-z0-9]+)_"
    r"(?P<precision>bf16|fp8cs|fp8mx|fp8sc|nvfp4)"
    r"(?:_[a-z0-9_]+)?_config$"
)


@dataclass(frozen=True)
class PerformanceRecipeMetadata:
    """Canonical dimensions encoded in a flat performance recipe name."""

    num_gpus: int
    gpus_per_node: int
    family: str
    hardware: str
    precision: str


def performance_recipe_family(recipe_name: str) -> str:
    """Map an exported flat recipe name to the one family package that owns it."""
    for prefix, family in PERFORMANCE_RECIPE_FAMILY_PREFIXES:
        if recipe_name.startswith(prefix):
            return family
    raise ValueError(f"Performance recipe '{recipe_name}' has no registered family prefix.")


def performance_recipe_metadata(recipe_name: str) -> PerformanceRecipeMetadata:
    """Parse canonical allocation and precision metadata from a flat recipe name."""
    match = PERFORMANCE_RECIPE_PATTERN.search(recipe_name)
    if match is None:
        raise ValueError(
            f"Performance recipe '{recipe_name}' is not a canonical flat recipe name; expected "
            "..._<N>gpu_<hardware>_<precision>_config."
        )

    hardware = match.group("hardware")
    if hardware not in PERFORMANCE_GPUS_PER_NODE:
        choices = ", ".join(PERFORMANCE_GPUS_PER_NODE)
        raise ValueError(f"Performance recipe targets unsupported hardware '{hardware}'; choose from: {choices}.")

    num_gpus = int(match.group("num_gpus"))
    gpus_per_node = PERFORMANCE_GPUS_PER_NODE[hardware]
    if num_gpus % gpus_per_node != 0:
        raise ValueError(
            f"Performance recipe requests {num_gpus} GPUs, which is not divisible by the canonical "
            f"{gpus_per_node} GPUs per {hardware} node."
        )

    return PerformanceRecipeMetadata(
        num_gpus=num_gpus,
        gpus_per_node=gpus_per_node,
        family=performance_recipe_family(recipe_name),
        hardware=hardware,
        precision=match.group("precision"),
    )


def selected_performance_recipe(argv: list[str]) -> PerformanceRecipeMetadata | None:
    """Return metadata when forwarded runner arguments explicitly select performance."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--recipe-source", choices=["library", "performance"], default="library")
    parser.add_argument("--recipe")
    args, _ = parser.parse_known_args(argv)
    if args.recipe_source != "performance":
        return None
    if args.recipe is None:
        raise ValueError("--recipe-source performance requires an explicit --recipe name.")
    return performance_recipe_metadata(args.recipe)
