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

"""Translate exact performance recipe names for the unchanged runner."""

from __future__ import annotations

import argparse
import ast
import functools
import re
from dataclasses import dataclass
from pathlib import Path


PERFORMANCE_RECIPE_PATTERN = re.compile(
    r"^(?P<model_recipe_name>[a-z0-9_]+)_(?P<task>pretrain|sft|peft)_"
    r"(?P<num_gpus>[1-9][0-9]*)gpu_"
    r"(?P<hardware>[a-z0-9]+)_"
    r"(?P<precision>bf16|fp8cs|fp8mx|fp8sc|nvfp4)"
    r"(?:_(?P<config_variant>[a-z0-9_]+))?_config$"
)
PERFORMANCE_RECIPE_ROOT = Path(__file__).resolve().parents[2] / "src" / "megatron" / "bridge" / "perf_recipes"
PRECISION_ARGUMENTS = {
    "bf16": "bf16",
    "fp8cs": "fp8_cs",
    "fp8mx": "fp8_mx",
    "fp8sc": "fp8_sc",
    "nvfp4": "nvfp4",
}
DOMAIN_BY_FAMILY = {
    "qwen_vl": "qwen3vl",
    "wan": "diffusion",
}

# These exact names are also established library recipes. Keep their existing
# SFT/PEFT route while performance pretraining names use the compatibility path.
LIBRARY_RECIPE_PRECEDENCE_COLLISIONS = frozenset(
    {
        "llama3_70b_peft_8gpu_h100_bf16_config",
        "llama3_70b_sft_32gpu_h100_bf16_config",
    }
)

PERFORMANCE_SELECTION_OPTIONS = frozenset(
    {
        "--domain",
        "-m",
        "--model_family_name",
        "-mr",
        "--model_recipe_name",
        "-ng",
        "--num_gpus",
        "--task",
        "-g",
        "--gpu",
        "-c",
        "--compute_dtype",
        "-cv",
        "--config_variant",
        "--model",
        "--mode",
        "--dataset",
        "--step-func",
        "--step_func",
    }
)


@dataclass(frozen=True)
class PerformanceRecipeMetadata:
    """Selector values encoded by an exact performance recipe export."""

    recipe_name: str
    model_family_name: str
    model_recipe_name: str
    task: str
    num_gpus: int
    hardware: str
    compute_dtype: str
    config_variant: str | None
    domain: str


@functools.lru_cache(maxsize=1)
def performance_recipe_exports() -> dict[str, str]:
    """Map exact top-level performance exports to their family package.

    Reading the source keeps Slurm submission dependency-free and avoids
    importing the GPU training stack on a login node.
    """
    exports: dict[str, str] = {}
    for init_path in sorted(PERFORMANCE_RECIPE_ROOT.glob("*/__init__.py")):
        family = init_path.parent.name
        module = ast.parse(init_path.read_text(), filename=str(init_path))
        candidates: set[str] = set()
        for node in module.body:
            if isinstance(node, ast.ImportFrom):
                candidates.update(alias.asname or alias.name for alias in node.names)
            elif isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                candidates.add(node.name)

        for candidate in candidates:
            if not candidate.endswith("_config"):
                continue
            previous_family = exports.setdefault(candidate, family)
            if previous_family != family:
                raise ValueError(
                    f"Performance recipe '{candidate}' is exported by both '{previous_family}' and '{family}'."
                )
    return exports


def performance_recipe_metadata(recipe_name: str) -> PerformanceRecipeMetadata:
    """Parse an exact exported performance recipe into legacy selectors."""
    family = performance_recipe_exports().get(recipe_name)
    match = PERFORMANCE_RECIPE_PATTERN.fullmatch(recipe_name)
    if family is None or match is None:
        raise ValueError(f"No canonical flat performance recipe named '{recipe_name}' is exported.")

    return PerformanceRecipeMetadata(
        recipe_name=recipe_name,
        model_family_name=family,
        model_recipe_name=match.group("model_recipe_name"),
        task=match.group("task"),
        num_gpus=int(match.group("num_gpus")),
        hardware=match.group("hardware"),
        compute_dtype=PRECISION_ARGUMENTS[match.group("precision")],
        config_variant=match.group("config_variant"),
        domain=DOMAIN_BY_FAMILY.get(family, "llm"),
    )


def resolved_performance_recipe_metadata(recipe_name: str) -> PerformanceRecipeMetadata | None:
    """Return metadata when an exact performance recipe should use the adapter."""
    if recipe_name in LIBRARY_RECIPE_PRECEDENCE_COLLISIONS or recipe_name not in performance_recipe_exports():
        return None
    return performance_recipe_metadata(recipe_name)


def _selector_conflicts(arguments: list[str]) -> list[str]:
    """Return user options that would override exact-name selection."""
    return sorted(
        {
            argument.split("=", 1)[0]
            for argument in arguments
            if argument.split("=", 1)[0] in PERFORMANCE_SELECTION_OPTIONS
        }
    )


def resolve_performance_recipe_args(
    arguments: list[str],
) -> tuple[PerformanceRecipeMetadata | None, list[str]]:
    """Translate one exact performance recipe into the unchanged runner CLI."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--recipe", action="append")
    parsed, remaining = parser.parse_known_args(arguments)
    recipe_names = parsed.recipe or []
    performance_names = [name for name in recipe_names if resolved_performance_recipe_metadata(name) is not None]
    if not performance_names:
        return None, arguments
    if len(recipe_names) != 1:
        raise ValueError("Pass exactly one --recipe when selecting an exact performance recipe.")

    metadata = performance_recipe_metadata(performance_names[0])
    conflicts = _selector_conflicts(remaining)
    if conflicts:
        raise ValueError(
            f"Recipe '{metadata.recipe_name}' already selects the workload; omit: {', '.join(conflicts)}."
        )

    runner_args = [
        "--model_family_name",
        metadata.model_family_name,
        "--model_recipe_name",
        metadata.model_recipe_name,
        "--task",
        metadata.task,
        "--num_gpus",
        str(metadata.num_gpus),
        "--gpu",
        metadata.hardware,
        "--compute_dtype",
        metadata.compute_dtype,
        "--domain",
        metadata.domain,
    ]
    if metadata.config_variant is not None:
        runner_args.extend(["--config_variant", metadata.config_variant])
    return metadata, [*runner_args, *remaining]
