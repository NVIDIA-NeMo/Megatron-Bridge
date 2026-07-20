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
import ast
import functools
import re
from dataclasses import dataclass
from pathlib import Path


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
    r"_(?P<task>pretrain|sft|peft)_(?P<num_gpus>[1-9][0-9]*)gpu_"
    r"(?P<hardware>[a-z0-9]+)_"
    r"(?P<precision>bf16|fp8cs|fp8mx|fp8sc|nvfp4)"
    r"(?:_[a-z0-9_]+)?_config$"
)

PERFORMANCE_RECIPE_ROOT = Path(__file__).resolve().parents[2] / "src" / "megatron" / "bridge" / "perf_recipes"

# These names are exported by both recipe packages today. Bare recipe lookup
# selects the performance definition; each functional workload remains
# available through its generic library alias.
PERFORMANCE_RECIPE_PRECEDENCE_COLLISIONS = frozenset(
    {
        "deepseek_v3_pretrain_1024gpu_h100_bf16_config",
        "gpt_oss_120b_pretrain_64gpu_h100_bf16_config",
        "llama3_70b_peft_8gpu_h100_bf16_config",
        "llama3_70b_sft_32gpu_h100_bf16_config",
        "qwen3_235b_a22b_pretrain_256gpu_h100_bf16_config",
    }
)

LIBRARY_RECIPE_PRECEDENCE_COLLISIONS: frozenset[str] = frozenset()

PERFORMANCE_FORWARD_STEPS = {
    "qwen_vl": "qwen3_vl_step",
    "wan": "wan_step",
}

PERFORMANCE_PUBLIC_MODES = frozenset({"pretrain", "sft", "lora", "dora"})


@dataclass(frozen=True)
class PerformanceRecipeMetadata:
    """Canonical dimensions encoded in a flat performance recipe name."""

    num_gpus: int
    family: str
    hardware: str
    precision: str
    task: str


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

    return PerformanceRecipeMetadata(
        num_gpus=int(match.group("num_gpus")),
        family=performance_recipe_family(recipe_name),
        hardware=match.group("hardware"),
        precision=match.group("precision"),
        task=match.group("task"),
    )


@functools.lru_cache(maxsize=None)
def performance_recipe_names(family: str | None = None) -> frozenset[str]:
    """Return exact recipe names exported by the flat performance package.

    This source-only index keeps the Slurm submission path lightweight: it can
    identify performance recipes without importing the GPU training stack on a
    login node.
    """
    recipe_names: set[str] = set()
    recipe_root = PERFORMANCE_RECIPE_ROOT / family if family is not None else PERFORMANCE_RECIPE_ROOT
    for init_path in recipe_root.rglob("__init__.py"):
        module = ast.parse(init_path.read_text(), filename=str(init_path))
        for node in ast.walk(module):
            if isinstance(node, ast.alias):
                candidate = node.asname or node.name
                if candidate.endswith("_config"):
                    recipe_names.add(candidate)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith("_config"):
                recipe_names.add(node.value)
    return frozenset(recipe_names)


def available_performance_recipe_metadata(recipe_name: str) -> PerformanceRecipeMetadata | None:
    """Return metadata only when the exact name is exported as a performance recipe."""
    try:
        metadata = performance_recipe_metadata(recipe_name)
    except ValueError:
        return None
    if recipe_name not in performance_recipe_names(metadata.family):
        return None
    return metadata


def resolved_performance_recipe_metadata(recipe_name: str) -> PerformanceRecipeMetadata | None:
    """Return performance metadata after applying unified lookup precedence."""
    if recipe_name in LIBRARY_RECIPE_PRECEDENCE_COLLISIONS:
        return None
    return available_performance_recipe_metadata(recipe_name)


def performance_recipe_step(metadata: PerformanceRecipeMetadata) -> str:
    """Return the default forward-step registry name for a performance recipe."""
    return PERFORMANCE_FORWARD_STEPS.get(metadata.family, "gpt_step")


def validate_performance_recipe_scope(
    metadata: PerformanceRecipeMetadata,
    *,
    mode: str,
    step_func: str | None = None,
    dataset: str | None = None,
) -> None:
    """Validate task, forward step, and dataset selection for an exact performance recipe."""
    if mode not in PERFORMANCE_PUBLIC_MODES:
        choices = ", ".join(sorted(PERFORMANCE_PUBLIC_MODES))
        raise ValueError(f"Unsupported performance mode '{mode}'; choose from: {choices}.")

    requested_task = "peft" if mode in {"lora", "dora"} else mode
    if requested_task != metadata.task:
        raise ValueError(f"Mode '{mode}' is incompatible with performance task '{metadata.task}'.")
    if metadata.task == "peft" and mode != "lora":
        raise ValueError("Performance PEFT recipes are fixed LoRA configs; omit --mode or pass --mode lora.")

    expected_step = performance_recipe_step(metadata)
    if step_func is not None and step_func.lower() != expected_step:
        raise ValueError(
            f"Performance family '{metadata.family}' uses the canonical {expected_step} forward step; "
            f"omit --step-func or pass {expected_step}."
        )
    if dataset is not None:
        raise ValueError(
            "Performance recipes own their canonical dataset; omit --dataset or continue using scripts/performance "
            "for non-canonical benchmark data."
        )


def _infer_recipe_mode(recipe_name: str) -> str | None:
    """Infer the public training mode encoded in a complete recipe name."""
    normalized_name = f"_{recipe_name.lower().strip('_')}_"
    if "_pretrain_" in normalized_name:
        return "pretrain"
    if "_sft_" in normalized_name or "_finetune_" in normalized_name:
        return "sft"
    if any(marker in normalized_name for marker in ("_peft_", "_lora_", "_dora_")):
        return "lora"
    return None


def validate_selected_performance_recipe(argv: list[str], metadata: PerformanceRecipeMetadata) -> None:
    """Validate a selected performance recipe before submitting its Slurm job."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--recipe")
    parser.add_argument("--mode")
    parser.add_argument("--step-func", "--step_func", dest="step_func")
    parser.add_argument("--dataset")
    args, _ = parser.parse_known_args(argv)
    mode = args.mode or _infer_recipe_mode(args.recipe or "")
    if mode is None:
        raise ValueError("Unable to infer training mode for the selected performance recipe; pass --mode.")
    validate_performance_recipe_scope(
        metadata,
        mode=mode,
        step_func=args.step_func,
        dataset=args.dataset,
    )


def selected_performance_recipe(argv: list[str]) -> PerformanceRecipeMetadata | None:
    """Return metadata when forwarded runner arguments name a performance recipe."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--recipe")
    args, _ = parser.parse_known_args(argv)
    if args.recipe is None:
        return None
    return resolved_performance_recipe_metadata(args.recipe)
