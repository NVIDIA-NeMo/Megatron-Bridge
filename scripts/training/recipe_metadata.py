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

"""Lightweight recipe metadata, selection, and forward-step resolution."""

from __future__ import annotations

import argparse
import ast
import functools
import re
from dataclasses import dataclass
from pathlib import Path


BENCHMARK_RECIPE_FAMILY_PREFIXES = (
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

BENCHMARK_RECIPE_PATTERN = re.compile(
    r"_(?P<task>pretrain|sft|peft)_(?P<num_gpus>[1-9][0-9]*)gpu_"
    r"(?P<hardware>[a-z0-9]+)_"
    r"(?P<precision>bf16|fp8cs|fp8mx|fp8sc|nvfp4)"
    r"(?:_[a-z0-9_]+)?_config$"
)

BENCHMARK_RECIPE_ROOT = Path(__file__).resolve().parents[2] / "src" / "megatron" / "bridge" / "perf_recipes"

# These names are exported by both recipe packages today. Bare recipe lookup
# selects the benchmark definition; each library workload remains
# available through its generic library alias.
BENCHMARK_RECIPE_PRECEDENCE_COLLISIONS = frozenset(
    {
        "deepseek_v3_pretrain_1024gpu_h100_bf16_config",
        "gpt_oss_120b_pretrain_64gpu_h100_bf16_config",
        "llama3_70b_peft_8gpu_h100_bf16_config",
        "llama3_70b_sft_32gpu_h100_bf16_config",
        "qwen3_235b_a22b_pretrain_256gpu_h100_bf16_config",
        "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
    }
)

LIBRARY_RECIPE_PRECEDENCE_COLLISIONS: frozenset[str] = frozenset()

PUBLIC_MODES = frozenset({"pretrain", "sft", "lora", "dora"})
TEXT_FORWARD_STEPS = frozenset({"gpt_step", "llm_step"})

# Put specific multimodal families before the text default. This registry is
# source-agnostic: library and benchmark recipes with the same identity use
# the same forward step.
RECIPE_FORWARD_STEP_PREFIXES = (
    ("nemotron_nano_v2_vl_", "llava_step"),
    ("nemotron_omni_", "nemotron_omni_step"),
    ("qwen2_audio_", "audio_lm_step"),
    ("qwen3_omni_", "qwen3_omni_step"),
    ("qwen3_vl_", "qwen3_vl_step"),
    ("qwen35_vl_", "qwen3_vl_step"),
    ("qwen36_vl_", "qwen3_vl_step"),
    ("step37_", "step37_flickr8k_step"),
    ("gemma3_vl_", "vlm_step"),
    ("gemma4_vl_", "vlm_step"),
    ("glm_45v_", "vlm_step"),
    ("kimi_k25_vl_", "vlm_step"),
    ("ministral3_", "vlm_step"),
    ("qwen25_vl_", "vlm_step"),
    ("flux_", "flux_step"),
    ("wan_", "wan_step"),
)


@dataclass(frozen=True)
class BenchmarkRecipeMetadata:
    """Canonical dimensions encoded in a flat benchmark recipe name."""

    recipe_name: str
    num_gpus: int
    family: str
    hardware: str
    precision: str
    task: str


def benchmark_recipe_family(recipe_name: str) -> str:
    """Map an exported benchmark name to the family package that owns it."""
    for prefix, family in BENCHMARK_RECIPE_FAMILY_PREFIXES:
        if recipe_name.startswith(prefix):
            return family
    raise ValueError(f"Benchmark recipe '{recipe_name}' has no registered family prefix.")


def benchmark_recipe_metadata(recipe_name: str) -> BenchmarkRecipeMetadata:
    """Parse canonical allocation and precision metadata from a benchmark name."""
    match = BENCHMARK_RECIPE_PATTERN.search(recipe_name)
    if match is None:
        raise ValueError(
            f"Benchmark recipe '{recipe_name}' is not a canonical flat recipe name; expected "
            "..._<N>gpu_<hardware>_<precision>_config."
        )

    return BenchmarkRecipeMetadata(
        recipe_name=recipe_name,
        num_gpus=int(match.group("num_gpus")),
        family=benchmark_recipe_family(recipe_name),
        hardware=match.group("hardware"),
        precision=match.group("precision"),
        task=match.group("task"),
    )


@functools.lru_cache(maxsize=None)
def benchmark_recipe_names(family: str | None = None) -> frozenset[str]:
    """Return exact recipe names exported by the flat benchmark package.

    This source-only index keeps the Slurm submission path lightweight: it can
    identify benchmark recipes without importing the GPU training stack on a
    login node.
    """
    recipe_names: set[str] = set()
    recipe_root = BENCHMARK_RECIPE_ROOT / family if family is not None else BENCHMARK_RECIPE_ROOT
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


def available_benchmark_recipe_metadata(recipe_name: str) -> BenchmarkRecipeMetadata | None:
    """Return metadata only when the exact name is exported as a benchmark recipe."""
    try:
        metadata = benchmark_recipe_metadata(recipe_name)
    except ValueError:
        return None
    if recipe_name not in benchmark_recipe_names(metadata.family):
        return None
    return metadata


def resolved_benchmark_recipe_metadata(recipe_name: str) -> BenchmarkRecipeMetadata | None:
    """Return benchmark metadata after applying unified lookup precedence."""
    if recipe_name in LIBRARY_RECIPE_PRECEDENCE_COLLISIONS:
        return None
    return available_benchmark_recipe_metadata(recipe_name)


def recipe_step(recipe_name: str) -> str:
    """Return the default forward-step registry name for any recipe."""
    for prefix, step_name in RECIPE_FORWARD_STEP_PREFIXES:
        if recipe_name.startswith(prefix):
            return step_name
    return "llm_step"


def recipe_steps_match(requested_step: str, expected_step: str) -> bool:
    """Return whether two registry names select compatible forward steps."""
    requested_step = requested_step.lower()
    expected_step = expected_step.lower()
    return requested_step == expected_step or {requested_step, expected_step} <= TEXT_FORWARD_STEPS


def recipe_task(mode: str) -> str:
    """Map a public training mode to the task encoded in a recipe name."""
    return "peft" if mode in {"lora", "dora"} else mode


def infer_recipe_mode(recipe_name: str) -> str | None:
    """Infer the public training mode encoded in a complete recipe name."""
    normalized_name = f"_{recipe_name.lower().strip('_')}_"
    if "_pretrain_" in normalized_name:
        return "pretrain"
    if "_sft_" in normalized_name or "_finetune_" in normalized_name:
        return "sft"
    if "_dora_" in normalized_name:
        return "dora"
    if "_peft_" in normalized_name or "_lora_" in normalized_name:
        return "lora"
    return None


def validate_benchmark_recipe_scope(
    metadata: BenchmarkRecipeMetadata,
    *,
    mode: str,
    step_func: str | None = None,
    dataset: str | None = None,
) -> None:
    """Validate task, forward step, and dataset selection for a benchmark recipe."""
    if mode not in PUBLIC_MODES:
        choices = ", ".join(sorted(PUBLIC_MODES))
        raise ValueError(f"Unsupported benchmark mode '{mode}'; choose from: {choices}.")

    requested_task = recipe_task(mode)
    if requested_task != metadata.task:
        raise ValueError(f"Mode '{mode}' is incompatible with benchmark task '{metadata.task}'.")
    if metadata.task == "peft" and mode != "lora":
        raise ValueError("Benchmark PEFT recipes are fixed LoRA configs; omit --mode or pass --mode lora.")

    expected_step = recipe_step(metadata.recipe_name)
    if step_func is not None and not recipe_steps_match(step_func, expected_step):
        raise ValueError(
            f"Benchmark recipe '{metadata.recipe_name}' uses the canonical {expected_step} forward step; "
            f"omit --step-func or pass {expected_step}."
        )
    if dataset is not None:
        raise ValueError(
            "Benchmark recipes own their canonical dataset; omit --dataset or continue using scripts/performance "
            "for non-canonical benchmark data."
        )


def validate_selected_benchmark_recipe(argv: list[str], metadata: BenchmarkRecipeMetadata) -> None:
    """Validate a selected benchmark recipe before submitting its Slurm job."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--recipe")
    parser.add_argument("--mode")
    parser.add_argument("--step-func", "--step_func", dest="step_func")
    parser.add_argument("--dataset")
    args, _ = parser.parse_known_args(argv)
    mode = args.mode or infer_recipe_mode(args.recipe or "")
    if mode is None:
        raise ValueError("Unable to infer training mode for the selected benchmark recipe; pass --mode.")
    validate_benchmark_recipe_scope(
        metadata,
        mode=mode,
        step_func=args.step_func,
        dataset=args.dataset,
    )


def selected_benchmark_recipe(argv: list[str]) -> BenchmarkRecipeMetadata | None:
    """Return metadata when forwarded runner arguments name a benchmark recipe."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--recipe")
    args, _ = parser.parse_known_args(argv)
    if args.recipe is None:
        return None
    return resolved_benchmark_recipe_metadata(args.recipe)
