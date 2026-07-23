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

from __future__ import annotations

import functools
import importlib
import logging
import os
import pkgutil
import re
import select
import shlex
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Default timeout for interactive config variant selection (in seconds)
CONFIG_VARIANT_SELECTION_TIMEOUT = 15
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PERF_RECIPES_ROOT = _REPO_ROOT / "src" / "megatron" / "bridge" / "perf_recipes"
_PERF_RECIPE_DEF_PATTERN = re.compile(r"^def ([a-zA-Z0-9_]+_config)\(", re.MULTILINE)

_PRECISION_NAME_MAP = {
    "bf16": "bf16",
    "fp8_cs": "fp8cs",
    "fp8_mx": "fp8mx",
    "fp8_sc": "fp8sc",
    "nvfp4": "nvfp4",
}

# Most workloads use the largest matching suffix-less flat recipe by default.
# These overrides cover workloads whose default recipe is the smaller GPU-count
# recipe.
_DEFAULT_GPU_COUNT_OVERRIDES = {
    ("llama3_8b", "pretrain", "b200", "bf16", None): 8,
    ("llama3_8b", "pretrain", "b200", "fp8cs", None): 8,
    ("llama3_8b", "pretrain", "gb200", "bf16", None): 8,
    ("llama3_8b", "pretrain", "gb200", "fp8cs", None): 8,
    ("llama3_8b", "pretrain", "gb300", "bf16", None): 8,
    ("llama3_8b", "pretrain", "gb300", "fp8cs", None): 8,
    ("llama3_8b", "pretrain", "gb300", "fp8mx", None): 8,
    ("llama3_8b", "pretrain", "gb300", "nvfp4", None): 8,
    ("llama3_8b", "pretrain", "h100", "bf16", None): 8,
    ("llama3_8b", "pretrain", "h100", "fp8cs", None): 8,
    ("nemotronh_56b", "pretrain", "b200", "fp8cs", None): 64,
    ("nemotronh_56b", "pretrain", "gb300", "fp8cs", None): 64,
    ("qwen3_30b_a3b", "pretrain", "b200", "bf16", None): 8,
    ("qwen3_30b_a3b", "pretrain", "b200", "fp8cs", None): 8,
    ("qwen3_30b_a3b", "pretrain", "gb200", "bf16", None): 8,
    ("qwen3_30b_a3b", "pretrain", "gb200", "fp8cs", None): 8,
    ("qwen3_30b_a3b", "pretrain", "gb300", "bf16", None): 8,
    ("qwen3_30b_a3b", "pretrain", "gb300", "fp8cs", None): 8,
    ("qwen3_30b_a3b", "pretrain", "h100", "bf16", None): 16,
    ("qwen3_30b_a3b", "pretrain", "h100", "fp8cs", None): 16,
}


def configure_slurm_gpu_tuning(
    executor: Any,
    *,
    enable_vboost: bool,
    lock_gpu_freq: int | None,
    peak_mem_clk: int | None,
) -> None:
    """Add optional GPU tuning commands to a Slurm executor before submission."""
    commands = []
    job_dir = executor.tunnel.job_dir

    if enable_vboost:
        commands.append(
            " ".join(
                [
                    "\n# Enable VBoost\n",
                    "srun",
                    f"--ntasks={executor.nodes}",
                    "--output",
                    os.path.join(job_dir, "vboost.out"),
                    "--error",
                    os.path.join(job_dir, "vboost.err"),
                    "bash -c",
                    shlex.quote("sudo nvidia-smi boost-slider --vboost 1"),
                ]
            )
        )

    if lock_gpu_freq is not None:
        commands.append(
            "\n".join(
                [
                    "",
                    "# Lock GPU graphics clock",
                    " ".join(
                        [
                            "srun",
                            "--ntasks-per-node=1",
                            "--output",
                            os.path.join(job_dir, "lock_gpu_freq.out"),
                            "--error",
                            os.path.join(job_dir, "lock_gpu_freq.err"),
                            "bash -c",
                            shlex.quote(f"sudo nvidia-smi -lgc {lock_gpu_freq}"),
                        ]
                    ),
                    "",
                ]
            )
        )

    if peak_mem_clk is not None:
        commands.append(
            "\n".join(
                [
                    "",
                    "# Lock GPU memory clock",
                    " ".join(
                        [
                            "srun",
                            f"--ntasks={executor.nodes}",
                            "--ntasks-per-node=1",
                            "--output",
                            os.path.join(job_dir, "peak_mem_clock.out"),
                            "--error",
                            os.path.join(job_dir, "peak_mem_clock.err"),
                            "bash -c",
                            shlex.quote(f"sudo nvidia-smi -lmc {peak_mem_clk},{peak_mem_clk}"),
                        ]
                    ),
                    "",
                ]
            )
        )

    if commands:
        executor.setup_lines = f"{executor.setup_lines or ''}{''.join(commands)}"


@dataclass
class WorkloadBaseConfig:
    """Container for workload base configs. This object exists because we cannot import MBridge on the headnode but need a place to store recipe overrides."""

    # NOTE: `num_gpus` is for representation purposes only. It is only meant to
    # communicate number of GPUs to be used for a specific workload. In this
    # refactored path, the value is derived from the selected flat perf recipe.

    # NOTE: You can specify number of GPUs to use for a SLURM job from command
    # line like `-ng/--num_gpus <num_gpus>` ("scripts/performance/README.md")
    # or update your sbatch script.
    num_gpus: int = 1

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int | None = None

    global_batch_size: int = 1
    micro_batch_size: int = 1

    env_vars: dict[str, str | int | float | bool] = field(default_factory=dict)

    use_megatron_fsdp: bool | None = None
    nccl_ub: bool | None = None
    cuda_graph_impl: str | None = None
    cuda_graph_scope: str | list[str] | None = None
    cpu_offloading_num_layers: int | None = None
    recompute_num_layers: int | None = None
    recompute_modules: list[str] | None = None

    # Fine-grained activation offloading
    fine_grained_activation_offloading: bool | None = None
    offload_modules: list[str] | None = None

    outer_dp_sharding_strategy: str | None = None
    num_distributed_optimizer_instances: int | None = None

    # MoE configuration
    moe_flex_dispatcher_backend: str | None = None
    moe_a2a_overlap: bool | None = False
    cutedsl_fused_grouped_mlp: bool | None = False
    fp8_dot_product_attention: bool | None = None
    peft: str | None = None

    # Pipeline parallelism layout
    pp_layout: str | None = None

    # TransformerEngine per-module precision overrides
    te_precision_config_file: str | None = None

    @property
    def sequence_parallel(self) -> bool:
        """Get the sequence parallel flag."""
        return bool(self.tensor_model_parallel_size > 1)

    @property
    def gbs_scaling_factor(self) -> float:
        """Get the global batch size scaling factor."""
        return self.global_batch_size / self.num_gpus


def explicit_environment_override_names(
    cli_overrides: list[str], base_env_vars: dict, effective_env_vars: dict
) -> set[str]:
    """Return recipe env names explicitly selected through Hydra overrides."""
    names = set()
    for override in cli_overrides:
        key = override.split("=", 1)[0].lstrip("+~")
        if key == "env_vars":
            from hydra.core.override_parser.overrides_parser import OverridesParser

            override_value = OverridesParser.create().parse_override(override).value()
            if isinstance(override_value, Mapping):
                names.update(override_value)
            names.update(
                name
                for name in base_env_vars.keys() | effective_env_vars.keys()
                if name not in base_env_vars
                or name not in effective_env_vars
                or base_env_vars[name] != effective_env_vars[name]
            )
        elif key.startswith("env_vars."):
            names.add(key.removeprefix("env_vars.").split(".", 1)[0])
    return names


def apply_target_topology_environment(
    config: Any,
    *,
    gpu: str,
    protected_env_names: set[str] | None = None,
) -> None:
    """Adapt only HybridEP topology for the launch target.

    Some hardware targets reuse a recipe whose static defaults describe H100.
    This step changes only the final HybridEP topology for the selected target;
    model-specific environment stays explicit in the recipe. Explicit
    ``env_vars`` overrides remain final.
    """
    topology_names = {
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
        "NVLINK_DOMAIN_SIZE",
        "USE_MNNVL",
    }
    protected = protected_env_names or set()
    if getattr(config.model, "moe_flex_dispatcher_backend", None) != "hybridep":
        for name in topology_names - protected:
            config.env_vars.pop(name, None)
        return

    expert_model_parallel_size = getattr(config.model, "expert_model_parallel_size", 1)
    if expert_model_parallel_size is None:
        expert_model_parallel_size = 1
    if expert_model_parallel_size <= 0:
        raise ValueError("HybridEP expert parallel size must be positive.")

    normalized_gpu = gpu.lower()
    if normalized_gpu in {"h100", "b200", "b300"}:
        topology = {
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": min(expert_model_parallel_size, 8),
            "NVLINK_DOMAIN_SIZE": 8,
            "USE_MNNVL": 0,
        }
    elif normalized_gpu in {"gb200", "gb300", "vr200", "r100"}:
        if expert_model_parallel_size > 72:
            raise ValueError("HybridEP expert parallel size must not exceed the 72-rank NVLink domain.")
        topology = {
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": expert_model_parallel_size,
            "NVLINK_DOMAIN_SIZE": 72,
            "USE_MNNVL": 1,
        }
    else:
        raise ValueError(f"Unsupported GPU type for HybridEP topology: {gpu!r}.")

    for name, value in topology.items():
        if name not in protected:
            config.env_vars[name] = value


def apply_feature_environment(
    config: Any,
    *,
    nccl_ub_override: bool | None,
    protected_env_names: set[str] | None = None,
) -> None:
    """Keep CLI-selected feature flags and their process settings consistent."""
    protected = protected_env_names or set()
    if nccl_ub_override is not True:
        return

    for name, value in {"NCCL_NVLS_ENABLE": 1, "NCCL_CTA_POLICY": 1}.items():
        if name not in protected:
            config.env_vars[name] = value


def apply_argparse_overrides(config: Any, args: Any) -> Any:
    """Apply argparse values that affect recipe configuration.

    The bootstrap and training entrypoints call this helper before Hydra
    overrides so parallelism and environment-relevant settings match.
    """
    if getattr(args, "nccl_ub", False):
        config.ddp.nccl_ub = True

    # Keep all parallelism overrides together so bootstrap topology and the
    # final training config resolve the same user input.
    tensor_model_parallel_size = getattr(args, "tensor_model_parallel_size", None)
    if tensor_model_parallel_size is not None:
        config.model.tensor_model_parallel_size = tensor_model_parallel_size

    pipeline_model_parallel_size = getattr(args, "pipeline_model_parallel_size", None)
    if pipeline_model_parallel_size is not None:
        config.model.pipeline_model_parallel_size = pipeline_model_parallel_size

    context_parallel_size = getattr(args, "context_parallel_size", None)
    if context_parallel_size is not None:
        config.model.context_parallel_size = context_parallel_size

    virtual_pipeline_model_parallel_size = getattr(args, "virtual_pipeline_model_parallel_size", -1)
    if virtual_pipeline_model_parallel_size != -1:
        config.model.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size

    expert_model_parallel_size = getattr(args, "expert_model_parallel_size", None)
    if expert_model_parallel_size is not None:
        config.model.expert_model_parallel_size = expert_model_parallel_size

    expert_tensor_parallel_size = getattr(args, "expert_tensor_parallel_size", None)
    if expert_tensor_parallel_size is not None:
        config.model.expert_tensor_parallel_size = expert_tensor_parallel_size

    dispatcher_backend = getattr(args, "moe_flex_dispatcher_backend", -1)
    num_moe_experts = getattr(config.model, "num_moe_experts", None)
    if dispatcher_backend in {"deepep", "hybridep"} and num_moe_experts not in {None, 0}:
        config.model.moe_flex_dispatcher_backend = dispatcher_backend
    elif dispatcher_backend in {"deepep", "hybridep"}:
        logger.warning("Ignoring flex dispatcher override for a model without MoE experts.")
    elif dispatcher_backend is None:
        config.model.moe_flex_dispatcher_backend = None

    return config


def finalize_config_overrides(config: Any) -> Any:
    """Reconcile config invariants after argparse and Hydra overrides.

    Hydra has final precedence over primary fields such as ``ddp.nccl_ub`` and
    ``model.moe_flex_dispatcher_backend``. Dependent fields are therefore
    normalized only after both override layers have completed.
    """
    if getattr(config.ddp, "nccl_ub", False):
        config.ddp.average_in_collective = False
        if getattr(config.ddp, "use_megatron_fsdp", False):
            config.ddp.fsdp_manual_registration = True
    elif getattr(config.ddp, "fsdp_manual_registration", False):
        config.ddp.fsdp_manual_registration = False

    dispatcher_backend = getattr(config.model, "moe_flex_dispatcher_backend", None)
    num_moe_experts = getattr(config.model, "num_moe_experts", None)
    if dispatcher_backend in {"deepep", "hybridep"} and num_moe_experts not in {None, 0}:
        config.model.moe_token_dispatcher_type = "flex"
        config.model.moe_shared_expert_overlap = False
    elif dispatcher_backend is None and getattr(config.model, "moe_token_dispatcher_type", None) == "flex":
        config.model.moe_token_dispatcher_type = "alltoall"

    return config


def _normalize_precision_name(precision: str) -> str:
    return _PRECISION_NAME_MAP.get(precision.lower(), precision.lower().replace("_", ""))


def _recipe_variant_suffix(config_variant: str | None) -> str:
    # Temporary compatibility for legacy nemo-ci configs that still pass the
    # removed v1/v2 labels. They select the canonical recipe, not a variant.
    if config_variant is None or config_variant.lower() in {"v1", "v2"}:
        return ""
    return f"_{config_variant.lower()}"


def _recipe_variant_name(config_variant: str | None) -> str | None:
    # Keep legacy nemo-ci v1/v2 labels out of canonical recipe names.
    return None if config_variant is None or config_variant.lower() in {"v1", "v2"} else config_variant.lower()


def _display_config_variant(config_variant: str | None) -> str:
    return "default" if config_variant is None else config_variant


def _recipe_function_name(
    *,
    model_recipe_name: str,
    task: str,
    num_gpus: int,
    gpu: str,
    precision: str,
    config_variant: str | None = None,
) -> str:
    precision_name = _normalize_precision_name(precision)
    variant_suffix = _recipe_variant_suffix(config_variant)
    return f"{model_recipe_name}_{task}_{num_gpus}gpu_{gpu}_{precision_name}{variant_suffix}_config"


def _select_default_recipe_name(
    *,
    model_recipe_name: str,
    task: str,
    gpu: str,
    precision: str,
    config_variant: str | None,
    matches: list[tuple[int, str]],
) -> str:
    override = _DEFAULT_GPU_COUNT_OVERRIDES.get(
        (model_recipe_name, task, gpu, _normalize_precision_name(precision), _recipe_variant_name(config_variant))
    )
    if override is not None:
        for gpu_count, name in matches:
            if gpu_count == override:
                return name
        available_gpu_counts = [str(gpu_count) for gpu_count, _ in matches]
        raise ValueError(
            f"Default GPU count {override} for {model_recipe_name}/{task}/{gpu}/"
            f"{_normalize_precision_name(precision)}/{_display_config_variant(config_variant)} did not match "
            f"available flat perf recipes: {', '.join(available_gpu_counts)}."
        )
    return matches[-1][1]


def _emit(message: str = "", *, end: str = "\n", flush: bool = False) -> None:
    sys.stdout.write(f"{message}{end}")
    if flush:
        sys.stdout.flush()


@functools.lru_cache(maxsize=1)
def perf_recipe_family_modules() -> tuple[str, ...]:
    """Return import paths for flat performance recipe family packages."""
    import megatron.bridge.perf_recipes as perf_recipes

    module_names = [
        f"{perf_recipes.__name__}.{module_info.name}"
        for module_info in pkgutil.iter_modules(perf_recipes.__path__)
        if module_info.ispkg and not module_info.name.startswith("_")
    ]
    return tuple(sorted(module_names))


def find_perf_recipe(recipe_name: str) -> Callable | None:
    """Find a flat perf recipe function by exported function name."""
    for module_name in perf_recipe_family_modules():
        recipe_fn = getattr(importlib.import_module(module_name), recipe_name, None)
        if callable(recipe_fn):
            return recipe_fn
    return None


@functools.lru_cache(maxsize=1)
def flat_perf_recipe_names() -> tuple[str, ...]:
    """Return flat perf recipe names without importing recipe family modules."""
    names = set()
    if _PERF_RECIPES_ROOT.exists():
        for path in _PERF_RECIPES_ROOT.rglob("*.py"):
            names.update(_PERF_RECIPE_DEF_PATTERN.findall(path.read_text()))
    return tuple(sorted(names))


def get_perf_recipe_by_name(
    model_recipe_name: str,
    task: str,
    num_gpus: int,
    gpu: str,
    precision: str,
    config_variant: str | None = None,
):
    """Load an environment-finalized recipe from ``megatron.bridge.perf_recipes``."""
    name = _recipe_function_name(
        model_recipe_name=model_recipe_name,
        task=task,
        num_gpus=num_gpus,
        gpu=gpu,
        precision=precision,
        config_variant=config_variant,
    )
    recipe_fn = find_perf_recipe(name)
    if recipe_fn is None:
        searched_modules = ", ".join(perf_recipe_family_modules()) or "none"
        raise ValueError(f"No perf recipe {name!r} found in perf recipe packages: {searched_modules}.")
    return recipe_fn()


def _variant_pattern(
    *,
    model_recipe_name: str,
    task: str,
    gpu: str,
    precision: str,
    config_variant: str | None,
    num_gpus: int | None = None,
) -> re.Pattern:
    gpu_count = r"\d+" if num_gpus is None else str(num_gpus)
    precision_name = _normalize_precision_name(precision)
    variant_suffix = _recipe_variant_suffix(config_variant)
    return re.compile(
        rf"^{re.escape(model_recipe_name)}_{re.escape(task)}_{gpu_count}gpu_{re.escape(gpu)}_"
        rf"{re.escape(precision_name)}{re.escape(variant_suffix)}_config$"
    )


def _matching_perf_recipe_names(
    *,
    model_recipe_name: str,
    task: str,
    gpu: str,
    precision: str,
    config_variant: str | None,
    num_gpus: int | None = None,
) -> list[tuple[int, str]]:
    pattern = _variant_pattern(
        model_recipe_name=model_recipe_name,
        task=task,
        gpu=gpu,
        precision=precision,
        config_variant=config_variant,
        num_gpus=num_gpus,
    )
    matches: list[tuple[int, str]] = []
    for name in flat_perf_recipe_names():
        match = pattern.match(name)
        if match:
            gpu_match = re.search(r"_(\d+)gpu_", name)
            if gpu_match is not None:
                matches.append((int(gpu_match.group(1)), name))
    return sorted(matches)


def _first_matching_perf_recipe_name(
    *,
    model_recipe_name: str,
    task: str,
    gpu: str,
    precision: str,
    config_variant: str | None,
    num_gpus: int | None = None,
) -> str:
    matches = _matching_perf_recipe_names(
        model_recipe_name=model_recipe_name,
        task=task,
        gpu=gpu,
        precision=precision,
        config_variant=config_variant,
        num_gpus=num_gpus,
    )
    if not matches:
        available_variants = _list_available_config_variants(
            model_recipe_name=model_recipe_name,
            gpu=gpu,
            compute_dtype=precision,
            task=task,
        )
        raise ValueError(
            f"No flat perf recipe found for {model_recipe_name}/{task}/{gpu}/{precision}"
            f"/{_display_config_variant(config_variant)}. Available variants: "
            f"{[_display_config_variant(variant) for variant in available_variants]}"
        )
    if num_gpus is None:
        return _select_default_recipe_name(
            model_recipe_name=model_recipe_name,
            task=task,
            gpu=gpu,
            precision=precision,
            config_variant=config_variant,
            matches=matches,
        )
    return matches[0][1]


def _workload_base_config_from_recipe(config, *, num_gpus: int) -> WorkloadBaseConfig:
    model = config.model
    train = config.train
    ddp = config.ddp
    comm_overlap = getattr(config, "comm_overlap", None)
    mixed_precision = getattr(config, "mixed_precision", None)
    return WorkloadBaseConfig(
        num_gpus=num_gpus,
        tensor_model_parallel_size=model.tensor_model_parallel_size,
        pipeline_model_parallel_size=model.pipeline_model_parallel_size,
        context_parallel_size=getattr(model, "context_parallel_size", 1),
        virtual_pipeline_model_parallel_size=model.virtual_pipeline_model_parallel_size,
        expert_model_parallel_size=getattr(model, "expert_model_parallel_size", 1),
        expert_tensor_parallel_size=getattr(model, "expert_tensor_parallel_size", None),
        global_batch_size=train.global_batch_size,
        micro_batch_size=train.micro_batch_size,
        env_vars=dict(getattr(config, "env_vars", {})),
        use_megatron_fsdp=getattr(ddp, "use_megatron_fsdp", None),
        nccl_ub=getattr(ddp, "nccl_ub", None),
        cuda_graph_impl=getattr(model, "cuda_graph_impl", None),
        cuda_graph_scope=getattr(model, "cuda_graph_scope", None),
        cpu_offloading_num_layers=getattr(model, "cpu_offloading_num_layers", None),
        recompute_num_layers=getattr(model, "recompute_num_layers", None),
        recompute_modules=getattr(model, "recompute_modules", None),
        fine_grained_activation_offloading=getattr(model, "fine_grained_activation_offloading", None),
        offload_modules=getattr(model, "offload_modules", None),
        outer_dp_sharding_strategy=getattr(ddp, "outer_dp_sharding_strategy", None),
        num_distributed_optimizer_instances=getattr(ddp, "num_distributed_optimizer_instances", None),
        moe_flex_dispatcher_backend=getattr(model, "moe_flex_dispatcher_backend", None),
        moe_a2a_overlap=getattr(comm_overlap, "overlap_moe_expert_parallel_comm", False),
        cutedsl_fused_grouped_mlp=getattr(model, "use_transformer_engine_op_fuser", False),
        fp8_dot_product_attention=getattr(mixed_precision, "fp8_dot_product_attention", None),
        peft=getattr(config, "peft", None),
        pp_layout=getattr(model, "pipeline_model_parallel_layout", None),
    )


def get_workload_base_config(
    model_family_name: str,
    model_recipe_name: str,
    gpu: str,
    compute_dtype: str,
    task: str,
    config_variant: str | None = None,
) -> WorkloadBaseConfig:
    """Return a compatibility workload config derived from flat perf recipes."""
    del model_family_name
    recipe_name = _first_matching_perf_recipe_name(
        model_recipe_name=model_recipe_name,
        task=task,
        gpu=gpu,
        precision=compute_dtype,
        config_variant=config_variant,
    )
    gpu_match = re.search(r"_(\d+)gpu_", recipe_name)
    if gpu_match is None:
        raise ValueError(f"Unable to infer GPU count from perf recipe name {recipe_name!r}.")
    recipe_fn = find_perf_recipe(recipe_name)
    if recipe_fn is None:
        raise ValueError(f"No perf recipe {recipe_name!r} found.")
    return _workload_base_config_from_recipe(recipe_fn(), num_gpus=int(gpu_match.group(1)))


def get_exp_name_config(
    args,
    model_family_name: str,
    model_recipe_name: str,
    gpu: str,
    compute_dtype: str,
    task: str,
    config_variant: str | None = None,
) -> str:
    """Get the experiment name from the flat perf recipe and user overrides."""
    base_config = get_workload_base_config(
        model_family_name,
        model_recipe_name,
        gpu,
        compute_dtype,
        task,
        config_variant,
    )
    num_gpus = args.num_gpus if args.num_gpus is not None else base_config.num_gpus
    tp_size = (
        args.tensor_model_parallel_size
        if args.tensor_model_parallel_size is not None
        else base_config.tensor_model_parallel_size
    )
    pp_size = (
        args.pipeline_model_parallel_size
        if args.pipeline_model_parallel_size is not None
        else base_config.pipeline_model_parallel_size
    )
    cp_size = (
        args.context_parallel_size if args.context_parallel_size is not None else base_config.context_parallel_size
    )
    vp_size = (
        args.virtual_pipeline_model_parallel_size
        if args.virtual_pipeline_model_parallel_size != -1
        else base_config.virtual_pipeline_model_parallel_size
    )
    ep_size = (
        args.expert_model_parallel_size
        if args.expert_model_parallel_size is not None
        else base_config.expert_model_parallel_size
    )
    etp_size = (
        args.expert_tensor_parallel_size
        if args.expert_tensor_parallel_size is not None
        else base_config.expert_tensor_parallel_size
    )
    mbs_size = args.micro_batch_size if args.micro_batch_size is not None else base_config.micro_batch_size

    if args.global_batch_size is not None:
        gbs_size = args.global_batch_size
    elif num_gpus != base_config.num_gpus:
        gbs_size = int(base_config.gbs_scaling_factor * num_gpus)
    else:
        gbs_size = base_config.global_batch_size

    return (
        f"gpus{num_gpus}_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}"
        f"_ep{ep_size}_etp{etp_size}_mbs{mbs_size}_gbs{gbs_size}"
    )


def list_available_config_variants(
    *,
    model_family_name: str,
    model_recipe_name: str,
    gpu: str,
    compute_dtype: str,
    task: str,
) -> list[str | None]:
    """List all available config variants for a model/task/gpu/dtype combination."""
    del model_family_name
    return _list_available_config_variants(
        model_recipe_name=model_recipe_name,
        gpu=gpu,
        compute_dtype=compute_dtype,
        task=task,
    )


def _list_available_config_variants(
    *,
    model_recipe_name: str,
    gpu: str,
    compute_dtype: str,
    task: str,
) -> list[str | None]:
    precision_name = _normalize_precision_name(compute_dtype)
    prefix = f"{model_recipe_name}_{task}_"
    suffix = f"gpu_{gpu}_{precision_name}"
    variants: set[str | None] = set()
    for name in flat_perf_recipe_names():
        if not name.startswith(prefix) or not name.endswith("_config") or suffix not in name:
            continue
        middle = name.removesuffix("_config").split(suffix, maxsplit=1)[1]
        variants.add(middle.removeprefix("_") or None)
    return sorted(variants, key=lambda variant: (variant is not None, variant or ""))


def get_perf_optimized_recipe(
    model_family_name: str,
    model_recipe_name: str,
    train_task: str,
    gpu: str,
    compute_dtype: str,
    mock: bool = True,
    config_variant: str | None = None,
    optimizer_type: str | None = None,
    num_gpus: int | None = None,
):
    """Get a performance optimized recipe from flat perf recipes."""
    del model_family_name, mock
    recipe_name = _first_matching_perf_recipe_name(
        model_recipe_name=model_recipe_name,
        task=train_task,
        gpu=gpu,
        precision=compute_dtype,
        config_variant=config_variant,
        num_gpus=num_gpus,
    )
    recipe_fn = find_perf_recipe(recipe_name)
    if recipe_fn is None:
        raise ValueError(f"No perf recipe {recipe_name!r} found.")
    cfg = recipe_fn()
    if optimizer_type == "adam" and model_recipe_name == "kimi_k2":
        from megatron.bridge.recipes.kimi.kimi_k2 import _apply_kimi_k2_optimizer

        lr_decay_iters = cfg.scheduler.lr_decay_iters
        lr_warmup_iters = cfg.scheduler.lr_warmup_iters
        _apply_kimi_k2_optimizer(cfg, optimizer_type)
        cfg.scheduler.lr_decay_iters = lr_decay_iters
        cfg.scheduler.lr_warmup_iters = lr_warmup_iters
    return cfg


def build_recipe_config(model_family_name: str, model_recipe_name: str, train_task: str, wandb_experiment_name: str):
    """Build a recipe config and set its run output paths.

    Recipe builders no longer accept output-directory kwargs. This function
    calls the selected builder and then configures those paths on its result.

    The old API was: recipe_builder(dir="/nemo_run/", name=wandb_experiment_name)
    This set:
        - run_output_dir = "/nemo_run/{name}"
        - checkpoint_dir = "/nemo_run/{name}/checkpoints"
        - tensorboard_dir = "/nemo_run/{name}/tb_logs"
    """
    family_pkg_path = f"megatron.bridge.recipes.{model_family_name}"
    family_pkg = importlib.import_module(family_pkg_path)

    if model_recipe_name == "deepseek_v3_32nodes" and train_task == "pretrain":
        model_recipe_name = "deepseek_v3_pretrain_config_32nodes"
    elif train_task == "peft":
        model_recipe_name = f"{model_recipe_name}_peft_config"
    else:
        model_recipe_name = f"{model_recipe_name}_{train_task}_config"

    recipe_builder = getattr(family_pkg, model_recipe_name)

    # Recipe builders no longer accept output path kwargs. Configure the
    # returned ConfigContainer instead.
    cfg = recipe_builder()

    # Set output directories that were previously configured via dir="/nemo_run/" and name=wandb_experiment_name
    run_output_dir = os.path.join("/nemo_run", wandb_experiment_name)

    # Checkpoint paths
    cfg.checkpoint.save = os.path.join(run_output_dir, "checkpoints")
    cfg.checkpoint.load = os.path.join(run_output_dir, "checkpoints")

    # Logger paths
    cfg.logger.tensorboard_dir = os.path.join(run_output_dir, "tb_logs")
    cfg.logger.wandb_exp_name = wandb_experiment_name
    cfg.logger.wandb_save_dir = os.path.join(run_output_dir, "wandb")

    return cfg


class _Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"


def _display_config_variants(
    model_family_name: str,
    model_recipe_name: str,
    gpu: str,
    compute_dtype: str,
    task: str,
    variants: list[str | None],
    timeout: int,
) -> None:
    """Display available config variants with their derived workload configs."""
    c = _Colors

    _emit(f"\n{c.DIM}{'=' * 80}{c.RESET}")
    _emit(
        f"{c.BOLD}{c.WHITE}Available config variants for {c.CYAN}{model_recipe_name}{c.WHITE}/"
        f"{c.MAGENTA}{task}{c.WHITE}/{c.YELLOW}{gpu}{c.WHITE}/{c.GREEN}{compute_dtype}{c.WHITE}:{c.RESET}"
    )
    _emit(f"{c.DIM}{'=' * 80}{c.RESET}")

    highlight_fields = {"num_gpus", "global_batch_size"}

    for i, variant in enumerate(variants, 1):
        default_marker = f" {c.GREEN}(default){c.RESET}" if i == 1 else ""
        variant_label = _display_config_variant(variant)
        _emit(f"\n  {c.BOLD}{c.CYAN}[{i}]{c.RESET} {c.BOLD}{c.WHITE}{variant_label}{c.RESET}{default_marker}")
        _emit(f"  {c.DIM}{'-' * 76}{c.RESET}")

        try:
            config = get_workload_base_config(
                model_family_name,
                model_recipe_name,
                gpu,
                compute_dtype,
                task,
                variant,
            )
            for field in fields(config):
                value = getattr(config, field.name)
                if value is not None:
                    if field.name in highlight_fields:
                        _emit(f"      {c.CYAN}{field.name}: {value}{c.RESET}")
                    else:
                        _emit(f"      {field.name}: {value}")
        except ValueError:
            _emit(f"      {c.DIM}(config not found){c.RESET}")

    _emit(f"\n{c.DIM}{'=' * 80}{c.RESET}")
    _emit(f"\nSelect [1-{len(variants)}] (default: 1, timeout: {timeout}s): ", end="", flush=True)


def _get_user_selection_with_timeout(num_variants: int, timeout: int) -> int:
    """Get user selection with timeout, returning 1-based choice index.

    Args:
        num_variants: Number of available variants to choose from
        timeout: Timeout in seconds for user input

    Returns:
        1-based index of the selected variant (defaults to 1 on timeout/invalid input)
    """
    try:
        ready, _, _ = select.select([sys.stdin], [], [], float(timeout))
        if ready:
            user_input = sys.stdin.readline().strip()
            if user_input == "":
                return 1
            try:
                choice = int(user_input)
                if choice < 1 or choice > num_variants:
                    _emit("Invalid choice. Using default (1).")
                    return 1
                return choice
            except ValueError:
                _emit("Invalid input. Using default (1).")
                return 1
        else:
            _emit("\nTimeout - proceeding with default (1)")
            return 1
    except (OSError, AttributeError):
        # select.select doesn't work on Windows, fall back to default
        logger.warning("Interactive selection not available on this platform. Using default variant.")
        return 1


def select_config_variant_interactive(
    model_family_name: str,
    model_recipe_name: str,
    gpu: str,
    compute_dtype: str,
    task: str,
    timeout: int = CONFIG_VARIANT_SELECTION_TIMEOUT,
    force_interactive: bool = False,
) -> str | None:
    """Interactively select a config variant if multiple variants are available."""
    variants = list_available_config_variants(
        model_family_name=model_family_name,
        model_recipe_name=model_recipe_name,
        gpu=gpu,
        compute_dtype=compute_dtype,
        task=task,
    )

    if not variants:
        logger.warning(
            "No config variants found for %s/%s/%s/%s. Using default variant.",
            model_recipe_name,
            task,
            gpu,
            compute_dtype,
        )
        return None

    if len(variants) == 1 and not force_interactive:
        logger.info("Only one config variant available: %s", _display_config_variant(variants[0]))
        return variants[0]

    _display_config_variants(model_family_name, model_recipe_name, gpu, compute_dtype, task, variants, timeout)
    selection = _get_user_selection_with_timeout(len(variants), timeout)
    selected_variant = variants[selection - 1]
    logger.info("Selected config variant: %s", _display_config_variant(selected_variant))
    return selected_variant
