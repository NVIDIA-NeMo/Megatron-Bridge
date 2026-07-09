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

"""Composable environment defaults for library and performance recipes."""

import re
from collections.abc import Callable, Mapping, MutableMapping, Set
from functools import wraps
from typing import ParamSpec

from megatron.bridge.training.config import ConfigContainer


LIBRARY_PROCESS_ENV_DEFAULTS = {
    "NCCL_GRAPH_REGISTER": 0,
    "NCCL_NVLS_ENABLE": 0,
    "NVTE_NORM_BWD_USE_CUDNN": 1,
    "NVTE_NORM_FWD_USE_CUDNN": 1,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
    "TORCH_NCCL_HIGH_PRIORITY": 1,
}

_HYBRIDEP_ENV_NAMES = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NVLINK_DOMAIN_SIZE",
    "USE_MNNVL",
}
_LIBRARY_MANAGED_ENV_NAMES = set(LIBRARY_PROCESS_ENV_DEFAULTS) | _HYBRIDEP_ENV_NAMES | {"NCCL_CTA_POLICY"}
_RECIPE_NAME_PATTERN = re.compile(
    r"^(?P<model_recipe_name>.+)_(?P<train_task>pretrain|sft|peft)_\d+gpu_"
    r"(?P<gpu>[a-z0-9]+)_(?P<precision>bf16|fp8cs|fp8mx|fp8sc|nvfp4)(?:_.+)?_config$"
)
_PRECISION_NAMES = {
    "bf16": "bf16",
    "fp8cs": "fp8_cs",
    "fp8mx": "fp8_mx",
    "fp8sc": "fp8_sc",
    "nvfp4": "nvfp4",
}
_P = ParamSpec("_P")


def _set_derived(
    env_vars: MutableMapping[str, str | int | float | bool],
    name: str,
    value: str | int | float | bool,
    protected_env_names: Set[str],
) -> None:
    """Set a derived value unless the user explicitly protected that name."""
    if name not in protected_env_names:
        env_vars[name] = value


def _remove_derived(
    env_vars: MutableMapping[str, str | int | float | bool],
    names: set[str],
    protected_env_names: Set[str],
) -> None:
    """Remove prior derived values without deleting explicit overrides."""
    for name in names - protected_env_names:
        env_vars.pop(name, None)


def _hybridep_topology(gpu: str, expert_model_parallel_size: int) -> Mapping[str, int]:
    """Return topology-dependent HybridEP process settings."""
    if expert_model_parallel_size <= 0:
        raise ValueError("HybridEP expert parallel size must be positive.")

    normalized_gpu = gpu.lower()
    if normalized_gpu in {"h100", "b200", "b300"}:
        return {
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": min(expert_model_parallel_size, 8),
            "NVLINK_DOMAIN_SIZE": 8,
            "USE_MNNVL": 0,
        }
    if normalized_gpu not in {"gb200", "gb300", "vr200", "r100"}:
        raise ValueError(f"Unsupported GPU type for HybridEP topology: {gpu!r}.")
    if expert_model_parallel_size > 72:
        raise ValueError("HybridEP expert parallel size must not exceed the 72-rank NVLink domain.")
    return {
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": expert_model_parallel_size,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }


def recipe_environment_metadata(recipe_name: str) -> dict[str, str]:
    """Parse environment metadata from a canonical hardware recipe name."""
    match = _RECIPE_NAME_PATTERN.fullmatch(recipe_name)
    if match is None:
        raise ValueError(f"Invalid hardware recipe name: {recipe_name!r}.")
    return {
        "model_recipe_name": match["model_recipe_name"],
        "train_task": match["train_task"],
        "gpu": match["gpu"],
        "compute_dtype": _PRECISION_NAMES[match["precision"]],
    }


def apply_library_recipe_environment(
    config: ConfigContainer,
    *,
    gpu: str,
    protected_env_names: set[str] | None = None,
) -> None:
    """Move legacy executor process defaults onto a resolved library recipe.

    Explicit environment overrides remain protected. Reapplying this function
    after model overrides removes stale topology or NCCL-UB-derived values.

    Args:
        config: Final library recipe to update.
        gpu: Target GPU architecture.
        protected_env_names: Explicit environment overrides that derived rules
            must not replace or remove.
    """
    protected = protected_env_names or set()
    env_vars = config.env_vars
    model = config.model
    ddp = config.ddp

    _remove_derived(env_vars, _LIBRARY_MANAGED_ENV_NAMES, protected)
    for name, value in LIBRARY_PROCESS_ENV_DEFAULTS.items():
        _set_derived(env_vars, name, value, protected)

    if getattr(ddp, "nccl_ub", None) is True:
        _set_derived(env_vars, "NCCL_NVLS_ENABLE", 1, protected)
        _set_derived(env_vars, "NCCL_CTA_POLICY", 1, protected)

    if getattr(model, "moe_flex_dispatcher_backend", None) == "hybridep":
        expert_model_parallel_size = getattr(model, "expert_model_parallel_size", 1) or 1
        for name, value in _hybridep_topology(gpu, expert_model_parallel_size).items():
            _set_derived(env_vars, name, value, protected)


def library_recipe_environment(
    *, model_family_name: str
) -> Callable[[Callable[_P, ConfigContainer]], Callable[_P, ConfigContainer]]:
    """Finalize process environment defaults on a hardware library recipe.

    Args:
        model_family_name: Family package containing the recipe builder. Kept
            explicit on each builder for discoverability and future exceptions.

    Returns:
        A decorator that finalizes ``ConfigContainer.env_vars``.
    """

    def decorate(recipe_fn: Callable[_P, ConfigContainer]) -> Callable[_P, ConfigContainer]:
        metadata = recipe_environment_metadata(recipe_fn.__name__)

        @wraps(recipe_fn)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> ConfigContainer:
            config = recipe_fn(*args, **kwargs)
            apply_library_recipe_environment(config, gpu=metadata["gpu"])
            return config

        wrapped.__recipe_environment_family__ = model_family_name
        return wrapped

    return decorate


def set_common_recipe_environment_defaults(config: ConfigContainer) -> None:
    """Set common Transformer Engine and compilation environment defaults."""
    defaults = {
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "TORCHINDUCTOR_WORKER_START": "fork",
        "QUANTIZATION_TYPE_DEBUG": 1,
    }
    for name, value in defaults.items():
        config.env_vars.setdefault(name, value)


def set_hybridep_environment_defaults(
    config: ConfigContainer,
    *,
    ranks_per_nvlink_domain: int,
    use_mnnvl: bool,
) -> None:
    """Set HybridEP topology defaults on a recipe config.

    Args:
        config: Recipe config to update.
        ranks_per_nvlink_domain: Number of HybridEP ranks in each NVLink domain.
        use_mnnvl: Whether the workload uses a multi-node NVLink domain.
    """
    defaults = {
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": ranks_per_nvlink_domain,
        "NVLINK_DOMAIN_SIZE": 72 if use_mnnvl else 8,
        "USE_MNNVL": int(use_mnnvl),
    }
    for name, value in defaults.items():
        config.env_vars.setdefault(name, value)
