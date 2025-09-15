# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import functools
import re
import types
from typing import Iterable, List, Optional, Tuple

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model
from rich.table import Table
from transformers.configuration_utils import PretrainedConfig


def weights_verification_table(bridge, megatron_model) -> Table:
    """
    Returns a table comparing weights between a Hugging Face model and a Megatron-LM model.

    Args:
        bridge (AutoBridge): The bridge object containing model information.
        megatron_model: The Megatron-LM model instance.

    Returns:
        Table: A rich Table object with the comparison.
    """
    table = Table(title="Hugging Face Weights Verification")
    table.add_column("Weight Name", style="cyan")
    table.add_column("Shape")
    table.add_column("DType")
    table.add_column("Device")
    table.add_column("Matches Original", justify="center")

    # Check each weight against the original HF-model
    for name, param in bridge.export_hf_weights(megatron_model, show_progress=True):
        original_param = bridge.hf_pretrained.state[name]
        table.add_row(
            name,
            str(tuple(param.shape)),
            str(param.dtype).replace("torch.", ""),
            str(param.device),
            "✅" if torch.allclose(param, original_param.to(param.device), atol=1e-6) else "❌",
        )

    return table


def get_module_and_param_from_name(
    models: MegatronModule | List[MegatronModule],
    param_name: str,
    vp_stage: Optional[int] = None,
) -> Tuple[torch.nn.Module, torch.Tensor] | Tuple[torch.nn.Module, torch.Tensor, Tuple]:
    """
    Get parameter from specific VP stage, ensuring that parameter
    attributes are preserved. Supports both absolute and relative parameter names.

    Args:
        models: List of Megatron model instances or a submodule
        param_name: Dot-separated parameter name (can be absolute or relative to models)
        vp_stage: Virtual pipeline stage index (None for single stage)

    Returns:
        Tuple of (module, parameter) where module owns the parameter

    Raises:
        ValueError: If vp_stage is out of range or parameter doesn't exist

    Examples:
        Basic usage with full model:
        >>> module, param = get_module_and_param_from_name(
        ...     models=full_model,
        ...     param_name="transformer.layers.0.attention.query.weight"
        ... )

        Usage with model list and VP stage:
        >>> module, param = get_module_and_param_from_name(
        ...     models=[model1, model2, model3],
        ...     param_name="layers.0.mlp.dense.bias",
        ...     vp_stage=1
        ... )

        Usage with submodule and relative path:
        >>> linear_module = model.transformer.layers[0].mlp.dense
        >>> module, param = get_module_and_param_from_name(
        ...     models=linear_module,
        ...     param_name="weight"
        ... )

        Usage with submodule and absolute path (automatic suffix matching):
        >>> linear_module = model.transformer.layers[0].mlp.dense
        >>> module, param = get_module_and_param_from_name(
        ...     models=linear_module,
        ...     param_name="transformer.layers.0.mlp.dense.weight"
        ... )
        # Automatically matches "weight" suffix and returns the parameter

        Edge case with partial path matching:
        >>> attention_module = model.transformer.layers[0].attention
        >>> module, param = get_module_and_param_from_name(
        ...     models=attention_module,
        ...     param_name="layers.0.attention.query.weight"
        ... )
        # Matches "query.weight" suffix within the attention module
    """

    if isinstance(models, list):
        if vp_stage is None:
            model = models[0]
        else:
            if vp_stage >= len(models):
                raise ValueError(f"VP stage {vp_stage} out of range (max: {len(models) - 1})")
            model = models[vp_stage]
    else:
        model = models

    module = unwrap_model(model)
    splitted_name = param_name.split(".")

    # Try to find the parameter using the given parts
    def try_get_param(parts):
        param = module
        temp_module = module

        for i, part in enumerate(parts):
            if not hasattr(param, part):
                return None
            param = getattr(param, part)
            if i < len(parts) - 1:
                temp_module = getattr(temp_module, part)

        return temp_module, param

    # First try the full parameter name (current behavior)
    result = try_get_param(splitted_name)
    if result is not None:
        return result

    # If full name doesn't work, try suffixes of the parameter name
    # This handles cases where models is a submodule but param_name is absolute
    for start_idx in range(1, len(splitted_name)):
        suffix_parts = splitted_name[start_idx:]
        result = try_get_param(suffix_parts)
        if result is not None:
            return result

    # If no approach works, raise an error
    raise ValueError(f"Parameter '{param_name}' not found in model at VP stage {vp_stage}")


def remove_non_pickleables(obj, max_depth: int = 2, current_depth: int = 0):
    """Remove non-pickleable objects from a configuration object recursively.

    This utility function identifies and removes objects that cannot be pickled for
    inter-process communication, including functions, bound methods, partial
    functions, and other problematic callables.

    Args:
        obj: The object to clean
        max_depth: Maximum recursion depth (default: 2)
        current_depth: Current recursion depth (internal use)

    Returns:
        The cleaned object with non-pickleables removed
    """

    # Stop recursion if max depth reached
    if current_depth >= max_depth:
        return obj

    # Handle None
    if obj is None:
        return obj

    # Check if object is a problematic callable
    if callable(obj):
        # Allow classes/types but remove function objects, methods, partials
        if isinstance(obj, type):
            return obj
        elif hasattr(obj, "__call__") and (
            isinstance(obj, (types.FunctionType, types.MethodType, functools.partial)) or hasattr(obj, "__self__")
        ):  # bound methods
            return None

    # Handle dataclass/object with attributes
    if hasattr(obj, "__dict__"):
        # Create a copy to avoid modifying the original
        cleaned_obj = copy.copy(obj)

        for attr_name in list(vars(cleaned_obj).keys()):
            attr_value = getattr(cleaned_obj, attr_name)

            # Recursively clean attribute
            cleaned_value = remove_non_pickleables(attr_value, max_depth, current_depth + 1)

            # Set the cleaned value (or None if it was removed)
            setattr(cleaned_obj, attr_name, cleaned_value)

        return cleaned_obj

    # Handle lists
    elif isinstance(obj, list):
        return [remove_non_pickleables(item, max_depth, current_depth + 1) for item in obj]

    # Handle tuples
    elif isinstance(obj, tuple):
        return tuple(remove_non_pickleables(item, max_depth, current_depth + 1) for item in obj)

    # Handle dictionaries
    elif isinstance(obj, dict):
        return {key: remove_non_pickleables(value, max_depth, current_depth + 1) for key, value in obj.items()}

    # For primitive types and other safe objects, return as-is
    return obj


def extract_sort_key(param_name: str):
    """Extract sorting key based on layer and expert numbers."""

    # Extract at most 2 numbers: layer number and expert number
    # Pattern: *layers.d+.*d+ (layer number and potentially expert number)
    numbers = []
    # Find layer number
    layer_match = re.search(r"layers\.(\d+)", param_name)
    if layer_match:
        numbers.append(int(layer_match.group(1)))
    # Find expert number after bias or weight
    expert_match = re.search(r"(?:bias|weight)(\d+)", param_name)
    if expert_match:
        numbers.append(int(expert_match.group(1)))
    # Pad to ensure consistent comparison (max 2 numbers)
    while len(numbers) < 2:
        numbers.append(-1)
    numbers = numbers[:2]  # Keep at most 2 numbers
    return numbers, param_name


def get_causal_lm_class_via_auto_map(
    model_name_or_path: str,
    config: PretrainedConfig,
) -> type | None:
    """Return CausalLM class via config.auto_map if available; otherwise None.

    If auto_map["AutoModelForCausalLM"] is present in the config, returns the dynamically loaded class.
    Returns None when auto_map is absent or loading fails. Does not download weights.
    """
    auto_map = getattr(config, "auto_map", None)
    if auto_map and "AutoModelForCausalLM" in auto_map:
        auto_map_class = auto_map["AutoModelForCausalLM"]
        repo_id = model_name_or_path or getattr(config, "_name_or_path", None)
        if not repo_id:
            return None
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            return get_class_from_dynamic_module(
                class_reference=auto_map_class,
                pretrained_model_name_or_path=repo_id,
                cache_dir=None,
                force_download=False,
                resume_download=True,
                proxies=None,
                use_auth_token=None,
                revision=None,
                local_files_only=False,
                repo_id=repo_id,
            )
        except Exception:
            return None

    return None


def persistent_buffers(model: torch.nn.Module) -> Iterable[Tuple[str, torch.Tensor]]:
    """Return an iterator over persistent module buffers, yielding both the name of the buffer as well as the buffer itself."""

    for mod_prefix, mod in model.named_modules():
        # only local buffers; we'll add the prefix ourselves
        for local_name, buffer in mod.named_buffers(recurse=False):
            if local_name not in getattr(mod, "_non_persistent_buffers_set", set()):
                full_name = f"{mod_prefix + '.' if mod_prefix else ''}{local_name}"
                yield full_name, buffer
