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

import re
from typing import List, Optional, Tuple

import torch
from megatron.core.transformer.module import MegatronModule
from rich.table import Table

from megatron.bridge.utils.common_utils import unwrap_model


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
    for name, param in bridge(megatron_model, show_progress=True):
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
