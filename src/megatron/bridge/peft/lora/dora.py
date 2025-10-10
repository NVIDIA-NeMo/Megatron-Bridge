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

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch
from torch import nn

from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora.dora_layers import DoRALinear, ParallelLinearDoRAAdapter
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.utils import get_adapter_attributes_from_linear


logger = logging.getLogger(__name__)


@dataclass
class DoRA(PEFT, ModuleMatcher):
    """
    Implements the DoRA (Weight-Decomposed LowRank Adaptation) module for parameter-efficient fine-tuning.

    DoRA decomposes pre-trained weight into magnitude and direction, and uses a low-rank projection in the
    directional component to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of DoRA to specific modules within the model architecture.

    Args:
        target_modules (List[str], optional): A list of module names to apply DoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply DoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply DoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply DoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply DoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add DoRA to only linear_qkv
                on the first two layers.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 64.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'.
        lora_A_init_method (str): Initialization method for the low-rank matrix A. Defaults to "xavier".
        lora_B_init_method (str): Initialization method for the low-rank matrix B. Defaults to "zero".
    """

    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    dim: int = 32
    alpha: int = 64
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"

    def __post_init__(self):
        """Initialize attributes from parent classes and validate configuration."""
        assert self.dropout_position == "pre", (
            "DoRA only supports pre-adapter dropout at this time. Please set DoRA(..., dropout_position='pre')"
        )

    def transform(self, m: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """
        Applies DoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply DoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with DoRA applied, or the original module if not a target.
        """
        # Skip already transformed modules
        if isinstance(m, DoRALinear):
            return m

        if (ans := self.match(m, name, prefix)) is not None:
            (match, full_name) = ans
            input_is_parallel, in_features, out_features, disable_sp_comm, base_linear_is_parallel = (
                get_adapter_attributes_from_linear(m)
            )
            logger.info(f"Adding DoRA to: {full_name}")
            adapter = ParallelLinearDoRAAdapter(
                in_features,
                out_features,
                self.dim,
                base_linear_name=full_name,
                activation="identity",
                norm_type=None,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                gather_output=False,
                input_is_parallel=input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(m, "config", None),
                alpha=self.alpha,
                disable_sequence_parallel_comm=disable_sp_comm,
                base_linear_is_parallel=base_linear_is_parallel,
            )
            return DoRALinear(m, adapter)
        return m

    def get_adapter_parameter_patterns(self) -> Dict[str, List[str]]:
        """DoRA has A, B matrices plus magnitude vectors."""
        return {
            ".weight": [
                ".adapter.linear_in.weight",
                ".adapter.linear_out.weight",
                ".adapter.weight_magnitude",  # DoRA-specific
            ]
        }

    def merge(self, model):
        """Merge DoRA adapter weights into base model weights.

        DoRA merging involves both magnitude and direction components.
        The merge process reconstructs the full weight matrix from the
        decomposed components.

        Args:
            model: The model with DoRA adapters applied

        Returns:
            The model with DoRA adapters merged and unwrapped
        """
        # First merge DoRA weights using the DoRAMerge transform
        merge_transform = DoRAMerge()
        merge_transform(model, training=False)

        # Then unwrap adapter modules to return clean base structure
        unwrapped_model = []
        for stage in (model if isinstance(model, list) else [model]):
            unwrapped_stage = self._unwrap_dora_modules(stage)
            unwrapped_model.append(unwrapped_stage)

        return unwrapped_model

    def _unwrap_dora_modules(self, module):
        """Recursively unwrap DoRA adapter modules."""
        # Handle DoRA wrapper types
        if isinstance(module, DoRALinear):
            # Return the unwrapped base module (no weight merge for now)
            return module.to_wrap

        # For non-adapter modules, recursively unwrap children
        for name, child in list(module.named_children()):
            unwrapped_child = self._unwrap_dora_modules(child)
            setattr(module, name, unwrapped_child)

        return module


class DoRAMerge(PEFT):
    """
    Implements the DoRA weight merge for parameter-efficient fine-tuning.
    """

    @torch.no_grad()
    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """
        Merges the DoRA adapter with the base model weights.

        DoRA decomposition: W = m * (W0 + BA) / ||W0 + BA||
        Where:
        - m: magnitude vector
        - W0: base weight
        - BA: low-rank adaptation
        - ||.||: L2 norm along appropriate dimension

        Args:
            module (nn.Module): The module to apply DoRA merge to.
            name (str, optional): Name of the module to merge. Defaults to None.
            prefix (str, optional): Prefix for the module name. Defaults to None.

        Returns:
            nn.Module: The modified module with the DoRA adapter merged into the base model weights.
        """
        if isinstance(module, DoRALinear):
            logging.info(f"merging DoRALinear {(prefix if prefix else '') + '.' + (name if name else '')}")
            base_weight = module.to_wrap.weight
            adapter = module.adapter

            # Calculate low-rank adaptation: BA
            lora_delta = (
                adapter.alpha
                / adapter.dim
                * adapter.linear_out.weight.to(base_weight.device)
                @ adapter.linear_in.weight.to(base_weight.device)
            )

            # Calculate directional component: W0 + BA
            directional = base_weight + lora_delta

            # Calculate magnitude scaling: m / ||W0 + BA||
            if hasattr(adapter, 'weight_magnitude'):
                # Get magnitude vector
                magnitude = adapter.weight_magnitude.to(base_weight.device)

                # Calculate L2 norm along the output dimension (dim=1 for weight matrices)
                direction_norms = torch.norm(directional, dim=1, keepdim=True)

                # Avoid division by zero
                direction_norms = torch.clamp(direction_norms, min=1e-8)

                # Apply magnitude scaling: W = m * (W0 + BA) / ||W0 + BA||
                merged_weight = magnitude.unsqueeze(1) * directional / direction_norms
            else:
                # Fallback: if no magnitude vector, use directional component
                logging.warning(f"No magnitude vector found for DoRA module {name}, using directional component only")
                merged_weight = directional

            # Update base weight with merged result
            module.to_wrap.weight.data = merged_weight
            return module

        return module
