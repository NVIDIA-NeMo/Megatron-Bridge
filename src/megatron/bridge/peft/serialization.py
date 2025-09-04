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

"""Serialization utilities for PEFT adapter parameter handling.

This module provides utilities for normalizing HuggingFace PEFT parameter names
and handling parameter mappings with distributed training support.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping


def normalize_hf_key(key: str, adapter_name: Optional[str] = "default") -> str:
    """
    Normalize HuggingFace PEFT keys to canonical form.
    
    Removes common prefixes and adapter namespaces to create clean parameter names
    that can be matched against Megatron parameter patterns.
    
    Args:
        key: The original HuggingFace parameter key
        adapter_name: The adapter name to remove from keys
    
    Returns:
        Normalized parameter key
    
    Examples:
        >>> normalize_hf_key("base_model.model.layers.0.self_attn.q_proj.lora_A.weight")
        "layers.0.self_attn.q_proj.lora_A.weight"
        >>> normalize_hf_key("model.layers.0.mlp.gate_proj.lora_B.weight", "default")
        "layers.0.mlp.gate_proj.lora_B.weight"
    """
    # Remove common prefixes
    prefixes = ["base_model.model.", "base_model.", "model."]
    for prefix in prefixes:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    
    # Remove adapter namespace if present
    if adapter_name and key.startswith(f"{adapter_name}."):
        key = key[len(adapter_name) + 1:]
    
    # Handle modules_to_save prefix
    if key.startswith("modules_to_save."):
        key = key[len("modules_to_save."):]
    
    return key


class AdapterAutoMapping(MegatronParamMapping[torch.Tensor]):
    """Auto-detecting parameter mapping for PEFT adapters with EP/TP support.
    
    This mapping class automatically handles tensor parallel and expert parallel
    distribution based on the base module's parallelism type and adapter structure.
    """
    
    @property
    def is_expert(self) -> bool:
        """Check if this is an expert parameter."""
        return ".mlp.experts.linear_fc" in self.megatron_param
    
    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: nn.Module
    ) -> torch.Tensor:
        """Convert HF weights to Megatron format with correct parallelism.
        
        Args:
            hf_weights: HuggingFace format weights
            megatron_module: The Megatron module to load weights into
            
        Returns:
            Weights distributed according to the base module's parallelism
        """
        # Use module directly to determine parallelism type
        base = megatron_module.to_wrap if hasattr(megatron_module, "to_wrap") else megatron_module
        partition_dim = getattr(base, "partition_dim", None)
        
        # Determine if this is A (linear_in) or B (linear_out)
        is_matrix_a = "linear_in" in self.megatron_param
        
        # Apply correct distribution based on base parallelism
        if partition_dim == 0:  # Column-parallel base
            if is_matrix_a:
                # A is replicated for column-parallel
                return self._replicate_to_tp_ranks(hf_weights)
            else:
                # B follows column-parallel (split dim 0)
                return self._column_parallel_scatter(hf_weights)
        elif partition_dim == 1:  # Row-parallel base
            if is_matrix_a:
                # A follows row-parallel (split dim 1 for weight, dim 0 for bias)
                dim = 1 if hf_weights.ndim == 2 else 0
                return self._row_parallel_scatter(hf_weights, dim=dim)
            else:
                # B is replicated for row-parallel
                return self._replicate_to_tp_ranks(hf_weights)
        else:  # Replicated base
            return self._replicate_to_tp_ranks(hf_weights)
    
    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Convert Megatron weights to HF format with proper gathering.
        
        Args:
            megatron_weights: Megatron format weights (potentially sharded)
            megatron_module: The Megatron module
            
        Returns:
            Dictionary with HF parameter name and gathered weights
        """
        if megatron_weights is None:
            return {}
            
        # Handle EP gathering if needed
        if self.is_expert:
            gathered = self.gather_from_ep_ranks(
                megatron_weights,
                megatron_module,
                self.hf_param
            )
            return gathered
        
        # Standard TP gathering
        gathered_weights = self.gather_from_tp_ranks(megatron_weights)
        
        # Reconstruct full weight from gathered pieces
        if len(gathered_weights) > 1:
            # Need to determine concat dimension based on base parallelism
            base = megatron_module.to_wrap if hasattr(megatron_module, "to_wrap") else megatron_module
            partition_dim = getattr(base, "partition_dim", None)
            is_matrix_a = "linear_in" in self.megatron_param
            
            if partition_dim == 0 and not is_matrix_a:
                # Column-parallel B matrix: concat on dim 0
                full_weight = torch.cat(gathered_weights, dim=0)
            elif partition_dim == 1 and is_matrix_a:
                # Row-parallel A matrix: concat on appropriate dim
                concat_dim = 1 if megatron_weights.ndim == 2 else 0
                full_weight = torch.cat(gathered_weights, dim=concat_dim)
            else:
                # Replicated case: just take the first (they should be identical)
                full_weight = gathered_weights[0]
        else:
            full_weight = gathered_weights[0]
        
        return {self.hf_param: full_weight}
    
    def _replicate_to_tp_ranks(self, tensor: torch.Tensor) -> torch.Tensor:
        """Replicate tensor to all TP ranks."""
        return self.broadcast_tensor_to_tp_ranks(tensor, src_rank=0)
    
    def _column_parallel_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scatter tensor for column parallel (split dim 0)."""
        tp_size = self.tp_size()
        if tp_size == 1:
            return tensor
            
        # Split tensor along dimension 0
        chunk_size = tensor.size(0) // tp_size
        chunks = torch.chunk(tensor, tp_size, dim=0)
        
        return self.scatter_to_tp_ranks(
            list(chunks),
            output_shape=torch.Size([chunk_size] + list(tensor.shape[1:])),
            dtype=tensor.dtype,
            device=tensor.device
        )
    
    def _row_parallel_scatter(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Scatter tensor for row parallel."""
        tp_size = self.tp_size()
        if tp_size == 1:
            return tensor
            
        # Split tensor along specified dimension
        chunk_size = tensor.size(dim) // tp_size
        chunks = torch.chunk(tensor, tp_size, dim=dim)
        
        # Calculate output shape
        output_shape = list(tensor.shape)
        output_shape[dim] = chunk_size
        
        return self.scatter_to_tp_ranks(
            list(chunks),
            output_shape=torch.Size(output_shape),
            dtype=tensor.dtype,
            device=tensor.device
        )