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

"""Transform utilities for PEFT adapter conversions.

This module provides utility functions for converting between fused and canonical
LoRA adapter formats, handling the splitting and merging of QKV and MLP weights.
"""

from typing import Dict, Tuple

import torch
from megatron.core.transformer.transformer_config import TransformerConfig


class TransformFns:
    """Collection of transform functions for PEFT adapter conversion.

    These functions handle the conversion between fused LoRA adapters (applied to
    entire fused layers) and canonical LoRA adapters (applied to individual projections).
    """

    @staticmethod
    def split_qkv_lora_a(
        tensor: torch.Tensor, config: TransformerConfig
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split fused QKV LoRA A matrix into Q, K, V components.

        Args:
            tensor: Fused QKV LoRA A matrix [rank, hidden_size]
            config: Transformer configuration

        Returns:
            Tuple of (Q_A, K_A, V_A) tensors
        """
        # For LoRA A matrix, we replicate the same tensor for Q, K, V
        # since the input dimension is the same for all three
        return tensor, tensor, tensor

    @staticmethod
    def split_qkv_lora_b(
        tensor: torch.Tensor, config: TransformerConfig
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split fused QKV LoRA B matrix into Q, K, V components.

        Args:
            tensor: Fused QKV LoRA B matrix [qkv_size, rank]
            config: Transformer configuration

        Returns:
            Tuple of (Q_B, K_B, V_B) tensors
        """
        # Calculate dimensions
        kv_channels = config.kv_channels or (config.hidden_size // config.num_attention_heads)
        num_query_groups = config.num_query_groups
        heads_per_group = config.num_attention_heads // num_query_groups

        q_size = heads_per_group * kv_channels
        k_size = kv_channels
        v_size = kv_channels

        # Split the tensor
        q_tensor = tensor[:q_size]  # [q_size, rank]
        k_tensor = tensor[q_size : q_size + k_size]  # [k_size, rank]
        v_tensor = tensor[q_size + k_size : q_size + k_size + v_size]  # [v_size, rank]

        return q_tensor, k_tensor, v_tensor

    @staticmethod
    def merge_qkv_lora_a(q_tensor: torch.Tensor, k_tensor: torch.Tensor, v_tensor: torch.Tensor) -> torch.Tensor:
        """Merge Q, K, V LoRA A matrices into fused QKV format.

        Args:
            q_tensor: Q LoRA A matrix
            k_tensor: K LoRA A matrix
            v_tensor: V LoRA A matrix

        Returns:
            Fused QKV LoRA A matrix

        Raises:
            ValueError: If Q, K, V LoRA A matrices are not identical
        """
        # Strict mode: detect inequality and raise with guidance
        if not (torch.equal(q_tensor, k_tensor) and torch.equal(q_tensor, v_tensor)):
            raise ValueError(
                "Mismatched LoRA A matrices for Q/K/V projections. "
                "Fused linear_qkv requires shared A matrix across all projections. "
                "Consider using canonical LoRA layout for different A matrices per projection."
            )
        return q_tensor

    @staticmethod
    def merge_qkv_lora_b(q_tensor: torch.Tensor, k_tensor: torch.Tensor, v_tensor: torch.Tensor) -> torch.Tensor:
        """Merge Q, K, V LoRA B matrices into fused QKV format.

        Args:
            q_tensor: Q LoRA B matrix
            k_tensor: K LoRA B matrix
            v_tensor: V LoRA B matrix

        Returns:
            Fused QKV LoRA B matrix
        """
        # Concatenate along the output dimension (first dimension for weights)
        return torch.cat([q_tensor, k_tensor, v_tensor], dim=0)

    @staticmethod
    def split_fc1_lora_a(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split fused FC1 LoRA A matrix into gate and up components.

        Args:
            tensor: Fused FC1 LoRA A matrix [rank, hidden_size]

        Returns:
            Tuple of (gate_A, up_A) tensors
        """
        # For A matrices, input dimension is the same for gate and up
        return tensor, tensor

    @staticmethod
    def split_fc1_lora_b(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split fused FC1 LoRA B matrix into gate and up components.

        Args:
            tensor: Fused FC1 LoRA B matrix [ffn_hidden_size, rank]

        Returns:
            Tuple of (gate_B, up_B) tensors
        """
        # Split evenly between gate and up projections
        split_point = tensor.size(0) // 2
        gate_tensor = tensor[:split_point]  # [ffn_hidden_size/2, rank]
        up_tensor = tensor[split_point:]  # [ffn_hidden_size/2, rank]

        return gate_tensor, up_tensor

    @staticmethod
    def merge_fc1_lora_a(gate_tensor: torch.Tensor, up_tensor: torch.Tensor) -> torch.Tensor:
        """Merge gate and up LoRA A matrices into fused FC1 format.

        Args:
            gate_tensor: Gate LoRA A matrix
            up_tensor: Up LoRA A matrix

        Returns:
            Fused FC1 LoRA A matrix

        Raises:
            ValueError: If gate and up LoRA A matrices are not identical
        """
        # Strict mode: detect inequality and raise with guidance
        if not torch.equal(gate_tensor, up_tensor):
            raise ValueError(
                "Mismatched LoRA A matrices for gate/up projections. "
                "Fused linear_fc1 requires shared A matrix across gate and up projections. "
                "Consider using canonical LoRA layout for different A matrices per projection."
            )
        return gate_tensor

    @staticmethod
    def merge_fc1_lora_b(gate_tensor: torch.Tensor, up_tensor: torch.Tensor) -> torch.Tensor:
        """Merge gate and up LoRA B matrices into fused FC1 format.

        Args:
            gate_tensor: Gate LoRA B matrix
            up_tensor: Up LoRA B matrix

        Returns:
            Fused FC1 LoRA B matrix
        """
        # Concatenate along the output dimension
        return torch.cat([gate_tensor, up_tensor], dim=0)


def create_target_module_mapping() -> Dict[str, list[str]]:
    """Create mapping between Megatron and HuggingFace target module names.

    This mapping helps convert between the different naming conventions used
    by Megatron and HuggingFace for the same logical components.

    Returns:
        Dictionary mapping Megatron target names to HF target names
    """
    return {
        # Attention components
        "linear_q": ["q_proj"],
        "linear_k": ["k_proj"],
        "linear_v": ["v_proj"],
        "linear_qkv": ["q_proj", "k_proj", "v_proj"],
        "linear_proj": ["o_proj"],
        # MLP components
        "linear_fc1_up": ["up_proj"],
        "linear_fc1_gate": ["gate_proj"],
        "linear_fc1": ["up_proj", "gate_proj"],
        "linear_fc2": ["down_proj"],
    }


def infer_hf_target_modules(megatron_targets: list[str]) -> list[str]:
    """Infer HuggingFace target modules from Megatron target modules.

    Args:
        megatron_targets: List of Megatron target module names

    Returns:
        List of corresponding HuggingFace target module names
    """
    mapping = create_target_module_mapping()
    hf_targets = []

    for target in megatron_targets:
        if target in mapping:
            hf_targets.extend(mapping[target])
        else:
            # Pass through unknown targets as-is
            hf_targets.append(target)

    return list(set(hf_targets))  # Remove duplicates
