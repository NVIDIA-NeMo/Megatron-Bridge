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

from typing import Dict, Optional

import torch
import torch.nn as nn

from megatron.bridge.models.param_mapping import MegatronParamMapping


class BaichuanQKVMapping(MegatronParamMapping[torch.Tensor]):
    """
    Custom mapping for Baichuan's W_pack QKV format.

    Baichuan uses a unique packing format where Q, K, V are concatenated
    along dimension 0 in a specific order, different from the standard
    Megatron QKV interleaving.

    HF format (W_pack):
        - Shape: [3 * hidden_size, hidden_size]
        - Layout: [Q_full, K_full, V_full] concatenated

    Megatron format:
        - Interleaved per head group for GQA efficiency
        - Layout: [q1...qn, k1, v1, q1...qn, k2, v2, ...]
    """

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Convert Baichuan W_pack format to Megatron interleaved QKV format."""
        # Get dimensions from the module's config
        config = megatron_module.config
        head_num = config.num_attention_heads
        num_query_groups = config.num_query_groups
        heads_per_group = head_num // num_query_groups
        hidden_size = config.hidden_size
        head_size = config.kv_channels

        # Unpack the W_pack tensor
        # W_pack shape: [3 * hidden_size, hidden_size]
        qkv_weights = hf_weights.unflatten(0, (3, hidden_size))

        # Extract Q, K, V
        # Each has shape [hidden_size, hidden_size]
        q_weight = qkv_weights[0]  # [hidden_size, hidden_size]
        k_weight = qkv_weights[1]  # [hidden_size, hidden_size]
        v_weight = qkv_weights[2]  # [hidden_size, hidden_size]

        # Reshape to separate heads
        # Q: [hidden_size, hidden_size] -> [num_heads, head_size, hidden_size]
        q_weight = q_weight.view(head_num, head_size, hidden_size)
        # K, V: [hidden_size, hidden_size] -> [num_query_groups, head_size, hidden_size]
        k_weight = k_weight.view(num_query_groups, head_size, hidden_size)
        v_weight = v_weight.view(num_query_groups, head_size, hidden_size)

        # Interleave according to Megatron format
        qkv_interleaved = []
        for i in range(num_query_groups):
            # Add Q heads for this group
            start_idx = i * heads_per_group
            end_idx = (i + 1) * heads_per_group
            qkv_interleaved.append(q_weight[start_idx:end_idx])
            # Add K for this group
            qkv_interleaved.append(k_weight[i : i + 1])
            # Add V for this group
            qkv_interleaved.append(v_weight[i : i + 1])

        # Concatenate all interleaved weights
        qkv_weight = torch.cat(qkv_interleaved, dim=0)
        # Reshape to final format
        qkv_weight = qkv_weight.reshape(head_size * (head_num + 2 * num_query_groups), hidden_size)

        # Handle tensor parallel if needed
        if self.tp_size > 1:
            # Distribute across TP ranks
            return self._distribute_qkv_tp(qkv_weight, megatron_module)

        return qkv_weight

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Convert Megatron interleaved QKV format to Baichuan W_pack format."""
        # Handle cross-PP broadcast
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights)

        if megatron_weights is None:
            return {}

        # Get dimensions
        config = megatron_module.config if megatron_module else None
        if config is None:
            return {}

        head_num = config.num_attention_heads
        num_query_groups = config.num_query_groups
        heads_per_group = head_num // num_query_groups
        hidden_size = config.hidden_size
        head_size = config.kv_channels
        qkv_total_dim = head_num + 2 * num_query_groups

        # Gather from TP ranks if needed
        if self.tp_size > 1:
            megatron_weights = self._gather_qkv_tp(megatron_weights)
            if self.tp_rank != 0:
                return {}

        # Reshape to separate heads
        # [qkv_total_dim * head_size, hidden_size] -> [qkv_total_dim, head_size, hidden_size]
        qkv_weight = megatron_weights.reshape(qkv_total_dim, head_size, hidden_size)

        # Extract Q, K, V from interleaved format
        q_weights = []
        k_weights = []
        v_weights = []

        idx = 0
        for i in range(num_query_groups):
            # Extract Q heads for this group
            q_weights.append(qkv_weight[idx : idx + heads_per_group])
            idx += heads_per_group
            # Extract K for this group
            k_weights.append(qkv_weight[idx : idx + 1])
            idx += 1
            # Extract V for this group
            v_weights.append(qkv_weight[idx : idx + 1])
            idx += 1

        # Concatenate Q, K, V separately
        q_weight = torch.cat(q_weights, dim=0).reshape(hidden_size, hidden_size)
        k_weight = torch.cat(k_weights, dim=0).reshape(hidden_size, hidden_size)
        v_weight = torch.cat(v_weights, dim=0).reshape(hidden_size, hidden_size)

        # Pack into W_pack format [3 * hidden_size, hidden_size]
        w_pack = torch.cat([q_weight, k_weight, v_weight], dim=0)

        return {str(self.hf_param): w_pack}

    def _distribute_qkv_tp(self, qkv_weight: torch.Tensor, module: nn.Module) -> torch.Tensor:
        """Distribute QKV weights across tensor parallel ranks."""
        # For column parallel, we split along dim 0
        if self.tp_rank == 0:
            splits = torch.chunk(qkv_weight, self.tp_size, dim=0)
        else:
            splits = None

        output_shape = module.weight.shape
        return self.scatter_to_tp_ranks(
            splits,
            output_shape,
            module.weight.dtype,
            module.weight.device,
        )

    def _gather_qkv_tp(self, local_weight: torch.Tensor) -> torch.Tensor:
        """Gather QKV weights from all tensor parallel ranks."""
        gathered = self.gather_from_tp_ranks(local_weight)
        return torch.cat(gathered, dim=0)
