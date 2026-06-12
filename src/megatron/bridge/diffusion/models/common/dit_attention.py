# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=C0115,C0116,C0301

import copy
from dataclasses import dataclass
from typing import Union

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.extensions.transformer_engine import SplitAlongDim
from megatron.core.transformer.attention import (
    CrossAttention,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


def _mcore_cross_attention_accepts_head_wise_gate() -> bool:
    code = getattr(CrossAttention.get_query_key_value_tensors, "__code__", None)
    return code is not None and "head_wise_gate" in code.co_varnames


@dataclass
class DiTCrossAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a cross-attention.
    """

    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class DiTSelfAttention(SelfAttention):  # noqa: D101
    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        cp_comm_type: str = None,
        pg_collection=None,
        name: str | None = None,
    ):
        super().__init__(
            config,
            submodules,
            layer_number,
            attn_mask_type,
            cp_comm_type,
            pg_collection,
            name=name,
        )

        self.layernorm_across_heads = getattr(self.config, "layernorm_across_heads", False)

        # override q_layernorm
        if submodules.q_layernorm is not None:
            if self.layernorm_across_heads:
                q_layernorm_size = self.query_projection_size
            else:
                q_layernorm_size = self.hidden_size_per_attention_head
            norm_config = copy.deepcopy(self.config)
            norm_config.normalization = "RMSNorm"
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                eps=norm_config.layernorm_epsilon,
                hidden_size=q_layernorm_size,
                config=norm_config,
            )
        else:
            self.q_layernorm = None

        # override k_layernorm
        if submodules.k_layernorm is not None:
            if self.layernorm_across_heads:
                k_layernorm_size = self.kv_projection_size
            else:
                k_layernorm_size = self.hidden_size_per_attention_head
            norm_config = copy.deepcopy(self.config)
            norm_config.normalization = "RMSNorm"
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                eps=norm_config.layernorm_epsilon,
                hidden_size=k_layernorm_size,
                config=norm_config,
            )
        else:
            self.k_layernorm = None

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, output_gate=None, head_wise_gate=False, split_qkv=True
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        if head_wise_gate:
            # TODO: remove this guard when DiT attention supports MCore head_wise_attn_gate rows.
            raise NotImplementedError("DiTSelfAttention does not support head_wise_gate.")

        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:
            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        # gather query and key heads across TP ranks if self.layernorm_across_heads is True
        if self.layernorm_across_heads and parallel_state.get_tensor_model_parallel_world_size() > 1:
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)
            query = tensor_parallel.gather_from_tensor_model_parallel_region(query)
            key = tensor_parallel.gather_from_tensor_model_parallel_region(key)
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)

        if self.q_layernorm is not None:
            if self.layernorm_across_heads:
                q_flat = query.reshape(query.size(0), query.size(1), -1).contiguous()  # [sq, b, np*hn]
                q_flat = self.q_layernorm(q_flat)
                query = q_flat.view(
                    query.size(0), query.size(1), -1, self.hidden_size_per_attention_head
                )  # [sq, b, np, hn]
            else:
                query = self.q_layernorm(query.contiguous())

        if self.k_layernorm is not None:
            if self.layernorm_across_heads:
                k_flat = key.reshape(key.size(0), key.size(1), -1).contiguous()
                k_flat = self.k_layernorm(k_flat)
                key = k_flat.view(key.size(0), key.size(1), -1, self.hidden_size_per_attention_head)
            else:
                key = self.k_layernorm(key.contiguous())

        # scatter query and key heads across TP ranks if self.layernorm_across_heads is True
        if self.layernorm_across_heads and parallel_state.get_tensor_model_parallel_world_size() > 1:
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)
            query = tensor_parallel.scatter_to_tensor_model_parallel_region(query)
            key = tensor_parallel.scatter_to_tensor_model_parallel_region(key)
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)
            query = query.contiguous()  # important becuase TE attention expects contiguous tensors
            key = key.contiguous()  # important becuase TE attention expects contiguous tensors

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value


class DiTCrossAttention(CrossAttention):  # noqa: D101
    def __init__(
        self,
        config: TransformerConfig,
        submodules: DiTCrossAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        cp_comm_type: str = None,
        pg_collection=None,
        name: str | None = None,
    ):
        super().__init__(
            config,
            submodules,
            layer_number,
            attn_mask_type,
            cp_comm_type,
            pg_collection,
            name=name,
        )

        self.layernorm_across_heads = getattr(self.config, "layernorm_across_heads", False)

        # override q_layernorm
        if submodules.q_layernorm is not None:
            if self.layernorm_across_heads:
                q_layernorm_size = self.query_projection_size
            else:
                q_layernorm_size = self.hidden_size_per_attention_head
            norm_config = copy.deepcopy(self.config)
            norm_config.normalization = "RMSNorm"
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                eps=norm_config.layernorm_epsilon,
                hidden_size=q_layernorm_size,
                config=norm_config,
            )
        else:
            self.q_layernorm = None

        # override k_layernorm
        if submodules.k_layernorm is not None:
            if self.layernorm_across_heads:
                k_layernorm_size = self.kv_projection_size
            else:
                k_layernorm_size = self.hidden_size_per_attention_head
            norm_config = copy.deepcopy(self.config)
            norm_config.normalization = "RMSNorm"
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                eps=norm_config.layernorm_epsilon,
                hidden_size=k_layernorm_size,
                config=norm_config,
            )
        else:
            self.k_layernorm = None

        linear_kv_hidden_size = getattr(self.config, "crossattn_emb_size", self.config.hidden_size)
        self.linear_kv = build_module(
            submodules.linear_kv,
            linear_kv_hidden_size,
            2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            name=(name + ".linear_kv") if name is not None else None,
        )

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states, output_gate=None, head_wise_gate=False, split_qkv=True
    ):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """

        kwargs = {"output_gate": output_gate, "split_qkv": split_qkv}
        if _mcore_cross_attention_accepts_head_wise_gate():
            kwargs["head_wise_gate"] = head_wise_gate
        elif head_wise_gate:
            # TODO: remove this guard when Megatron-Core main exposes the head_wise_gate keyword.
            raise NotImplementedError("DiTCrossAttention does not support head_wise_gate.")

        query, key, value = super().get_query_key_value_tensors(hidden_states, key_value_states, **kwargs)

        # gather query and key heads across TP ranks if self.layernorm_across_heads is True
        if self.layernorm_across_heads and parallel_state.get_tensor_model_parallel_world_size() > 1:
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)
            query = tensor_parallel.gather_from_tensor_model_parallel_region(query)
            key = tensor_parallel.gather_from_tensor_model_parallel_region(key)
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)

        if self.q_layernorm is not None:
            if self.layernorm_across_heads:
                q_flat = query.reshape(query.size(0), query.size(1), -1).contiguous()  # [sq, b, np*hn]
                q_flat = self.q_layernorm(q_flat)
                query = q_flat.view(
                    query.size(0), query.size(1), -1, self.hidden_size_per_attention_head
                )  # [sq, b, np, hn]
            else:
                query = self.q_layernorm(query.contiguous())

        if self.k_layernorm is not None:
            if self.layernorm_across_heads:
                k_flat = key.reshape(key.size(0), key.size(1), -1).contiguous()
                k_flat = self.k_layernorm(k_flat)
                key = k_flat.view(key.size(0), key.size(1), -1, self.hidden_size_per_attention_head)
            else:
                key = self.k_layernorm(key.contiguous())

        # scatter query and key heads across TP ranks if self.layernorm_across_heads is True
        if self.layernorm_across_heads and parallel_state.get_tensor_model_parallel_world_size() > 1:
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)
            query = tensor_parallel.scatter_to_tensor_model_parallel_region(query)
            key = tensor_parallel.scatter_to_tensor_model_parallel_region(key)
            query = query.transpose(-2, -1)
            key = key.transpose(-2, -1)
            query = query.contiguous()  # important becuase TE attention expects contiguous tensors
            key = key.contiguous()  # important becuase TE attention expects contiguous tensors

        return query, key, value
