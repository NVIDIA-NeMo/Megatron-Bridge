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
from typing import Optional, Union

import torch
import torch.nn as nn
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.jit import jit_fuser
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

from megatron.bridge.diffusion.models.common.dit_attention import (
    DiTCrossAttention,
    DiTCrossAttentionSubmodules,
    DiTSelfAttention,
)


@dataclass
class WanWithAdaLNSubmodules(TransformerLayerSubmodules):  # noqa: D101
    temporal_self_attention: Union[ModuleSpec, type] = IdentityOp
    full_self_attention: Union[ModuleSpec, type] = IdentityOp
    norm1: Union[ModuleSpec, type] = None
    norm3: Union[ModuleSpec, type] = None
    norm2: Union[ModuleSpec, type] = None


class WanAdaLN(MegatronModule):
    """
    Adaptive Layer Normalization Module for DiT.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, config.hidden_size) / config.hidden_size**0.5)

        setattr(self.modulation, "sequence_parallel", config.sequence_parallel)

    @jit_fuser
    def forward(self, timestep_emb):
        e = (self.modulation + timestep_emb).transpose(0, 1)
        e = e.chunk(6, dim=0)
        return e

    @jit_fuser
    def normalize_modulate(self, norm, hidden_states, shift, scale):
        return self.modulate(norm(hidden_states), shift, scale)

    @jit_fuser
    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    @jit_fuser
    def scale_add(self, residual, x, gate):
        return residual + gate * x


class WanLayerWithAdaLN(TransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    DiT with Adapative Layer Normalization.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        def _replace_no_cp_submodules(submodules):
            modified_submods = copy.deepcopy(submodules)
            modified_submods.cross_attention = IdentityOp
            return modified_submods

        # Replace any submodules that will have CP disabled and build them manually later after TransformerLayer init.
        # modified_submods = _replace_no_cp_submodules(submodules)
        super().__init__(
            config=config, submodules=submodules, layer_number=layer_number, hidden_dropout=hidden_dropout
        )

        # TODO (pmannan): Override Cross Attention to disable CP.
        # Disable TP Comm overlap as well. Not disabling will attempt re-use of buffer size same as
        #   Q and lead to incorrect tensor shapes.
        # if submodules.cross_attention != IdentityOp:
        #     cp_override_config = copy.deepcopy(config)
        #     cp_override_config.context_parallel_size = 1
        #     cp_override_config.tp_comm_overlap = False
        #     self.cross_attention = build_module(
        #         submodules.cross_attention,
        #         config=cp_override_config,
        #         layer_number=layer_number,
        #     )
        # else:
        #     self.cross_attention = None

        self.full_self_attention = build_module(
            submodules.full_self_attention,
            config=self.config,
            layer_number=layer_number,
        )

        self.adaLN = WanAdaLN(config=self.config)
        self.norm1 = build_module(
            submodules.norm1,
            normalized_shape=config.hidden_size,
            eps=config.layernorm_epsilon,
            elementwise_affine=False,
        )
        self.norm3 = build_module(
            submodules.norm3,
            normalized_shape=config.hidden_size,
            eps=config.layernorm_epsilon,
            elementwise_affine=True,
        )
        self.norm2 = build_module(
            submodules.norm2,
            normalized_shape=config.hidden_size,
            eps=config.layernorm_epsilon,
            elementwise_affine=False,
        )

        # set attributes "average_gradients_across_tp_domain" for nn.Parameter objects
        # this is used for gradient averaging across TP domain with sequence parallelism
        self._mark_trainable_params_for_tp_grad_avg([self.norm3, self.adaLN])

    def _mark_trainable_params_for_tp_grad_avg(self, modules: Optional[list] = None) -> None:
        """Mark selected modules' trainable parameters to average gradients across TP domain."""
        target_modules = modules if modules is not None else [self]
        for module in target_modules:
            for _name, param in module.named_parameters(recurse=True):
                if isinstance(param, nn.Parameter) and param.requires_grad:
                    setattr(param, "average_gradients_across_tp_domain", True)

    @jit_fuser
    def add_residual(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + residual

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        inference_context=None,
        rotary_pos_cos_sin=None,
        **kwargs,
    ):
        # the timestep embedding is stored in attention_mask argument
        timestep_emb = attention_mask
        rope_emb = rotary_pos_emb

        shift_full, scale_full, gate_full, shift_mlp, scale_mlp, gate_mlp = self.adaLN(timestep_emb)

        # ******************************************** full self attention *******************************************

        # adaLN with scale + shift + gate
        pre_full_attn_layernorm_output_ada = self.adaLN.normalize_modulate(
            self.norm1,
            hidden_states,
            shift=shift_full,
            scale=scale_full,
        )

        attention_output, bias = self.full_self_attention(
            pre_full_attn_layernorm_output_ada,
            attention_mask=None,
            rotary_pos_emb=rope_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params["self_attention"],
        )
        if bias is not None:
            attention_output = attention_output + bias

        hidden_states = self.adaLN.scale_add(residual=hidden_states, x=attention_output, gate=gate_full)

        # ******************************************** cross attention ******************************************************

        # TODO (pmannan): Disable CP for CrossAttention as KV context is small.
        # But needs better support for packed sequences and padding to ensure correct calculations
        # packed_seq_params['cross_attention'].cu_seqlens_q = torch.tensor(
        #     [0, hidden_states.shape[0]],
        #     device=packed_seq_params['cross_attention'].cu_seqlens_kv.device,
        #     dtype=torch.int32)
        attention_output, bias = self.cross_attention(
            self.norm3(hidden_states),
            attention_mask=context_mask,
            key_value_states=context,
            packed_seq_params=packed_seq_params["cross_attention"],
        )
        if bias is not None:
            attention_output = attention_output + bias

        hidden_states = self.add_residual(hidden_states, attention_output)

        # ******************************************** mlp ******************************************************

        pre_mlp_layernorm_output_ada = self.adaLN.normalize_modulate(
            self.norm2,
            hidden_states,
            shift=shift_mlp,
            scale=scale_mlp,
        )

        mlp_output, bias = self.mlp(pre_mlp_layernorm_output_ada)
        if bias is not None:
            mlp_output = mlp_output + bias

        hidden_states = self.adaLN.scale_add(residual=hidden_states, x=mlp_output, gate=gate_mlp)

        # TODO: Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor. ???
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        # output = hidden_states

        return output, context


def get_wan_block_with_transformer_engine_spec() -> ModuleSpec:  # noqa: D103
    params = {"attn_mask_type": AttnMaskType.padding}
    return ModuleSpec(
        module=WanLayerWithAdaLN,
        submodules=WanWithAdaLNSubmodules(
            norm1=nn.LayerNorm,
            norm3=nn.LayerNorm,
            norm2=nn.LayerNorm,
            full_self_attention=ModuleSpec(
                module=DiTSelfAttention,
                params=params,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            cross_attention=ModuleSpec(
                module=DiTCrossAttention,
                params=params,
                submodules=DiTCrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )
