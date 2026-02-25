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

"""FLUX layer specifications and transformer blocks."""

import copy

import torch
import torch.nn as nn
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from megatron.bridge.diffusion.models.common.normalization import RMSNorm
from megatron.bridge.diffusion.models.flux.flux_attention import (
    FluxSingleAttention,
    JointSelfAttention,
    JointSelfAttentionSubmodules,
)


try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
        TERowParallelLinear,
    )
except ImportError:
    TEColumnParallelLinear = None
    TEDotProductAttention = None
    TENorm = None
    TERowParallelLinear = None

try:
    from megatron.core.transformer.cuda_graphs import CudaGraphManager
except ImportError:
    CudaGraphManager = None


class AdaLN(MegatronModule):
    """
    Adaptive Layer Normalization Module for DiT/FLUX models.

    Implements adaptive layer normalization that conditions on timestep embeddings.

    Args:
        config: Transformer configuration.
        n_adaln_chunks: Number of adaptive LN chunks for modulation outputs.
        norm: Normalization type to use.
        modulation_bias: Whether to use bias in modulation layers.
        use_second_norm: Whether to use a second layer norm.
    """

    def __init__(
        self,
        config: TransformerConfig,
        n_adaln_chunks: int = 9,
        norm=nn.LayerNorm,
        modulation_bias: bool = False,
        use_second_norm: bool = False,
    ):
        super().__init__(config)
        if norm == TENorm:
            self.ln = norm(config, config.hidden_size, config.layernorm_epsilon)
        else:
            self.ln = norm(config.hidden_size, elementwise_affine=False, eps=self.config.layernorm_epsilon)
        self.n_adaln_chunks = n_adaln_chunks
        self.activation = nn.SiLU()
        self.linear = ColumnParallelLinear(
            config.hidden_size,
            self.n_adaln_chunks * config.hidden_size,
            config=config,
            init_method=nn.init.normal_,
            bias=modulation_bias,
            gather_output=True,
        )
        self.use_second_norm = use_second_norm
        if self.use_second_norm:
            self.ln2 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        nn.init.constant_(self.linear.weight, 0)

        setattr(self.linear.weight, "sequence_parallel", config.sequence_parallel)

    @jit_fuser
    def forward(self, timestep_emb):
        """Apply adaptive layer normalization modulation."""
        output = self.activation(timestep_emb)
        output, bias = self.linear(output)
        output = output + bias if bias is not None else output
        return output.chunk(self.n_adaln_chunks, dim=-1)

    @jit_fuser
    def modulate(self, x, shift, scale):
        """Apply modulation with shift and scale."""
        return x * (1 + scale) + shift

    @jit_fuser
    def scale_add(self, residual, x, gate):
        """Add gated output to residual."""
        return residual + gate * x

    @jit_fuser
    def modulated_layernorm(self, x, shift, scale, layernorm_idx=0):
        """Apply layer norm followed by modulation."""
        if self.use_second_norm and layernorm_idx == 1:
            layernorm = self.ln2
        else:
            layernorm = self.ln
        # Optional Input Layer norm
        input_layernorm_output = layernorm(x).type_as(x)

        # DiT block specific
        return self.modulate(input_layernorm_output, shift, scale)

    @jit_fuser
    def scaled_modulated_layernorm(self, residual, x, gate, shift, scale, layernorm_idx=0):
        """Apply scale, add, and modulated layer norm."""
        hidden_states = self.scale_add(residual, x, gate)
        shifted_pre_mlp_layernorm_output = self.modulated_layernorm(hidden_states, shift, scale, layernorm_idx)
        return hidden_states, shifted_pre_mlp_layernorm_output


class AdaLNContinuous(MegatronModule):
    """
    A variant of AdaLN used for FLUX models.

    Continuous adaptive layer normalization that outputs scale and shift
    directly from conditioning embeddings.

    Args:
        config: Transformer configuration.
        conditioning_embedding_dim: Dimension of the conditioning embedding.
        modulation_bias: Whether to use bias in modulation layer.
        norm_type: Type of normalization ("layer_norm" or "rms_norm").
    """

    def __init__(
        self,
        config: TransformerConfig,
        conditioning_embedding_dim: int,
        modulation_bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__(config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(conditioning_embedding_dim, config.hidden_size * 2, bias=modulation_bias)
        )
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6, bias=modulation_bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(config.hidden_size, eps=1e-6)
        else:
            raise ValueError(f"Unknown normalization type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        """Apply continuous adaptive layer normalization."""
        emb = self.adaLN_modulation(conditioning_embedding)
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class MMDiTLayer(TransformerLayer):
    """
    Multi-modal transformer layer for FLUX double blocks.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    MMDiT layer implementation from [https://arxiv.org/pdf/2403.03206].

    Args:
        config: Transformer configuration.
        submodules: Transformer layer submodules.
        layer_number: Layer index.
        context_pre_only: Whether to only use context for pre-processing.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        context_pre_only: bool = False,
    ):
        hidden_size = config.hidden_size
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)

        # Enable per-Transformer layer cuda graph
        if CudaGraphManager is not None and config.enable_cuda_graph and config.cuda_graph_scope != "full_iteration":
            self.cudagraph_manager = CudaGraphManager(config, share_cudagraph_io_buffers=False)

        self.adaln = AdaLN(config, modulation_bias=True, n_adaln_chunks=6, use_second_norm=True)

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continuous" if context_pre_only else "ada_norm_zero"

        if context_norm_type == "ada_norm_continuous":
            self.adaln_context = AdaLNContinuous(config, hidden_size, modulation_bias=True, norm_type="layer_norm")
        elif context_norm_type == "ada_norm_zero":
            self.adaln_context = AdaLN(config, modulation_bias=True, n_adaln_chunks=6, use_second_norm=True)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, "
                f"currently only support `ada_norm_continous`, `ada_norm_zero`"
            )

        # Override config for context MLP to disable CP.
        # Disable TP Comm overlap as well.
        cp_override_config = copy.deepcopy(config)
        cp_override_config.context_parallel_size = 1
        cp_override_config.tp_comm_overlap = False

        if not context_pre_only:
            self.context_mlp = build_module(
                submodules.mlp,
                config=cp_override_config,
            )
        else:
            self.context_mlp = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
        emb=None,
    ):
        """
        Forward pass for MMDiT layer.

        Args:
            hidden_states: Image hidden states.
            encoder_hidden_states: Text encoder hidden states.
            attention_mask: Attention mask.
            context: Context tensor (unused).
            context_mask: Context mask (unused).
            rotary_pos_emb: Rotary position embeddings.
            inference_params: Inference parameters.
            packed_seq_params: Packed sequence parameters.
            emb: Timestep/conditioning embedding.

        Returns:
            Tuple of (hidden_states, encoder_hidden_states).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(emb)

        norm_hidden_states = self.adaln.modulated_layernorm(
            hidden_states, shift=shift_msa, scale=scale_msa, layernorm_idx=0
        )
        if self.context_pre_only:
            norm_encoder_hidden_states = self.adaln_context(encoder_hidden_states, emb)
        else:
            c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.adaln_context(emb)
            norm_encoder_hidden_states = self.adaln_context.modulated_layernorm(
                encoder_hidden_states, shift=c_shift_msa, scale=c_scale_msa, layernorm_idx=0
            )

        attention_output, encoder_attention_output = self.self_attention(
            norm_hidden_states,
            attention_mask=attention_mask,
            key_value_states=None,
            additional_hidden_states=norm_encoder_hidden_states,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = self.adaln.scale_add(hidden_states, x=attention_output, gate=gate_msa)
        norm_hidden_states = self.adaln.modulated_layernorm(
            hidden_states, shift=shift_mlp, scale=scale_mlp, layernorm_idx=1
        )

        mlp_output, mlp_output_bias = self.mlp(norm_hidden_states)
        hidden_states = self.adaln.scale_add(hidden_states, x=(mlp_output + mlp_output_bias), gate=gate_mlp)

        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            encoder_hidden_states = self.adaln_context.scale_add(
                encoder_hidden_states, x=encoder_attention_output, gate=c_gate_msa
            )
            norm_encoder_hidden_states = self.adaln_context.modulated_layernorm(
                encoder_hidden_states, shift=c_shift_mlp, scale=c_scale_mlp, layernorm_idx=1
            )

            context_mlp_output, context_mlp_output_bias = self.context_mlp(norm_encoder_hidden_states)
            encoder_hidden_states = self.adaln.scale_add(
                encoder_hidden_states, x=(context_mlp_output + context_mlp_output_bias), gate=c_gate_mlp
            )

        return hidden_states, encoder_hidden_states

    def __call__(self, *args, **kwargs):
        if hasattr(self, "cudagraph_manager"):
            return self.cudagraph_manager(self, args, kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)


class FluxSingleTransformerBlock(TransformerLayer):
    """
    FLUX Single Transformer Block.

    Single transformer layer mathematically equivalent to original Flux single transformer.
    This layer is re-implemented with megatron-core and altered in structure for better performance.

    Args:
        config: Transformer configuration.
        submodules: Transformer layer submodules.
        layer_number: Layer index.
        mlp_ratio: MLP hidden size ratio.
        n_adaln_chunks: Number of adaptive LN chunks.
        modulation_bias: Whether to use bias in modulation.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        mlp_ratio: int = 4,
        n_adaln_chunks: int = 3,
        modulation_bias: bool = True,
    ):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)

        # Enable per-Transformer layer cuda graph
        if CudaGraphManager is not None and config.enable_cuda_graph and config.cuda_graph_scope != "full_iteration":
            self.cudagraph_manager = CudaGraphManager(config, share_cudagraph_io_buffers=False)
        self.adaln = AdaLN(
            config=config, n_adaln_chunks=n_adaln_chunks, modulation_bias=modulation_bias, use_second_norm=False
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
        emb=None,
    ):
        """
        Forward pass for FLUX single transformer block.

        Args:
            hidden_states: Input hidden states.
            attention_mask: Attention mask.
            context: Context tensor (unused).
            context_mask: Context mask (unused).
            rotary_pos_emb: Rotary position embeddings.
            inference_params: Inference parameters.
            packed_seq_params: Packed sequence parameters.
            emb: Timestep/conditioning embedding.

        Returns:
            Tuple of (hidden_states, None).
        """
        residual = hidden_states

        shift, scale, gate = self.adaln(emb)

        norm_hidden_states = self.adaln.modulated_layernorm(hidden_states, shift=shift, scale=scale)

        mlp_hidden_states, mlp_bias = self.mlp(norm_hidden_states)

        attention_output = self.self_attention(
            norm_hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        hidden_states = mlp_hidden_states + mlp_bias + attention_output

        hidden_states = self.adaln.scale_add(residual, x=hidden_states, gate=gate)

        return hidden_states, None

    def __call__(self, *args, **kwargs):
        if hasattr(self, "cudagraph_manager"):
            return self.cudagraph_manager(self, args, kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)


# ============================================================================
# Layer Spec Functions
# ============================================================================


def get_flux_double_transformer_engine_spec() -> ModuleSpec:
    """
    Get the module specification for FLUX double transformer blocks.

    Returns:
        ModuleSpec for MMDiTLayer with JointSelfAttention.
    """
    return ModuleSpec(
        module=MMDiTLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=JointSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=JointSelfAttentionSubmodules(
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                    added_q_layernorm=TENorm,
                    added_k_layernorm=TENorm,
                    linear_qkv=TEColumnParallelLinear,
                    added_linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
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


def get_flux_single_transformer_engine_spec() -> ModuleSpec:
    """
    Get the module specification for FLUX single transformer blocks.

    Returns:
        ModuleSpec for FluxSingleTransformerBlock with FluxSingleAttention.
    """
    return ModuleSpec(
        module=FluxSingleTransformerBlock,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=FluxSingleAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                    linear_proj=TERowParallelLinear,
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
