# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_model import FalconH1Config
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.utils import log_single_rank
import logging
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP

logger = logging.getLogger(__name__)

@dataclass
class FalconH1Submodules:
    """
    Configuration class for specifying the submodules of FalconH1 hybrid mixer.
    Uses composition of existing Megatron components.
    """
    # Pre-norm layer (not needed with TELayerNormColumnParallelLinear)
    norm: Union[ModuleSpec, type] = IdentityOp

    # SSM component (MambaMixer)
    mamba_mixer: Union[ModuleSpec, type] = None

    # Attention component (SelfAttention)
    self_attention: Union[ModuleSpec, type] = None

    # Bias-dropout-add fusion
    falconh1_bda: Union[ModuleSpec, type] = IdentityOp

    # MLP component
    mlp: Union[ModuleSpec, type] = IdentityOp

    # Bias-dropout-add for MLP
    mlp_bda: Union[ModuleSpec, type] = IdentityOp


class FalconH1Layer(MegatronModule):
    """
    FalconH1 Hybrid Mixer that combines SSM (Mamba), Attention, and MLP mechanisms.

    This implementation uses COMPOSITION of existing Megatron components:
    - MambaMixer for SSM functionality
    - SelfAttention for attention functionality
    - MLP for feed-forward functionality
    - Standard bias-dropout-add for residual connections

    The layer can flexibly enable/disable each component based on configuration.
    """

    def __init__(
        self,
        config: FalconH1Config,
        submodules: FalconH1Submodules,
        layer_number: int,
        residual_in_fp32: bool = False,
        # Control which components are active
        use_mamba: bool = True,
        use_attention: bool = True,
        use_mlp: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        # SSM specific arguments (passed to MambaMixer)
        d_conv: int = 4,
        conv_init: Optional[float] = 1.0,
        expand: int = 1,
        A_init_range: Tuple[float, float] = (1, 16),
        D_has_hdim: bool = False,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        chunk_size: int = 128,
        # Attention specific arguments
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_dropout: float = 0.0,
    ):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.residual_in_fp32 = residual_in_fp32
        self.use_mamba = use_mamba
        self.use_attention = use_attention
        self.use_mlp = use_mlp

        # Get model communication process groups
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        # Hidden dropout for BDA
        self.hidden_dropout = config.hidden_dropout

        # Pre-normalization layer
        self.norm = build_module(
            submodules.norm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # SSM Component:MambaMixer
        if self.use_mamba:
            mamba_submodules = MambaMixerSubmodules(
                in_proj=submodules.mamba_mixer.submodules.in_proj,
                out_proj=submodules.mamba_mixer.submodules.out_proj,
            )

            self.mamba_mixer = build_module(
                submodules.mamba_mixer.module,  # Should be MambaMixer
                submodules=mamba_submodules,
                config=self.config,
                d_model=self.config.hidden_size,
                layer_number=self.layer_number,
                d_conv=d_conv,
                conv_init=conv_init,
                expand=expand,
                A_init_range=A_init_range,
                D_has_hdim=D_has_hdim,
                rmsnorm=rmsnorm,
                norm_before_gate=norm_before_gate,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                bias=bias,
                conv_bias=conv_bias,
                chunk_size=chunk_size,
                pg_collection=pg_collection,
            )
        else:
            self.mamba_mixer = None

        # Attention Component:SelfAttention
        if self.use_attention:
            attention_optional_kwargs = {}
            if self.config.context_parallel_size > 1 and self.config.cp_comm_type is not None:
                if isinstance(self.config.cp_comm_type, list):
                    attention_optional_kwargs["cp_comm_type"] = self.config.cp_comm_type[self.layer_number]
                else:
                    attention_optional_kwargs["cp_comm_type"] = self.config.cp_comm_type
            attention_optional_kwargs["pg_collection"] = pg_collection

            # Submodules for SelfAttention
            attention_submodules = SelfAttentionSubmodules(
                linear_qkv=submodules.self_attention.submodules.linear_qkv,
                core_attention=submodules.self_attention.submodules.core_attention,
                linear_proj=submodules.self_attention.submodules.linear_proj,
                q_layernorm=getattr(submodules.self_attention.submodules, 'q_layernorm', None),
                k_layernorm=getattr(submodules.self_attention.submodules, 'k_layernorm', None),
            )

            self.self_attention = build_module(
                submodules.self_attention.module,
                submodules=attention_submodules,
                config=self.config,
                layer_number=self.layer_number,
                attn_mask_type=attn_mask_type,
                **attention_optional_kwargs,
            )
        else:
            self.self_attention = None

        # Bias-Dropout-Add fusion for Mamba/Attention
        self.falconh1_bda = build_module(submodules.falconh1_bda)

        #MLP Component
        if self.use_mlp:
            additional_mlp_kwargs = {}

            from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
            from megatron.core.transformer.moe.moe_layer import MoELayer

            if isinstance(submodules.mlp, ModuleSpec):
                if submodules.mlp.module in (MoELayer, GroupedMLP, TEGroupedMLP, SequentialMLP):
                    additional_mlp_kwargs["pg_collection"] = pg_collection
                elif submodules.mlp.module == MLP:
                    assert hasattr(
                        pg_collection, 'tp'
                    ), 'TP process group is required for MLP in FalconH1Layer'
                    additional_mlp_kwargs["tp_group"] = pg_collection.tp
                else:
                    log_single_rank(
                        logger,
                        logging.WARNING,
                        f"Unknown MLP type: {type(submodules.mlp)}. Using default kwargs.",
                    )

            # Build the MLP module
            self.mlp = build_module(
                submodules.mlp,
                config=self.config,
                **additional_mlp_kwargs
            )

            # Set layer number if MLP supports it
            if hasattr(self.mlp, 'set_layer_number'):
                self.mlp.set_layer_number(self.layer_number)

            # Build MLP bias-dropout-add
            self.mlp_bda = build_module(submodules.mlp_bda)
        else:
            self.mlp = None
            self.mlp_bda = None

        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        Forward pass through the hybrid mixer using COMPOSITION.

        Flow:
        1. Mamba and/or Attention (parallel or sequential based on configuration)
        2. MLP (if enabled)

        Pure orchestration - no inline reimplementation!
        """

        residual = hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        hidden_states = self.norm(hidden_states)

        outputs = []
        biases = []

        # SSM Forward: MambaMixer
        if self.use_mamba and self.mamba_mixer is not None:
            mamba_output, mamba_bias = self.mamba_mixer(
                hidden_states * self.config.ssm_in_multiplier,
                inference_context=inference_context,
            )
            mamba_output = mamba_output * self.config.ssm_out_multiplier
            outputs.append(mamba_output)
            if mamba_bias is not None:
                biases.append(mamba_bias)

        # Attention Forward: SelfAttention
        if self.use_attention and self.self_attention is not None:
            attn_output, attn_bias = self.self_attention(
                hidden_states * self.config.attention_in_multiplier,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
            attn_output = self.config.attention_out_multiplier
            outputs.append(attn_output)
            if attn_bias is not None:
                biases.append(attn_bias)

        if len(outputs) == 0:
            # Fallback to identity if no components active
            combined_output = hidden_states
            combined_bias = None
        elif len(outputs) == 1:
            # Single component active
            combined_output = outputs[0]
            combined_bias = biases[0] if biases else None
        else:
            # Multiple components - add them
            combined_output = sum(outputs)
            combined_bias = sum(biases) if biases else None

        # Apply bias-dropout-add fusion for Mamba/Attention (residual connection)
        out_with_bias = (combined_output, combined_bias)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.falconh1_bda(
                training=self.training,
                fused=self.config.bias_dropout_fusion
            )(out_with_bias, residual, self.hidden_dropout)

        if self.use_mlp and self.mlp is not None:
            #residual for MLP
            mlp_residual = hidden_states
            if self.residual_in_fp32:
                mlp_residual = mlp_residual.to(torch.float32)

            mlp_output_with_bias = self.mlp(hidden_states)

            # MLP bias-dropout-add
            with self.bias_dropout_add_exec_handler():
                hidden_states = self.mlp_bda(
                    training=self.training,
                    fused=self.config.bias_dropout_fusion
                )(mlp_output_with_bias, mlp_residual, self.hidden_dropout)

        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """Allocate inference cache for active components."""
        caches = {}

        if self.use_mamba and self.mamba_mixer is not None:
            mamba_cache = self.mamba_mixer.allocate_inference_cache(
                batch_size, max_seqlen, dtype
            )
            caches['mamba'] = mamba_cache

        if self.use_attention and self.self_attention is not None:
            # TODO: Implement attention cache allocation if needed
            pass

        return caches

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        from megatron.core.transformer.utils import sharded_state_dict_default

        sharded_state_dict = {}
        #norm
        if hasattr(self, 'norm') and self.norm is not None:
            norm_sd = sharded_state_dict_default(
                self.norm, f'{prefix}norm.', sharded_offsets, metadata
            )
            sharded_state_dict.update(norm_sd)

        # SSM component
        if self.use_mamba and hasattr(self, 'mamba_mixer') and self.mamba_mixer is not None:
            mamba_sd = sharded_state_dict_default(
                self.mamba_mixer, f'{prefix}mamba_mixer.', sharded_offsets, metadata
            )
            sharded_state_dict.update(mamba_sd)

        # attention component
        if self.use_attention and hasattr(self, 'self_attention') and self.self_attention is not None:
            attn_sd = sharded_state_dict_default(
                self.self_attention, f'{prefix}self_attention.', sharded_offsets, metadata
            )
            sharded_state_dict.update(attn_sd)

        # MLP component
        if self.use_mlp and hasattr(self, 'mlp') and self.mlp is not None:
            mlp_sd = sharded_state_dict_default(
                self.mlp, f'{prefix}mlp.', sharded_offsets, metadata
            )
            sharded_state_dict.update(mlp_sd)

        return sharded_state_dict