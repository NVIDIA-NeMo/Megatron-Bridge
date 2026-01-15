# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import logging
import torch
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_model import FalconH1Model
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    ColumnParallelMapping,
    MambaConv1dMapping,
    MambaInProjMapping,
    QKVMapping,
    RowParallelMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.falcon_h1.falconh1_provider import FalconH1ModelProvider

logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source="FalconH1ForCausalLM", target=FalconH1Model)
class FalconH1Bridge(MegatronModelBridge):
    """
    Megatron Bridge for FalconH1 Causal LM.

    Handles conversion between HuggingFace FalconH1ForCausalLM and
    Megatron FalconH1Model formats, including weight mappings and
    configuration translation.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("tiiuae/Falcon-H1-7B-Instruct", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> FalconH1ModelProvider:
        hf_config = hf_pretrained.config

        return FalconH1ModelProvider(
            # Basic model dimensions
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            seq_length=hf_config.max_position_embeddings,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            kv_channels=getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),

            # Mamba-specific parameters
            mamba_state_dim=hf_config.mamba_d_state,
            mamba_head_dim=hf_config.mamba_d_head,
            mamba_num_heads=hf_config.mamba_n_heads,
            mamba_num_groups=hf_config.mamba_n_groups,
            expand=hf_config.mamba_expand,
            d_conv=hf_config.mamba_d_conv,
            chunk_size=hf_config.mamba_chunk_size,
            rmsnorm=hf_config.mamba_rms_norm,

            # Model configuration
            vocab_size=hf_config.vocab_size,
            layernorm_epsilon=hf_config.rms_norm_eps,

            # Position embeddings
            position_embedding_type="rope",
            rotary_base=int(hf_config.rope_theta),

            # Weights and biases
            share_embeddings_and_output_weights=hf_config.tie_word_embeddings,
            add_bias_linear=hf_config.projectors_bias,
            attention_dropout=hf_config.attention_dropout,
            hidden_dropout=getattr(hf_config, "hidden_dropout", 0.0),

            # Data types
            params_dtype=self.dtype_from_hf(hf_config, default=torch.bfloat16),
            fp16=(self.dtype_from_hf(hf_config, default=torch.bfloat16) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.bfloat16) == torch.bfloat16),

            # FalconH1 specific - uniform hybrid layers
            falconh1_ratio=1.0,
            use_mamba=True,
            use_attention=True,
            use_mlp=True,

            # Make vocab size divisible
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            # Add all MuP multipliers from HF config
            embedding_multiplier=getattr(hf_config, 'embedding_multiplier', 1.0),
            lm_head_multiplier=getattr(hf_config, 'lm_head_multiplier', 1.0),
            key_multiplier=getattr(hf_config, 'key_multiplier', 1.0),
            attention_in_multiplier=getattr(hf_config, 'attention_in_multiplier', 1.0),
            attention_out_multiplier=getattr(hf_config, 'attention_out_multiplier', 1.0),
            ssm_in_multiplier=getattr(hf_config, 'ssm_in_multiplier', 1.0),
            ssm_out_multiplier=getattr(hf_config, 'ssm_out_multiplier', 1.0),
            mlp_multipliers=tuple(getattr(hf_config, 'mlp_multipliers', [1.0, 1.0])),
            ssm_multipliers=tuple(getattr(hf_config, 'ssm_multipliers', [1.0, 1.0, 1.0, 1.0, 1.0])),
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Define parameter mappings between Megatron and HuggingFace formats."""

        # Simple 1:1 parameter mappings
        param_mappings = {
            # MLP mappings (FalconH1 uses gate_proj/up_proj combined)
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.feed_forward.down_proj.weight",

            # Attention output projection
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",

            # Layer norms for TELayerNormColumnParallelLinear layers
            "decoder.layers.*.mamba_mixer.in_proj.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.pre_ff_layernorm.weight",
            "decoder.final_norm.weight": "model.final_layernorm.weight",

            # Embeddings and output
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
        }

        mapping_list = []

        # Convert simple mappings to AutoMapping objects
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Mamba mixer components with proper tensor parallel handling
        for mamba_component in ["A_log", "D", "dt_bias", "norm.weight"]:
            mapping_list.append(
                ColumnParallelMapping(
                    megatron_param=f"decoder.layers.*.mamba_mixer.{mamba_component}",
                    hf_param=f"model.layers.*.mamba.{mamba_component}",
                )
            )

        # Mamba output projection (row parallel)
        mapping_list.append(
            RowParallelMapping(
                megatron_param="decoder.layers.*.mamba_mixer.out_proj.weight",
                hf_param="model.layers.*.mamba.out_proj.weight",
            )
        )

        # Mamba input projection with special handling
        mapping_list.append(
            MambaInProjMapping(
                megatron_param="decoder.layers.*.mamba_mixer.in_proj.weight",
                hf_param="model.layers.*.mamba.in_proj.weight",
            )
        )

        # Mamba conv1d components
        for conv_component in ["weight", "bias"]:
            mapping_list.append(
                MambaConv1dMapping(
                    megatron_param=f"decoder.layers.*.mamba_mixer.conv1d.{conv_component}",
                    hf_param=f"model.layers.*.mamba.conv1d.{conv_component}",
                )
            )

        # QKV mapping - combine separate Q, K, V into single QKV matrix
        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            )
        )

        # Handle up_proj separately if needed (FalconH1 might combine gate and up)
        mapping_list.append(
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1_up.weight",
                hf_param="model.layers.*.feed_forward.up_proj.weight",
            )
        )

        # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
        mapping_list.append(
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.feed_forward.gate_proj.weight",
                up="model.layers.*.feed_forward.up_proj.weight",
            )
        )

        return MegatronMappingRegistry(*mapping_list)