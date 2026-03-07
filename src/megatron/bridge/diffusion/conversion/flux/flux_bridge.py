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

from typing import Mapping

import torch
from diffusers import FluxTransformer2DModel

from megatron.bridge.diffusion.conversion.flux.flux_hf_pretrained import PreTrainedFlux
from megatron.bridge.diffusion.models.flux.flux_model import Flux
from megatron.bridge.diffusion.models.flux.flux_provider import FluxProvider
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    QKVMapping,
    RowParallelMapping,
)


@MegatronModelBridge.register_bridge(source=FluxTransformer2DModel, target=Flux)
class FluxBridge(MegatronModelBridge):
    """
    Megatron Bridge for FLUX model.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("black-forest-labs/FLUX.1-dev")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedFlux) -> FluxProvider:
        hf_config = hf_pretrained.config

        provider = FluxProvider(
            in_channels=hf_config.in_channels,
            patch_size=hf_config.patch_size,
            num_joint_layers=hf_config.num_layers,
            num_single_layers=hf_config.num_single_layers,
            num_attention_heads=hf_config.num_attention_heads,
            # out_channels: None
            # joint_attention_dim: 4096
            kv_channels=hf_config.attention_head_dim,
            num_query_groups=hf_config.num_attention_heads,
            vec_in_dim=hf_config.pooled_projection_dim,
            guidance_embed=hf_config.guidance_embeds,
            axes_dims_rope=hf_config.axes_dims_rope,
            bf16=False,
            params_dtype=torch.float32,
        )
        self.hidden_size = provider.hidden_size
        return provider

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Load weights from HuggingFace state dict.
        This function can be overridden by subclasses to preprocess the HF weights before conversion, such as renaming
        certain parameters to avoid mapping conflicts, or dequantize the weights.

        Note that loading is done lazily before this function is called, so the weights are actually loaded in
        this function when hf_state_dict.__getitem__ is called.

        Args:
            hf_param: The parameter name or dictionary of parameter names to load.
            hf_state_dict: The HuggingFace state dictionary.

        Returns:
            The loaded weights.
        """
        if isinstance(hf_param, str):
            if hf_param.endswith("weight_1"):
                hf_weights = hf_state_dict[hf_param.replace("weight_1", "weight")]
                hf_weights = hf_weights[:, self.hidden_size :]
            elif hf_param.endswith("weight_2"):
                hf_weights = hf_state_dict[hf_param.replace("weight_2", "weight")]
                hf_weights = hf_weights[:, : self.hidden_size]
            else:
                hf_weights = hf_state_dict[hf_param]
        else:
            hf_weights = {k: hf_state_dict[v] for k, v in hf_param.items()}
        return hf_weights

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return MegatronMappingRegistry containing parameter mappings from HF to Megatron format.

        Returns:
            MegatronMappingRegistry: Registry of parameter mappings
        """
        # Dictionary maps HF parameter names -> Megatron parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "norm_out.linear.bias": "norm_out.adaLN_modulation.1.bias",
            "norm_out.linear.weight": "norm_out.adaLN_modulation.1.weight",
            "proj_out.bias": "proj_out.bias",
            "proj_out.weight": "proj_out.weight",
            "time_text_embed.guidance_embedder.linear_1.bias": "guidance_embedding.in_layer.bias",
            "time_text_embed.guidance_embedder.linear_1.weight": "guidance_embedding.in_layer.weight",
            "time_text_embed.guidance_embedder.linear_2.bias": "guidance_embedding.out_layer.bias",
            "time_text_embed.guidance_embedder.linear_2.weight": "guidance_embedding.out_layer.weight",
            "x_embedder.bias": "img_embed.bias",
            "x_embedder.weight": "img_embed.weight",
            "context_embedder.bias": "txt_embed.bias",
            "context_embedder.weight": "txt_embed.weight",
            "time_text_embed.timestep_embedder.linear_1.bias": "timestep_embedding.time_embedder.in_layer.bias",
            "time_text_embed.timestep_embedder.linear_1.weight": "timestep_embedding.time_embedder.in_layer.weight",
            "time_text_embed.timestep_embedder.linear_2.bias": "timestep_embedding.time_embedder.out_layer.bias",
            "time_text_embed.timestep_embedder.linear_2.weight": "timestep_embedding.time_embedder.out_layer.weight",
            "time_text_embed.text_embedder.linear_1.bias": "vector_embedding.in_layer.bias",
            "time_text_embed.text_embedder.linear_1.weight": "vector_embedding.in_layer.weight",
            "time_text_embed.text_embedder.linear_2.bias": "vector_embedding.out_layer.bias",
            "time_text_embed.text_embedder.linear_2.weight": "vector_embedding.out_layer.weight",
            "transformer_blocks.*.norm1.linear.weight": "double_blocks.*.adaln.linear.weight",
            "transformer_blocks.*.norm1.linear.bias": "double_blocks.*.adaln.linear.bias",
            "transformer_blocks.*.norm1_context.linear.weight": "double_blocks.*.adaln_context.linear.weight",
            "transformer_blocks.*.norm1_context.linear.bias": "double_blocks.*.adaln_context.linear.bias",
            "transformer_blocks.*.attn.norm_q.weight": "double_blocks.*.self_attention.q_layernorm.weight",
            "transformer_blocks.*.attn.norm_k.weight": "double_blocks.*.self_attention.k_layernorm.weight",
            "transformer_blocks.*.attn.norm_added_q.weight": "double_blocks.*.self_attention.added_q_layernorm.weight",
            "transformer_blocks.*.attn.norm_added_k.weight": "double_blocks.*.self_attention.added_k_layernorm.weight",
            "transformer_blocks.*.attn.to_out.0.weight": "double_blocks.*.self_attention.linear_proj.weight",
            "transformer_blocks.*.attn.to_out.0.bias": "double_blocks.*.self_attention.linear_proj.bias",
            "transformer_blocks.*.attn.to_add_out.weight": "double_blocks.*.self_attention.added_linear_proj.weight",
            "transformer_blocks.*.attn.to_add_out.bias": "double_blocks.*.self_attention.added_linear_proj.bias",
            "transformer_blocks.*.ff.net.0.proj.weight": "double_blocks.*.mlp.linear_fc1.weight",
            "transformer_blocks.*.ff.net.0.proj.bias": "double_blocks.*.mlp.linear_fc1.bias",
            "transformer_blocks.*.ff.net.2.weight": "double_blocks.*.mlp.linear_fc2.weight",
            "transformer_blocks.*.ff.net.2.bias": "double_blocks.*.mlp.linear_fc2.bias",
            "transformer_blocks.*.ff_context.net.0.proj.weight": "double_blocks.*.context_mlp.linear_fc1.weight",
            "transformer_blocks.*.ff_context.net.0.proj.bias": "double_blocks.*.context_mlp.linear_fc1.bias",
            "transformer_blocks.*.ff_context.net.2.weight": "double_blocks.*.context_mlp.linear_fc2.weight",
            "transformer_blocks.*.ff_context.net.2.bias": "double_blocks.*.context_mlp.linear_fc2.bias",
            "single_transformer_blocks.*.norm.linear.weight": "single_blocks.*.adaln.linear.weight",
            "single_transformer_blocks.*.norm.linear.bias": "single_blocks.*.adaln.linear.bias",
            "single_transformer_blocks.*.proj_mlp.weight": "single_blocks.*.mlp.linear_fc1.weight",
            "single_transformer_blocks.*.proj_mlp.bias": "single_blocks.*.mlp.linear_fc1.bias",
            "single_transformer_blocks.*.attn.norm_q.weight": "single_blocks.*.self_attention.q_layernorm.weight",
            "single_transformer_blocks.*.attn.norm_k.weight": "single_blocks.*.self_attention.k_layernorm.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for hf_param, megatron_param in param_mappings.items():
            mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

        # Split proj_out into linear_fc2 and linear_proj
        # The proj_out weight is split between MLP output and attention projection
        # The proj_out bias is mapped to MLP output
        mapping_list.append(
            AutoMapping(
                hf_param="single_transformer_blocks.*.proj_out.bias",
                megatron_param="single_blocks.*.mlp.linear_fc2.bias",
            )
        )
        mapping_list.append(
            SplitRowParallelMapping(
                hf_param="single_transformer_blocks.*.proj_out.weight_1",
                megatron_param="single_blocks.*.mlp.linear_fc2.weight",
            )
        )
        mapping_list.append(
            SplitRowParallelMapping(
                hf_param="single_transformer_blocks.*.proj_out.weight_2",
                megatron_param="single_blocks.*.self_attention.linear_proj.weight",
            )
        )

        AutoMapping.register_module_type("Linear", "replicated")

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # Single blockQKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    q="single_transformer_blocks.*.attn.to_q.weight",
                    k="single_transformer_blocks.*.attn.to_k.weight",
                    v="single_transformer_blocks.*.attn.to_v.weight",
                    megatron_param="single_blocks.*.self_attention.linear_qkv.weight",
                ),
                # Single block QKV bias: Combine separate Q, K, V bias into single QKV bias
                QKVMapping(
                    q="single_transformer_blocks.*.attn.to_q.bias",
                    k="single_transformer_blocks.*.attn.to_k.bias",
                    v="single_transformer_blocks.*.attn.to_v.bias",
                    megatron_param="single_blocks.*.self_attention.linear_qkv.bias",
                ),
                # Double block Self-attention QKV weights
                QKVMapping(
                    q="transformer_blocks.*.attn.to_q.weight",
                    k="transformer_blocks.*.attn.to_k.weight",
                    v="transformer_blocks.*.attn.to_v.weight",
                    megatron_param="double_blocks.*.self_attention.linear_qkv.weight",
                ),
                # Double block Self-attention QKV bias
                QKVMapping(
                    q="transformer_blocks.*.attn.to_q.bias",
                    k="transformer_blocks.*.attn.to_k.bias",
                    v="transformer_blocks.*.attn.to_v.bias",
                    megatron_param="double_blocks.*.self_attention.linear_qkv.bias",
                ),
                # Double block Added (context) attention QKV weights
                QKVMapping(
                    q="transformer_blocks.*.attn.add_q_proj.weight",
                    k="transformer_blocks.*.attn.add_k_proj.weight",
                    v="transformer_blocks.*.attn.add_v_proj.weight",
                    megatron_param="double_blocks.*.self_attention.added_linear_qkv.weight",
                ),
                # Double block Added (context) attention QKV bias
                QKVMapping(
                    q="transformer_blocks.*.attn.add_q_proj.bias",
                    k="transformer_blocks.*.attn.add_k_proj.bias",
                    v="transformer_blocks.*.attn.add_v_proj.bias",
                    megatron_param="double_blocks.*.self_attention.added_linear_qkv.bias",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)


class SplitRowParallelMapping(RowParallelMapping):  # noqa: D101
    def __init__(self, megatron_param: str, hf_param: str):
        super().__init__(megatron_param, hf_param)
        self.allow_hf_name_mismatch = True
