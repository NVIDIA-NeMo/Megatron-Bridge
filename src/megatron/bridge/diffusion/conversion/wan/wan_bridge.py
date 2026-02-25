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

import torch
from diffusers import WanTransformer3DModel

from megatron.bridge.diffusion.conversion.wan.wan_hf_pretrained import PreTrainedWAN
from megatron.bridge.diffusion.models.wan.wan_model import WanModel
from megatron.bridge.diffusion.models.wan.wan_provider import WanModelProvider
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    KVMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name


@MegatronModelBridge.register_bridge(source=WanTransformer3DModel, target=WanModel)
class WanBridge(MegatronModelBridge):
    """
    Megatron Bridge for WAN model.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("WAN-3D-1.3B-v1")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedWAN) -> WanModelProvider:
        hf_config = hf_pretrained.config

        cls = WanModelProvider

        provider = cls(
            num_layers=hf_config.num_layers,
            hidden_size=hf_config.num_attention_heads * hf_config.attention_head_dim,
            kv_channels=hf_config.attention_head_dim,
            num_query_groups=hf_config.num_attention_heads,
            crossattn_emb_size=hf_config.num_attention_heads * hf_config.attention_head_dim,
            ffn_hidden_size=hf_config.ffn_dim,
            num_attention_heads=hf_config.num_attention_heads,
            in_channels=hf_config.in_channels,
            out_channels=hf_config.out_channels,
            text_dim=hf_config.text_dim,
            patch_spatial=hf_config.patch_size[1],
            patch_temporal=hf_config.patch_size[0],
            layernorm_epsilon=hf_config.eps,
            hidden_dropout=0,
            attention_dropout=0,
            use_cpu_initialization=True,
            freq_dim=hf_config.freq_dim,
            bf16=False,
            params_dtype=torch.float32,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return MegatronMappingRegistry containing parameter mappings from HF to Megatron format.

        Returns:
            MegatronMappingRegistry: Registry of parameter mappings
        """
        # Dictionary maps HF parameter names -> Megatron parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "scale_shift_table": "head.modulation",
            "patch_embedding.weight": "patch_embedding.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedder.linear_1.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedder.linear_1.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedder.linear_2.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedder.linear_2.bias",
            "condition_embedder.time_proj.weight": "time_proj.weight",
            "condition_embedder.time_proj.bias": "time_proj.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "blocks.*.scale_shift_table": "decoder.layers.*.adaLN.modulation",
            "blocks.*.attn1.to_out.0.weight": "decoder.layers.*.full_self_attention.linear_proj.weight",
            "blocks.*.attn1.to_out.0.bias": "decoder.layers.*.full_self_attention.linear_proj.bias",
            "blocks.*.attn1.norm_q.weight": "decoder.layers.*.full_self_attention.q_layernorm.weight",
            "blocks.*.attn1.norm_k.weight": "decoder.layers.*.full_self_attention.k_layernorm.weight",
            "blocks.*.attn2.to_q.weight": "decoder.layers.*.cross_attention.linear_q.weight",
            "blocks.*.attn2.to_q.bias": "decoder.layers.*.cross_attention.linear_q.bias",
            "blocks.*.attn2.to_out.0.weight": "decoder.layers.*.cross_attention.linear_proj.weight",
            "blocks.*.attn2.to_out.0.bias": "decoder.layers.*.cross_attention.linear_proj.bias",
            "blocks.*.attn2.norm_q.weight": "decoder.layers.*.cross_attention.q_layernorm.weight",
            "blocks.*.attn2.norm_k.weight": "decoder.layers.*.cross_attention.k_layernorm.weight",
            "blocks.*.norm2.weight": "decoder.layers.*.norm3.weight",
            "blocks.*.norm2.bias": "decoder.layers.*.norm3.bias",
            "blocks.*.ffn.net.0.proj.weight": "decoder.layers.*.mlp.linear_fc1.weight",
            "blocks.*.ffn.net.0.proj.bias": "decoder.layers.*.mlp.linear_fc1.bias",
            "blocks.*.ffn.net.2.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "blocks.*.ffn.net.2.bias": "decoder.layers.*.mlp.linear_fc2.bias",
            "proj_out.weight": "head.head.weight",
            "proj_out.bias": "head.head.bias",
        }

        # Custom WAN mapping to safely handle replicated params whose owning module
        # does not expose a top-level `.weight` (e.g., Head.modulation)
        class _ReplicatedByParamNameMapping(ReplicatedMapping):
            def hf_to_megatron(self, hf_weights, megatron_module):
                normalized_param = self._normalize_expert_param_name(self.megatron_param)
                _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)

                target_device = target_param.device
                target_dtype = target_param.dtype

                hf_weights = hf_weights.to(device=target_device, dtype=target_dtype)
                if self.tp_size == 1:
                    return hf_weights

                if target_device.type == "cuda" and torch.cuda.is_available():
                    if target_device.index != torch.cuda.current_device():
                        hf_weights = hf_weights.to(torch.cuda.current_device())

                if self.tp_rank > 0:
                    hf_weights = torch.empty_like(hf_weights)

                return self.broadcast_tensor_to_tp_ranks(hf_weights, src_rank=0)

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for hf_param, megatron_param in param_mappings.items():
            if hf_param in {"scale_shift_table", "blocks.*.scale_shift_table", "proj_out.weight", "proj_out.bias"}:
                # Use WAN-specific replicated mapping that resolves the exact param
                mapping_list.append(_ReplicatedByParamNameMapping(hf_param=hf_param, megatron_param=megatron_param))
            else:
                mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

        # Adding custom module types for AutoMapping
        AutoMapping.register_module_type("Linear", "replicated")
        AutoMapping.register_module_type("Conv3d", "replicated")
        AutoMapping.register_module_type("WanAdaLN", "replicated")
        AutoMapping.register_module_type("Head", "replicated")

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    q="blocks.*.attn1.to_q.weight",
                    k="blocks.*.attn1.to_k.weight",
                    v="blocks.*.attn1.to_v.weight",
                    megatron_param="decoder.layers.*.full_self_attention.linear_qkv.weight",
                ),
                # QKV bias: Combine separate Q, K, V bias into single QKV bias
                QKVMapping(
                    q="blocks.*.attn1.to_q.bias",
                    k="blocks.*.attn1.to_k.bias",
                    v="blocks.*.attn1.to_v.bias",
                    megatron_param="decoder.layers.*.full_self_attention.linear_qkv.bias",
                ),
                # K, V: Combine separate K, V matrices into single KV matrix
                KVMapping(
                    k="blocks.*.attn2.to_k.weight",
                    v="blocks.*.attn2.to_v.weight",
                    megatron_param="decoder.layers.*.cross_attention.linear_kv.weight",
                ),
                # K, V bias: Combine separate K, V bias into single KV bias
                KVMapping(
                    k="blocks.*.attn2.to_k.bias",
                    v="blocks.*.attn2.to_v.bias",
                    megatron_param="decoder.layers.*.cross_attention.linear_kv.bias",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
