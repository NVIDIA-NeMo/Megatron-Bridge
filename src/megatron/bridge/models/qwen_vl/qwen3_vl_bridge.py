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

import logging
from typing import Dict, Mapping, Union

import torch
import torch.nn as nn
from megatron.core import parallel_state
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider, Qwen3VLMoEModelProvider
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


@MegatronModelBridge.register_bridge(
    source=Qwen3VLForConditionalGeneration,
    target=Qwen3VLModel,
    provider=Qwen3VLModelProvider,
    model_type="qwen3_vl",
)
class Qwen3VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3-VL Conditional Generation.

    This bridge handles the conversion between HuggingFace Qwen3VLForConditionalGeneration
    and Megatron-Core Qwen3VLModel formats, including weight mappings and
    configuration translation for vision-language models.

    The weight mappings are based on the yan-mbridge implementation which defines:
    - Vision model direct mappings
    - Vision attention layer mappings
    - Vision MLP layer mappings
    - Language model mappings
    - Deepstack visual merger mappings

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen3VLModelProvider:
        """
        Create a Qwen3VLModelProvider from a HuggingFace pretrained model.

        Args:
            hf_pretrained: HuggingFace pretrained VLM model

        Returns:
            Qwen3VLModelProvider configured with the HF model's parameters
        """
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)

        vision_config = hf_config.vision_config
        vision_config.torch_dtype = provider_kwargs.get("params_dtype", torch.float32)

        provider = Qwen3VLModelProvider(**provider_kwargs)

        # Qwen3-specific settings
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_qkv_bias = text_config.attention_bias
        provider.add_bias_linear = False
        provider.qk_layernorm = True
        provider.hidden_dropout = 0.0
        provider.rotary_base = rope_theta_from_hf(text_config)

        # For VLMs, tie_word_embeddings lives on the top-level config, not text_config.
        # text_config inherits PretrainedConfig's default of True which is wrong.
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # VL-specific overrides
        provider.position_embedding_type = "mrope"
        provider.vision_config = vision_config
        provider.hf_text_config = text_config
        provider.head_dim = text_config.head_dim
        provider.bos_token_id = getattr(text_config, "bos_token_id", 151643)
        provider.eos_token_id = getattr(text_config, "eos_token_id", 151645)
        provider.vision_start_token_id = getattr(hf_config, "vision_start_token_id", 151652)
        provider.vision_end_token_id = getattr(hf_config, "vision_end_token_id", 151653)
        provider.image_token_id = getattr(hf_config, "image_token_id", 151655)
        provider.video_token_id = getattr(hf_config, "video_token_id", 151656)
        rope_cfg = getattr(text_config, "rope_parameters", None) or getattr(text_config, "rope_scaling", {})
        provider.mrope_section = rope_cfg.get("mrope_section", [24, 20, 20])

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            # vision module attn
            "vision_model.decoder.layers.*.self_attention.linear_proj.weight": "model.visual.blocks.*.attn.proj.weight",
            "vision_model.decoder.layers.*.self_attention.linear_proj.bias": "model.visual.blocks.*.attn.proj.bias",
            # vision module mlp
            "vision_model.decoder.layers.*.mlp.linear_fc1.weight": "model.visual.blocks.*.mlp.linear_fc1.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc1.bias": "model.visual.blocks.*.mlp.linear_fc1.bias",
            "vision_model.decoder.layers.*.mlp.linear_fc2.weight": "model.visual.blocks.*.mlp.linear_fc2.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc2.bias": "model.visual.blocks.*.mlp.linear_fc2.bias",
            # vision module norm
            "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.visual.blocks.*.norm1.weight",
            "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "model.visual.blocks.*.norm1.bias",
            "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.visual.blocks.*.norm2.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "model.visual.blocks.*.norm2.bias",
            # # vision module deepstack merger
            "vision_model.decoder.deepstack_merger_list.*.patch_norm.weight": "model.visual.deepstack_merger_list.*.norm.weight",
            "vision_model.decoder.deepstack_merger_list.*.patch_norm.bias": "model.visual.deepstack_merger_list.*.norm.bias",
            "vision_model.decoder.deepstack_merger_list.*.linear_fc1.weight": "model.visual.deepstack_merger_list.*.linear_fc1.weight",
            "vision_model.decoder.deepstack_merger_list.*.linear_fc1.bias": "model.visual.deepstack_merger_list.*.linear_fc1.bias",
            "vision_model.decoder.deepstack_merger_list.*.linear_fc2.weight": "model.visual.deepstack_merger_list.*.linear_fc2.weight",
            "vision_model.decoder.deepstack_merger_list.*.linear_fc2.bias": "model.visual.deepstack_merger_list.*.linear_fc2.bias",
            # vision module merger
            "vision_model.merger.patch_norm.**": "model.visual.merger.norm.**",
            "vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
            "vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
            "vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
            "vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
        }

        mapping_list = []

        # Convert simple 1:1 mappings to AutoMapping objects
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter transformation
        mapping_list.extend(
            [
                # QKV mapping: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                # QKV bias mapping (if attention_bias is True)
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.language_model.layers.*.self_attn.q_proj.bias",
                    k="model.language_model.layers.*.self_attn.k_proj.bias",
                    v="model.language_model.layers.*.self_attn.v_proj.bias",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
                ConcatenatedQKVMapping(
                    megatron_param="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    hf_param="model.visual.blocks.*.attn.qkv.weight",
                ),
                ConcatenatedQKVMapping(
                    megatron_param="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    hf_param="model.visual.blocks.*.attn.qkv.bias",
                ),
                ReplicatedMapping(
                    megatron_param="vision_model.patch_embed.proj.**",
                    hf_param="model.visual.patch_embed.proj.**",
                ),
                ReplicatedMapping(
                    megatron_param="vision_model.pos_embed.weight",
                    hf_param="model.visual.pos_embed.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)


@MegatronModelBridge.register_bridge(
    source=Qwen3VLMoeForConditionalGeneration,
    target=Qwen3VLModel,
    provider=Qwen3VLMoEModelProvider,
    model_type="qwen3_vl_moe",
)
class Qwen3VLMoEBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3-VL MoE (Mixture of Experts) Conditional Generation.

    This bridge handles the conversion between HuggingFace Qwen3VLMoEForConditionalGeneration
    and Megatron-Core Qwen3VL MoE model formats, including weight mappings and
    configuration translation for vision-language MoE models.

    The weight mappings handle:
    - Vision model weights (same as dense model)
    - Language model MoE layers with expert routing
    - Shared embeddings and output layers
    - QK layernorm specific to Qwen3 architecture

    This bridge works with any Qwen3VL MoE model size and automatically extracts
    the MoE configuration from the HuggingFace model.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    def __init__(self):
        super().__init__()
        self.hf_weights_cache: Dict[str, Dict[int, torch.Tensor]] = {}

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen3VLMoEModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)

        vision_config = hf_config.vision_config
        vision_config.torch_dtype = provider_kwargs.get("params_dtype", torch.float32)

        provider = Qwen3VLMoEModelProvider(**provider_kwargs)

        # Qwen3 MoE-specific settings
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_qkv_bias = text_config.attention_bias
        provider.add_bias_linear = False
        provider.qk_layernorm = True
        provider.hidden_dropout = 0.0
        provider.rotary_base = rope_theta_from_hf(text_config)

        # For VLMs, tie_word_embeddings lives on the top-level config, not text_config.
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # MoE specific parameters
        provider.moe_ffn_hidden_size = text_config.moe_intermediate_size
        provider.num_moe_experts = text_config.num_experts
        provider.moe_router_topk = text_config.num_experts_per_tok
        provider.decoder_sparse_step = getattr(text_config, "decoder_sparse_step", 1)
        provider.mlp_only_layers = getattr(text_config, "mlp_only_layers", [])
        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True

        # VL-specific overrides
        provider.position_embedding_type = "mrope"
        provider.vision_config = vision_config
        provider.hf_text_config = text_config
        provider.head_dim = getattr(
            text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads
        )
        provider.bos_token_id = getattr(text_config, "bos_token_id", 151643)
        provider.eos_token_id = getattr(text_config, "eos_token_id", 151645)
        provider.vision_start_token_id = getattr(hf_config, "vision_start_token_id", 151652)
        provider.vision_end_token_id = getattr(hf_config, "vision_end_token_id", 151653)
        provider.image_token_id = getattr(hf_config, "image_token_id", 151655)
        provider.video_token_id = getattr(hf_config, "video_token_id", 151656)
        rope_cfg = getattr(text_config, "rope_parameters", None) or getattr(text_config, "rope_scaling", {})
        provider.mrope_section = rope_cfg.get("mrope_section", [24, 20, 20])

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings for MoE models.

        The MoE mappings include:
        1. Standard language model mappings (embeddings, layer norms, output)
        2. Vision model mappings (same as dense model)
        3. QKV mappings with QK layernorm
        4. MoE-specific mappings:
           - Router weights for expert selection
           - Expert MLPs (multiple experts per layer)
           - Pre-MLP layernorm
        5. Deepstack visual merger mappings

        Returns:
            MegatronMappingRegistry with all MoE parameter mappings
        """
        # Language model direct mappings (same as dense model)
        param_mappings = {
            # Embeddings and output layers
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            # Layer normalization for attention (TE format - fused into linear)
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            # Layer normalization (non-TE/quantization format - separate modules)
            "language_model.decoder.layers.*.input_layernorm.weight": "model.language_model.layers.*.input_layernorm.weight",
            # MoE-specific: pre-MLP layernorm (already in non-TE format for MoE)
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            # Attention output projection
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            # QK layernorm weights (Qwen3 specific)
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            # MoE router weights
            "language_model.decoder.layers.*.mlp.router.weight": "model.language_model.layers.*.mlp.gate.weight",
            # vision module attn
            "vision_model.decoder.layers.*.self_attention.linear_proj.weight": "model.visual.blocks.*.attn.proj.weight",
            "vision_model.decoder.layers.*.self_attention.linear_proj.bias": "model.visual.blocks.*.attn.proj.bias",
            # vision module mlp
            "vision_model.decoder.layers.*.mlp.linear_fc1.weight": "model.visual.blocks.*.mlp.linear_fc1.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc1.bias": "model.visual.blocks.*.mlp.linear_fc1.bias",
            "vision_model.decoder.layers.*.mlp.linear_fc2.weight": "model.visual.blocks.*.mlp.linear_fc2.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc2.bias": "model.visual.blocks.*.mlp.linear_fc2.bias",
            # vision module norm
            "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.visual.blocks.*.norm1.weight",
            "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "model.visual.blocks.*.norm1.bias",
            "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.visual.blocks.*.norm2.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "model.visual.blocks.*.norm2.bias",
            # # vision module deepstack merger
            "vision_model.decoder.deepstack_merger_list.*.patch_norm.weight": "model.visual.deepstack_merger_list.*.norm.weight",
            "vision_model.decoder.deepstack_merger_list.*.patch_norm.bias": "model.visual.deepstack_merger_list.*.norm.bias",
            "vision_model.decoder.deepstack_merger_list.*.linear_fc1.weight": "model.visual.deepstack_merger_list.*.linear_fc1.weight",
            "vision_model.decoder.deepstack_merger_list.*.linear_fc1.bias": "model.visual.deepstack_merger_list.*.linear_fc1.bias",
            "vision_model.decoder.deepstack_merger_list.*.linear_fc2.weight": "model.visual.deepstack_merger_list.*.linear_fc2.weight",
            "vision_model.decoder.deepstack_merger_list.*.linear_fc2.bias": "model.visual.deepstack_merger_list.*.linear_fc2.bias",
            # vision module merger
            "vision_model.merger.patch_norm.**": "model.visual.merger.norm.**",
            "vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
            "vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
            "vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
            "vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
        }

        mapping_list = []

        # Convert simple 1:1 mappings to AutoMapping objects
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter transformation
        mapping_list.extend(
            [
                # QKV mapping: Combine separate Q, K, V matrices
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                # QKV bias mapping (if attention_bias is True)
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.language_model.layers.*.self_attn.q_proj.bias",
                    k="model.language_model.layers.*.self_attn.k_proj.bias",
                    v="model.language_model.layers.*.self_attn.v_proj.bias",
                ),
                ExpertMLPGateUpProjMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    hf_param="model.language_model.layers.*.mlp.experts.gate_up_proj",
                ),
                ExpertMLPDownProjMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.language_model.layers.*.mlp.experts.down_proj",
                ),
                # QKV mapping for vision model
                ConcatenatedQKVMapping(
                    megatron_param="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    hf_param="model.visual.blocks.*.attn.qkv.weight",
                ),
                ConcatenatedQKVMapping(
                    megatron_param="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    hf_param="model.visual.blocks.*.attn.qkv.bias",
                ),
                ReplicatedMapping(  # These patch_embed are conv, we need to use ReplicatedMapping
                    megatron_param="vision_model.patch_embed.proj.**",
                    hf_param="model.visual.patch_embed.proj.**",
                ),
                ReplicatedMapping(
                    megatron_param="vision_model.pos_embed.weight",
                    hf_param="model.visual.pos_embed.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        num_experts = self.hf_config.text_config.num_experts
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        experts_per_rank = num_experts // ep_size

        try:
            local_expert_number = extract_expert_number_from_param(task.param_name) % experts_per_rank
        except ValueError:
            # not an expert weight
            return converted_weights_dict

        assert len(converted_weights_dict) == 1, (
            f"There should be only one key in the converted_weights_dict, got keys: {converted_weights_dict.keys()}"
        )
        for key, value in converted_weights_dict.items():
            if key not in self.hf_weights_cache:
                self.hf_weights_cache[key] = {}

            # we end up with ep_size many weights to add to the cache
            # unpack the weights and re-index
            if ep_size == 1:
                self.hf_weights_cache[key][local_expert_number] = value
            else:
                assert value.shape[0] == ep_size
                for i, exp_val in enumerate(value):
                    global_expert_number = local_expert_number + (i * experts_per_rank)
                    self.hf_weights_cache[key][global_expert_number] = exp_val
            if len(self.hf_weights_cache[key]) == num_experts:
                logging.debug(f"All experts are loaded for {key}")
                merged = torch.cat([self.hf_weights_cache[key][i].unsqueeze(0) for i in range(num_experts)], dim=0)
                del self.hf_weights_cache[key]
                return {key: merged}
            else:
                # not all experts are loaded yet, return empty dict
                logging.debug(f"{len(self.hf_weights_cache[key])}/{num_experts} experts are loaded for {key}")
                return {}


def _align_weight_to_shape(weight: torch.Tensor, target_shape: torch.Size, name: str) -> torch.Tensor:
    """Auto-detect whether a transpose is needed to match the Megatron target shape.

    Transformers <5.0 stored fused expert weights transposed as
    [num_experts, hidden_size, 2*intermediate_size], while transformers 5.0+
    uses the standard nn.Linear convention [num_experts, 2*intermediate_size, hidden_size].
    This helper accepts either layout and transposes only when necessary, so the
    bridge works with both real checkpoints (old format) and toy models or new
    checkpoints created with transformers 5.0+.
    """
    if tuple(weight.shape) == tuple(target_shape):
        return weight
    if weight.ndim == 2 and tuple(weight.t().shape) == tuple(target_shape):
        return weight.t().contiguous()
    raise ValueError(f"Unexpected {name} shape {tuple(weight.shape)}; expected {tuple(target_shape)}.")


class ExpertMLPDownProjMapping(AutoMapping):
    """Mapping for expert MLP down projection weights between HF and Megatron formats.

    Uses _align_weight_to_shape so both pre-5.0 (transposed) and 5.0+
    (standard) HF expert weight layouts are handled transparently.
    """

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        expert_weight = hf_weights[global_expert_number] if hf_weights.ndim >= 3 else hf_weights

        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        expert_weight = _align_weight_to_shape(expert_weight, target_param.shape, "down_proj")
        return super().hf_to_megatron(expert_weight, megatron_module)

    def _validate_patterns(self, *args, **kwargs):
        pass


class ExpertMLPGateUpProjMapping(AutoMapping):
    """Mapping for expert MLP gate+up projection using shared GatedMLPMapping logic.

    Uses _align_weight_to_shape so both pre-5.0 (transposed) and 5.0+
    (standard) HF expert weight layouts are handled transparently.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        GatedMLPMapping._validate_patterns = lambda *args, **kwargs: None

        self._gated_mapping = GatedMLPMapping(
            megatron_param=self.megatron_param,
            gate=f"{self.hf_param}.gate",
            up=f"{self.hf_param}.up",
        )

    def hf_to_megatron(self, hf_weights: Union[torch.Tensor, Dict], megatron_module: nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        expert_weight = hf_weights[global_expert_number] if hf_weights.ndim >= 3 else hf_weights

        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        target_shape = target_param.shape
        gate_target_shape = (target_shape[0] // 2, target_shape[1])

        if target_shape[0] % 2 != 0:
            raise ValueError(f"Expected even fused dim for {self.megatron_param}, got {target_shape}.")

        if expert_weight.ndim == 3 and expert_weight.shape[0] == 2:
            gate = _align_weight_to_shape(expert_weight[0], gate_target_shape, "gate")
            up = _align_weight_to_shape(expert_weight[1], gate_target_shape, "up")
        else:
            expert_weight = _align_weight_to_shape(expert_weight, target_shape, "gate_up")
            gate, up = torch.chunk(expert_weight, 2, dim=0)

        return self._gated_mapping.hf_to_megatron({"gate": gate, "up": up}, megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> Dict[str, torch.Tensor]:
        converted = self._gated_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not converted:
            return {}

        fused: Dict[str, torch.Tensor] = {}
        for name, tensor in converted.items():
            if not name.endswith(".gate"):
                continue
            base_name = name[: -len(".gate")]
            up_tensor = converted.get(f"{base_name}.up")
            if up_tensor is None:
                continue
            concat_dim = 0 if tensor.ndim == 2 else 1
            fused[base_name] = torch.cat([tensor, up_tensor], dim=concat_dim)
        return fused

    def _validate_patterns(self, *args, **kwargs):
        pass
