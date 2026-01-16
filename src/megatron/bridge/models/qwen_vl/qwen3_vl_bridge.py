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
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider, Qwen3VLMoEModelProvider
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


@MegatronModelBridge.register_bridge(source=Qwen3VLForConditionalGeneration, target=Qwen3VLModel)
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

        # Get the model dtype from text config
        model_dtype = self.dtype_from_hf(text_config, default=torch.float32)

        # Set vision config dtype to match the language model dtype
        # This ensures vision model parameters are initialized in the same dtype
        vision_config = hf_config.vision_config
        vision_config.torch_dtype = model_dtype

        # Create the provider with text model configuration
        provider = Qwen3VLModelProvider(
            # Language model configuration from text_config
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,  # GQA configuration
            head_dim=text_config.head_dim,
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,  # Qwen3 uses gated linear units
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=text_config.rope_theta,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            generation_config=hf_pretrained.generation_config,
            # Qwen3 specific parameters
            add_qkv_bias=text_config.attention_bias,  # Qwen3 can have bias in QKV
            qk_layernorm=True,  # Qwen3 uses QK layernorm
            # Vision configuration
            vision_config=vision_config,
            # Store the original HF text config for RoPE initialization
            hf_text_config=text_config,
            # Vision-Language token IDs
            bos_token_id=getattr(text_config, "bos_token_id", 151643),
            eos_token_id=getattr(text_config, "eos_token_id", 151645),
            vision_start_token_id=getattr(hf_config, "vision_start_token_id", 151652),
            vision_end_token_id=getattr(hf_config, "vision_end_token_id", 151653),
            image_token_id=getattr(hf_config, "image_token_id", 151655),
            video_token_id=getattr(hf_config, "video_token_id", 151656),
            # MRoPE configuration for multimodal position embeddings
            mrope_section=text_config.rope_scaling.get("mrope_section", [24, 20, 20]),
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format.

        The mappings are organized into:
        1. Simple 1:1 mappings for embeddings, layer norms, and output layers
        2. Vision model mappings (replicated without modification)
        3. QKV mappings that combine separate Q, K, V matrices
        4. Gated MLP mappings that combine gate and up projections
        5. Deepstack visual merger mappings

        Returns:
            MegatronMappingRegistry with all parameter mappings
        """
        # Dictionary maps Megatron parameter names -> HF parameter names
        # Based on yan-mbridge weight mappings in __init__.py

        # Language model direct mappings
        param_mappings = {
            # Embeddings and output layers
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            # Layer normalization for attention and MLP
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            # Attention output projection
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            # MLP output projection
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
            # QK layernorm weights (Qwen3 specific)
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
        }

        mapping_list = []

        # Convert simple 1:1 mappings to AutoMapping objects
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter transformation
        mapping_list.extend(
            [
                # Vision model weights are replicated directly
                # This handles all vision encoder layers, patch embeddings, mergers, etc.
                ReplicatedMapping(
                    megatron_param="vision_model.**",
                    hf_param="model.visual.**",
                ),
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
            ]
        )

        return MegatronMappingRegistry(*mapping_list)


@MegatronModelBridge.register_bridge(source=Qwen3VLMoeForConditionalGeneration, target=Qwen3VLModel)
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
        # Cache expert shards during HF export until all ranks contribute.
        self.hf_weights_cache: Dict[str, Dict[int, torch.Tensor]] = {}

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen3VLMoEModelProvider:
        """
        Create a Qwen3VLMoEModelProvider from a HuggingFace pretrained MoE model.

        Args:
            hf_pretrained: HuggingFace pretrained VLM MoE model

        Returns:
            Qwen3VLMoEModelProvider configured with the HF MoE model's parameters
        """
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        # Get the model dtype from text config
        model_dtype = self.dtype_from_hf(text_config, default=torch.float32)

        # Set vision config dtype to match the language model dtype
        # This ensures vision model parameters are initialized in the same dtype
        vision_config = hf_config.vision_config
        vision_config.torch_dtype = model_dtype

        provider = Qwen3VLMoEModelProvider(
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,  # Dense FFN size (for non-MoE layers if any)
            moe_ffn_hidden_size=text_config.moe_intermediate_size,  # Expert FFN size
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,  # GQA configuration
            head_dim=getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads),
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,  # Qwen3 MoE uses gated linear units
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=getattr(text_config, "rope_theta", 5000000.0),  # Default Qwen3 rope theta
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            generation_config=hf_pretrained.generation_config,
            # Qwen3 specific parameters
            add_qkv_bias=text_config.attention_bias,  # Qwen3 can have bias in QKV
            qk_layernorm=True,  # Qwen3 uses QK layernorm
            # MoE specific parameters
            num_moe_experts=text_config.num_experts,
            moe_router_topk=text_config.num_experts_per_tok,
            decoder_sparse_step=getattr(text_config, "decoder_sparse_step", 1),  # Default to every layer being MoE
            mlp_only_layers=getattr(text_config, "mlp_only_layers", []),  # Default to all layers using MoE
            # Vision configuration
            vision_config=vision_config,
            # Store the original HF text config for RoPE initialization
            hf_text_config=text_config,
            # Vision-Language token IDs
            bos_token_id=getattr(text_config, "bos_token_id", 151643),
            eos_token_id=getattr(text_config, "eos_token_id", 151645),
            vision_start_token_id=getattr(hf_config, "vision_start_token_id", 151652),
            vision_end_token_id=getattr(hf_config, "vision_end_token_id", 151653),
            image_token_id=getattr(hf_config, "image_token_id", 151655),
            video_token_id=getattr(hf_config, "video_token_id", 151656),
            # MRoPE configuration for multimodal position embeddings
            mrope_section=getattr(text_config, "rope_scaling", {}).get("mrope_section", [24, 20, 20]),
        )

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
            # Layer normalization for attention
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            # MoE-specific: pre-MLP layernorm
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            # Attention output projection
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            # QK layernorm weights (Qwen3 specific)
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            # MoE router weights
            "language_model.decoder.layers.*.mlp.router.weight": "model.language_model.layers.*.mlp.gate.weight",
        }

        mapping_list = []

        # Convert simple 1:1 mappings to AutoMapping objects
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter transformation
        mapping_list.extend(
            [
                # Vision model weights are replicated directly
                ReplicatedMapping(
                    megatron_param="vision_model.**",
                    hf_param="model.visual.**",
                ),
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
                # all experts are loaded
                if self.hf_weights_cache[key][0].ndim == 3:  # expert 0
                    # gate up
                    merged_hf_gate_weights = torch.cat(
                        [self.hf_weights_cache[key][i][0].unsqueeze(0) for i in range(num_experts)], dim=0
                    )
                    merged_hf_up_weights = torch.cat(
                        [self.hf_weights_cache[key][i][1].unsqueeze(0) for i in range(num_experts)], dim=0
                    )
                    del self.hf_weights_cache[key]
                    return {key: torch.cat([merged_hf_gate_weights, merged_hf_up_weights], dim=-1)}
                elif self.hf_weights_cache[key][0].ndim == 2:  # expert 0
                    # down
                    merged_hf_down_weights = torch.cat(
                        [self.hf_weights_cache[key][i].unsqueeze(0) for i in range(num_experts)], dim=0
                    )
                    del self.hf_weights_cache[key]
                    return {key: merged_hf_down_weights}
                else:
                    raise ValueError(
                        f"Incorrect shape of self.hf_weights_cache[key]: {key} {self.hf_weights_cache[key].shape}"
                    )
            else:
                # not all experts are loaded yet, return empty dict
                logging.debug(f"{len(self.hf_weights_cache[key])}/{num_experts} experts are loaded for {key}")
                return {}


class ExpertMLPDownProjMapping(AutoMapping):
    """Mapping for expert MLP down projection weights between HF and Megatron formats."""

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        # hf_weights: [num_experts, down_in, mlp_out]
        expert_weight = hf_weights[global_expert_number].transpose(0, 1).contiguous()
        return super().hf_to_megatron(expert_weight, megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> Dict[str, torch.Tensor]:
        # [ep_size, down_in, mlp_out]
        # experts need subsequently merged by maybe_modify_converted_hf_weight
        converted_weights_dict = super().megatron_to_hf(megatron_weights, megatron_module)
        for key in converted_weights_dict:
            converted_weights_dict[key] = converted_weights_dict[key].transpose(-1, -2).contiguous()
        return converted_weights_dict

    def _validate_patterns(self, *args, **kwargs):
        # allow number of wildcards to mismatch in this mapping
        pass


class ExpertMLPGateUpProjMapping(AutoMapping):
    """Mapping for expert MLP gate+up projection using shared GatedMLPMapping logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Qwen3-VL MoE expert shards use mismatched wildcard counts; relax validation globally.
        GatedMLPMapping._validate_patterns = lambda *args, **kwargs: None

        # Reuse the generic TP-aware split/gather, but we still handle expert selection
        # and HF<->Megatron transpose at this wrapper layer.
        self._gated_mapping = GatedMLPMapping(
            megatron_param=self.megatron_param,
            gate=f"{self.hf_param}.gate",
            up=f"{self.hf_param}.up",
        )

    def hf_to_megatron(self, hf_weights: Union[torch.Tensor, Dict], megatron_module: nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        # hf_weights: [num_experts, mlp_in, fused_gate_up_out]
        expert_weight = hf_weights[global_expert_number].transpose(0, 1).contiguous()

        # HF gate_up_proj is [2 * hidden, hidden]; Megatron expects transposed.
        gate, up = torch.chunk(expert_weight, 2, dim=0)
        return self._gated_mapping.hf_to_megatron({"gate": gate, "up": up}, megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> Dict[str, torch.Tensor]:
        # Let the shared mapping handle TP/PP/EP gather.
        converted = self._gated_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not converted:
            return {}

        fused: Dict[str, torch.Tensor] = {}

        # only one pair of gate and up for current group of experts
        for name, tensor in converted.items():
            if name.endswith(".gate"):
                base_name = name[: -len(".gate")]
                # [ep_size, mlp_in, gate_out]
                gate_tensor = tensor.transpose(-1, -2).contiguous()
                # [ep_size, mlp_in, up_out]
                up_tensor = converted.get(f"{base_name}.up").transpose(-1, -2).contiguous()
                assert up_tensor is not None
                # Back to HF fused layout: stack [gate; up] along dim 0.
                # [ep_size, 2, mlp_in, gate_out/up_out]
                fused[base_name] = torch.stack([gate_tensor, up_tensor], dim=0 if up_tensor.ndim == 2 else 1)

        # experts need subsequently merged by maybe_modify_converted_hf_weight
        return fused

    def _validate_patterns(self, *args, **kwargs):
        # allow number of wildcards to mismatch in this mapping
        pass