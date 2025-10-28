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
import math
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import GenerationConfig, GptOssConfig, GptOssForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    QKVMapping,
)
from megatron.bridge.models.gpt_oss.gpt_oss_provider import GPTOSSProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


@MegatronModelBridge.register_bridge(source=GptOssForCausalLM, target=GPTModel)
class GPTOSSBridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for GPT-OSS models.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("openai/gpt-oss-model")
        >>> provider = bridge.to_megatron_provider()
    """

    def __init__(self):
        super().__init__()
        # gpt-oss HF weights has one weight for all the experts, but megatron has one for each expert
        # We need to cache the weights during import to load and dequantize the expert weights only once.
        # and we need to merge the weights of multiple experts during export.
        self.hf_weights_cache = {}

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM | GptOssConfig) -> GPTOSSProvider:
        if isinstance(hf_pretrained, PreTrainedCausalLM):
            hf_config = hf_pretrained.config
        else:
            hf_config = hf_pretrained

        # Extract generation config
        generation_config = getattr(hf_pretrained, "generation_config", None)
        if generation_config is None:
            try:
                generation_config = GenerationConfig.from_pretrained(str(hf_pretrained.name_or_path))
            except Exception:
                generation_config = None

        provider = GPTOSSProvider(
            num_layers=hf_config.num_hidden_layers,
            num_moe_experts=hf_config.num_local_experts,
            # for unit tests only:
            hidden_size=hf_config.hidden_size,
            moe_ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            vocab_size=hf_config.vocab_size,
        )
        return provider

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Load weights from HuggingFace state dict and dequantize if necessary."""
        if isinstance(hf_param, str):
            if hf_param in self.hf_weights_cache:
                return self.hf_weights_cache[hf_param]
            if hf_param in hf_state_dict:
                hf_weights = hf_state_dict[hf_param]
                if ".mlp.experts." in hf_param and len(hf_weights.shape) == 3:
                    hf_weights = hf_weights.transpose(-1, -2)
                self.hf_weights_cache[hf_param] = hf_weights
            else:
                blocks_key = hf_param + "_blocks"
                scales_key = hf_param + "_scales"
                if blocks_key in hf_state_dict and scales_key in hf_state_dict:
                    hf_weights = _dequantize_mxfp4(hf_state_dict[blocks_key], hf_state_dict[scales_key])
                    self.hf_weights_cache[hf_param] = hf_weights
                else:
                    raise KeyError(
                        f"Cannot locate weights for '{hf_param}'. Missing both de-quantized tensor and "
                        f"quantized representation (blocks='{blocks_key}', scales='{scales_key}')."
                    )
        else:
            hf_weights = {k: hf_state_dict[v] for k, v in hf_param.items()}
        return hf_weights

    def maybe_modify_converted_hf_weight(
        self, task: WeightConversionTask, converted_weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        num_experts = self.hf_config.num_local_experts
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
                merged_hf_weights = torch.cat(
                    [self.hf_weights_cache[key][i].unsqueeze(0) for i in range(num_experts)], dim=0
                )
                return {key: merged_hf_weights}
            else:
                # not all experts are loaded yet, return empty dict
                logging.debug(f"{len(self.hf_weights_cache[key])}/{num_experts} experts are loaded for {key}")
                return {}

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings from HF to Megatron format.
        Based on the GPT-OSS importer code provided.
        """

        # Dictionary maps HF parameter names -> Megatron parameter names
        param_mappings = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.self_attn.o_proj.bias": "decoder.layers.*.self_attention.linear_proj.bias",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.self_attn.sinks": "decoder.layers.*.self_attention.core_attention.softmax_offset",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.pre_mlp_layernorm.weight",
            "model.layers.*.mlp.router.bias": "decoder.layers.*.mlp.router.bias",
            "model.layers.*.mlp.router.weight": "decoder.layers.*.mlp.router.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for hf_param, megatron_param in param_mappings.items():
            mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

        mapping_list.extend(
            [
                QKVMapping(
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                ),
                QKVMapping(
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                ),
                # Register the de-quantized weight names. If HF model is quantized,
                # the logic in `modify_loaded_hf_weight` will find the blocks and scales tensors.
                # Export is always de-quantized
                GPTOSSMLPDownProjMapping(
                    hf_param="model.layers.*.mlp.experts.down_proj",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                ),
                GPTOSSMLPDownProjMapping(
                    hf_param="model.layers.*.mlp.experts.down_proj_bias",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.bias*",
                ),
                GPTOSSMLPGateUpProjMapping(
                    hf_param="model.layers.*.mlp.experts.gate_up_proj",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                ),
                GPTOSSMLPGateUpProjMapping(
                    hf_param="model.layers.*.mlp.experts.gate_up_proj_bias",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.bias*",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)


class GPTOSSMLPDownProjMapping(AutoMapping):
    """
    MLPDownProj for expert weights GPT-OSS models.
    """

    def __init__(self, megatron_param: str, hf_param: str, permute_dims: Optional[Tuple[int, ...]] = None):
        super().__init__(megatron_param, hf_param, permute_dims)
        self.allow_hf_name_mismatch = True

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        return super().hf_to_megatron(hf_weights[global_expert_number], megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> Dict[str, torch.Tensor]:
        # only bf16 export is supported currently
        if megatron_weights is None:
            return super().megatron_to_hf(megatron_weights, megatron_module)

        # GPT-OSS stores fc2 weight transposed vs Megatron when using BF16.
        if len(megatron_weights.shape) == 2:
            megatron_weights = megatron_weights.transpose(0, 1)
        return super().megatron_to_hf(megatron_weights.contiguous(), megatron_module)

    def _validate_patterns(self, *args, **kwargs):
        # allow number of wildcards to mismatch in this mapping
        pass


class GPTOSSMLPGateUpProjMapping(AutoMapping):
    """
    MLPGateUpProj for expert weights GPT-OSS models.
    """

    def __init__(self, megatron_param: str, hf_param: str, permute_dims: Optional[Tuple[int, ...]] = None):
        super().__init__(megatron_param, hf_param, permute_dims)
        self.allow_hf_name_mismatch = True

    @staticmethod
    def _interleave(gate_up_proj):
        return torch.cat((gate_up_proj[::2, ...], gate_up_proj[1::2, ...]), dim=0)

    def _uninterleave(self, elem):
        gate, up = torch.chunk(elem, 2, dim=0)
        output = torch.empty_like(elem)
        output[::2, ...] = gate
        output[1::2, ...] = up
        return output

    def hf_to_megatron(self, hf_weights: Union[torch.Tensor, Dict], megatron_module: nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        return super().hf_to_megatron(self._interleave(hf_weights[global_expert_number]), megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> Dict[str, torch.Tensor]:
        # only bf16 export is supported currently
        if megatron_weights is None:
            return super().megatron_to_hf(megatron_weights, megatron_module)

        megatron_weights = self._uninterleave(megatron_weights)
        if len(megatron_weights.shape) == 2:
            megatron_weights = megatron_weights.transpose(0, 1)
        return super().megatron_to_hf(megatron_weights.contiguous(), megatron_module)

    def _validate_patterns(self, *args, **kwargs):
        # allow number of wildcards to mismatch in this mapping
        pass


def _dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"
    FP4_VALUES = [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    scales = scales.to(torch.int32) - 127
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
