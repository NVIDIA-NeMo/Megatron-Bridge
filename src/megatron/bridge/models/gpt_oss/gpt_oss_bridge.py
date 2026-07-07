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

import copy
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_model import HybridModel
from transformers import GptOssForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    QKVMapping,
    _align_expert_weight_to_shape,
)
from megatron.bridge.models.conversion.quantization_utils import dequantize_mxfp4 as _dequantize_mxfp4
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.bridge.models.gpt_oss.model_config import GPTOSSModelConfig
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


try:
    from megatron.core.fusions.fused_bias_geglu import quick_gelu
except ImportError:
    # Fallback if fused_bias_geglu is not available
    quick_gelu = torch.nn.functional.gelu


@MegatronModelBridge.register_bridge(
    source=GptOssForCausalLM,
    target=HybridModel,
    provider=HybridModelProvider,
    model_type="gpt_oss",
)
class GPTOSSBridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for GPT-OSS models.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("openai/gpt-oss-model")
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = GPTOSSModelConfig

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> HybridModelProvider:
        """Convert HuggingFace config to HybridModelProvider."""
        provider = super().provider_bridge(hf_pretrained)

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = True
        provider.add_qkv_bias = False
        provider.share_embeddings_and_output_weights = False
        provider.position_embedding_type = "yarn"

        provider.moe_router_pre_softmax = False
        provider.moe_grouped_gemm = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.moe_router_load_balancing_type = "none"

        provider.bias_activation_fusion = True
        provider.bias_dropout_fusion = False

        provider.hidden_dropout = 0.0
        provider.fp16 = False
        provider.bf16 = True
        provider.params_dtype = torch.bfloat16

        # GPT-OSS specific activation
        provider.activation_func = quick_gelu
        provider.activation_func_clamp_value = 7.0
        provider.glu_linear_offset = 1.0

        num_hf_layers = provider.num_layers
        provider.hybrid_layer_pattern = (Symbols.ATTENTION + Symbols.MOE) * num_hf_layers
        provider.num_layers = len(provider.hybrid_layer_pattern)

        provider.softmax_type = "learnable"
        provider.window_size = (hf_pretrained.config.sliding_window - 1, 0)
        provider.window_attn_skip_freq = [
            flag for layer_idx in range(num_hf_layers) for flag in ((layer_idx + 1) % 2 != 0, False)
        ]

        # GPT-OSS uses intermediate_size for MoE FFN hidden size
        provider.moe_ffn_hidden_size = hf_pretrained.config.intermediate_size

        # YARN position embedding settings.
        provider.yarn_rotary_scaling_factor = 32.0
        provider.yarn_original_max_position_embeddings = 4096
        provider.yarn_beta_fast = 32.0
        provider.yarn_beta_slow = 1.0
        provider.yarn_correction_range_round_to_int = False
        provider.yarn_mscale = None
        provider.yarn_mscale_all_dim = None

        return provider

    def hf_config_to_model_config_kwargs(self, hf_config: Any) -> dict[str, Any]:
        """Convert GPT-OSS HF config to builder-backed HybridModel settings."""
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        unscaled_config = copy.copy(hf_config)
        if hasattr(unscaled_config, "rope_scaling"):
            unscaled_config.rope_scaling = None
        config_kwargs = super().hf_config_to_model_config_kwargs(unscaled_config)
        num_hf_layers = hf_config.num_hidden_layers

        yarn_values = {
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": None,
            "mscale_all_dim": None,
        }
        if isinstance(rope_scaling, dict):
            yarn_values.update({name: rope_scaling[name] for name in yarn_values if name in rope_scaling})

        config_kwargs.update(
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=True,
            add_qkv_bias=False,
            share_embeddings_and_output_weights=False,
            position_embedding_type="yarn",
            moe_router_pre_softmax=False,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            moe_router_load_balancing_type="none",
            bias_activation_fusion=True,
            bias_dropout_fusion=False,
            hidden_dropout=0.0,
            fp16=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
            activation_func=quick_gelu,
            activation_func_clamp_value=7.0,
            glu_linear_offset=1.0,
            num_layers=2 * num_hf_layers,
            hybrid_layer_pattern=(Symbols.ATTENTION + Symbols.MOE) * num_hf_layers,
            softmax_type="learnable",
            window_size=(hf_config.sliding_window - 1, 0),
            window_attn_skip_freq=[
                flag for layer_idx in range(num_hf_layers) for flag in ((layer_idx + 1) % 2 != 0, False)
            ],
            moe_ffn_hidden_size=hf_config.intermediate_size,
            yarn_rotary_scaling_factor=yarn_values["factor"],
            yarn_original_max_position_embeddings=yarn_values["original_max_position_embeddings"],
            yarn_beta_fast=yarn_values["beta_fast"],
            yarn_beta_slow=yarn_values["beta_slow"],
            yarn_mscale=yarn_values["mscale"],
            yarn_mscale_all_dim=yarn_values["mscale_all_dim"],
            yarn_correction_range_round_to_int=(
                rope_scaling.get("truncate", False) if isinstance(rope_scaling, dict) else False
            ),
        )
        return config_kwargs

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Convert Megatron Hybrid GPT-OSS config back to HuggingFace config."""
        hf_config = super().megatron_to_hf_config(provider)
        hybrid_layer_pattern = getattr(provider, "hybrid_layer_pattern", None)
        if hybrid_layer_pattern:
            main_pattern = hybrid_layer_pattern.split(Symbols.MTP_SEPARATOR)[0].replace(Symbols.PIPE, "")
            hf_config["num_hidden_layers"] = main_pattern.count(Symbols.ATTENTION)
        return hf_config

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Load weights from HuggingFace state dict with MXFP4 dequantization support.

        Per-expert ``down_proj`` is square for GPT-OSS-20B/120B (hidden == intermediate), so
        the bridge cannot auto-detect orientation from shape alone. BF16 checkpoints (e.g.
        ``unsloth/gpt-oss-20b-BF16``, and what ``transformers.GptOssForCausalLM`` produces at
        init) store it as ``[E, intermediate, hidden]``, matching ``gate_up_proj``'s
        ``[E, hidden, 2*intermediate]`` convention. MXFP4-dequantized weights come out as
        ``[E, hidden, intermediate]``. Megatron's TE ``RowParallelGroupedLinear`` expects
        per-expert ``(hidden, intermediate)``, so the BF16 path needs a transpose here while
        the MXFP4 path is already aligned. Without this, BF16 imports silently store down_proj
        in the wrong orientation and inference is broken.

        gate_up_proj is handled directly in GPTOSSMLPGateUpProjMapping.hf_to_megatron via
        _align_expert_weight_to_shape, which auto-detects the orientation difference between
        BF16 checkpoints ([num_experts, hidden, 2*intermediate]) and MXFP4-dequantized checkpoints
        ([num_experts, 2*intermediate, hidden]).
        """
        if isinstance(hf_param, str):
            if hf_param in hf_state_dict:
                hf_weights = hf_state_dict[hf_param]
                if hf_param.endswith(".mlp.experts.down_proj") and hf_weights.ndim == 3:
                    cfg = self.hf_pretrained.config
                    hidden = cfg.hidden_size
                    intermediate = cfg.intermediate_size
                    last2 = tuple(hf_weights.shape[-2:])
                    if last2 == (intermediate, hidden) and intermediate != hidden:
                        # Unambiguous BF16 layout (E, intermediate, hidden); transpose to (E, hidden, intermediate).
                        hf_weights = hf_weights.transpose(-1, -2).contiguous()
                    elif last2 == (hidden, intermediate) and intermediate != hidden:
                        # Already aligned with Megatron — no-op.
                        pass
                    elif intermediate == hidden:
                        # Square: HF GptOssForCausalLM init produces (E, intermediate, hidden), so a plain BF16
                        # checkpoint is in that layout. Transpose to (E, hidden, intermediate) for Megatron.
                        hf_weights = hf_weights.transpose(-1, -2).contiguous()
                return hf_weights
            blocks_key = hf_param + "_blocks"
            scales_key = hf_param + "_scales"
            if blocks_key in hf_state_dict and scales_key in hf_state_dict:
                hf_weights = _dequantize_mxfp4(hf_state_dict[blocks_key], hf_state_dict[scales_key])
                # MXFP4 dequant already emits [E, hidden, intermediate] for down_proj — leave as-is.
                return hf_weights
            raise KeyError(
                f"Cannot locate weights for '{hf_param}'. Missing both de-quantized tensor and "
                f"quantized representation (blocks='{blocks_key}', scales='{scales_key}')."
            )
        return {k: hf_state_dict[v] for k, v in hf_param.items()}

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings from HF to Megatron format.
        Based on the GPT-OSS importer code provided.
        """
        hf_config = self.hf_config
        num_hf_layers = hf_config.num_hidden_layers

        # Dictionary maps HF parameter names -> Megatron parameter names
        param_mappings = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for hf_param, megatron_param in param_mappings.items():
            mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))
        # HybridModel names the final normalization weight differently from GPTModel.
        mapping_list.append(AutoMapping(hf_param="model.norm.weight", megatron_param="decoder.final_norm.weight"))

        for hf_layer_idx in range(num_hf_layers):
            attention_layer_idx = 2 * hf_layer_idx
            moe_layer_idx = attention_layer_idx + 1
            mapping_list.extend(
                [
                    AutoMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.input_layernorm.weight",
                        megatron_param=(
                            f"decoder.layers.{attention_layer_idx}.self_attention.linear_qkv.layer_norm_weight"
                        ),
                    ),
                    AutoMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.self_attn.o_proj.bias",
                        megatron_param=f"decoder.layers.{attention_layer_idx}.self_attention.linear_proj.bias",
                    ),
                    AutoMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.self_attn.o_proj.weight",
                        megatron_param=f"decoder.layers.{attention_layer_idx}.self_attention.linear_proj.weight",
                    ),
                    AutoMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.self_attn.sinks",
                        megatron_param=(
                            f"decoder.layers.{attention_layer_idx}.self_attention.core_attention.softmax_offset"
                        ),
                    ),
                    QKVMapping(
                        q=f"model.layers.{hf_layer_idx}.self_attn.q_proj.weight",
                        k=f"model.layers.{hf_layer_idx}.self_attn.k_proj.weight",
                        v=f"model.layers.{hf_layer_idx}.self_attn.v_proj.weight",
                        megatron_param=f"decoder.layers.{attention_layer_idx}.self_attention.linear_qkv.weight",
                    ),
                    QKVMapping(
                        q=f"model.layers.{hf_layer_idx}.self_attn.q_proj.bias",
                        k=f"model.layers.{hf_layer_idx}.self_attn.k_proj.bias",
                        v=f"model.layers.{hf_layer_idx}.self_attn.v_proj.bias",
                        megatron_param=f"decoder.layers.{attention_layer_idx}.self_attention.linear_qkv.bias",
                    ),
                    AutoMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.post_attention_layernorm.weight",
                        megatron_param=f"decoder.layers.{moe_layer_idx}.pre_mlp_layernorm.weight",
                    ),
                    AutoMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.mlp.router.bias",
                        megatron_param=f"decoder.layers.{moe_layer_idx}.mlp.router.bias",
                    ),
                    AutoMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.mlp.router.weight",
                        megatron_param=f"decoder.layers.{moe_layer_idx}.mlp.router.weight",
                    ),
                    # Register the de-quantized weight names. If HF model is quantized,
                    # the logic in `modify_loaded_hf_weight` will find the blocks and scales tensors.
                    # Export is always de-quantized.
                    GPTOSSMLPDownProjMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.mlp.experts.down_proj",
                        megatron_param=f"decoder.layers.{moe_layer_idx}.mlp.experts.linear_fc2.weight*",
                    ),
                    GPTOSSMLPDownProjMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.mlp.experts.down_proj_bias",
                        megatron_param=f"decoder.layers.{moe_layer_idx}.mlp.experts.linear_fc2.bias*",
                    ),
                    GPTOSSMLPGateUpProjMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.mlp.experts.gate_up_proj",
                        megatron_param=f"decoder.layers.{moe_layer_idx}.mlp.experts.linear_fc1.weight*",
                    ),
                    GPTOSSMLPGateUpProjMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.mlp.experts.gate_up_proj_bias",
                        megatron_param=f"decoder.layers.{moe_layer_idx}.mlp.experts.linear_fc1.bias*",
                    ),
                    # SequentialMLP (moe_grouped_gemm=False): expert weights stored per local_expert.
                    GPTOSSMLPDownProjMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.mlp.experts.down_proj",
                        megatron_param=(
                            f"decoder.layers.{moe_layer_idx}.mlp.experts.local_experts.*.linear_fc2.weight"
                        ),
                    ),
                    GPTOSSMLPDownProjMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.mlp.experts.down_proj_bias",
                        megatron_param=(f"decoder.layers.{moe_layer_idx}.mlp.experts.local_experts.*.linear_fc2.bias"),
                    ),
                    GPTOSSMLPGateUpProjMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.mlp.experts.gate_up_proj",
                        megatron_param=(
                            f"decoder.layers.{moe_layer_idx}.mlp.experts.local_experts.*.linear_fc1.weight"
                        ),
                    ),
                    GPTOSSMLPGateUpProjMapping(
                        hf_param=f"model.layers.{hf_layer_idx}.mlp.experts.gate_up_proj_bias",
                        megatron_param=(f"decoder.layers.{moe_layer_idx}.mlp.experts.local_experts.*.linear_fc1.bias"),
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)


class GPTOSSMLPDownProjMapping(AutoMapping):
    """MLPDownProj for expert weights in GPT-OSS models."""

    is_grouped_export = True

    def __init__(self, megatron_param: str, hf_param: str, permute_dims: Optional[Tuple[int, ...]] = None):
        super().__init__(megatron_param, hf_param, permute_dims)
        self.allow_hf_name_mismatch = True

    @property
    def group_key(self) -> str:
        return self.hf_param

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        return super().hf_to_megatron(hf_weights[global_expert_number], megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> Dict[str, torch.Tensor]:
        # Megatron stores per-expert weight as (hidden, intermediate); HF down_proj
        # weight is (E, intermediate, hidden). Transpose the last two dims so the
        # grouped-export stack assembles in HF's layout. Under EP the parent's gather
        # may have already cat'd across the EP group, producing a 3D (ep_size, out, in)
        # tensor — handle that too. The bias has no orientation to align (per-expert
        # 1-D, stacked to (E, hidden) on export), so leave bias mappings untouched.
        if megatron_weights is not None:
            megatron_weights = megatron_weights.contiguous()
        result = super().megatron_to_hf(megatron_weights, megatron_module)
        if self.hf_param.endswith("_bias"):
            return result
        return {k: v.transpose(-1, -2).contiguous() if v.ndim >= 2 else v for k, v in result.items()}


class GPTOSSMLPGateUpProjMapping(AutoMapping):
    """MLPGateUpProj for expert weights in GPT-OSS models.

    GPT-OSS uses alternating row interleaving for gate/up projections.
    """

    is_grouped_export = True

    def __init__(self, megatron_param: str, hf_param: str, permute_dims: Optional[Tuple[int, ...]] = None):
        super().__init__(megatron_param, hf_param, permute_dims)
        self.allow_hf_name_mismatch = True

    @property
    def group_key(self) -> str:
        return self.hf_param

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
        expert_weight = hf_weights[global_expert_number] if hf_weights.ndim >= 2 else hf_weights
        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        expert_weight = _align_expert_weight_to_shape(expert_weight, target_param.shape, "gate_up_proj")
        return super().hf_to_megatron(self._interleave(expert_weight), megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> Dict[str, torch.Tensor]:
        if megatron_weights is None:
            return super().megatron_to_hf(megatron_weights, megatron_module)
        megatron_weights = self._uninterleave(megatron_weights)
        if len(megatron_weights.shape) == 2:
            megatron_weights = megatron_weights.transpose(0, 1)
        return super().megatron_to_hf(megatron_weights.contiguous(), megatron_module)
