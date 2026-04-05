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

"""Megatron Bridge for MiMo-V2-Flash (Hybrid Attention + Fine-Grained MoE).

MiMo-V2-Flash from Xiaomi features:
- Hybrid attention: alternating full and sliding-window attention layers
- Fine-grained MoE: 256 small experts with top-8 routing
- Asymmetric head dims: head_dim=192 for Q/K, v_head_dim=128 for V
- Partial rotary: only 33.4% of head dims get RoPE
- Dual rope bases: 5M (full attn) and 10K (SWA)
"""

from functools import partial
from typing import Mapping

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_provider import MiMoV2FlashModelProvider


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

_FP8_BLOCK_SIZE = 128


def _dequant_fp8_blockwise(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Block-wise FP8 dequantization: out = fp8_val * scale_inv per 128x128 block."""
    M, N = weight.shape
    B = _FP8_BLOCK_SIZE
    w = weight.float()
    out = torch.empty_like(w)
    sM, sN = scale_inv.shape
    for bi in range(sM):
        for bj in range(sN):
            r0, r1 = bi * B, min((bi + 1) * B, M)
            c0, c1 = bj * B, min((bj + 1) * B, N)
            out[r0:r1, c0:c1] = w[r0:r1, c0:c1] * scale_inv[bi, bj]
    return out.to(torch.bfloat16)


@MegatronModelBridge.register_bridge(
    source="MiMoV2FlashForCausalLM",
    target=GPTModel,
    provider=MiMoV2FlashModelProvider,
    model_type="mimo_v2_flash",
)
class MiMoV2FlashBridge(MegatronModelBridge):
    """Megatron Bridge for MiMo-V2-Flash."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MiMoV2FlashModelProvider:
        """Convert HuggingFace MiMo-V2-Flash config to MiMoV2FlashModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Required for mixed dense+MoE layers (moe_layer_freq)
        provider.transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)

        # Dual RoPE bases: (SWA local theta, full attention global theta)
        provider.rotary_base = (hf_config.swa_rope_theta, hf_config.rope_theta)

        # Hybrid attention pattern
        provider.hybrid_attention_pattern = list(hf_config.hybrid_layer_pattern)

        # Sliding window size for SWA layers
        provider.window_size = hf_config.sliding_window_size

        # Per-layer KV head counts (full attention vs SWA layers)
        provider.full_attn_num_query_groups = hf_config.num_key_value_heads
        provider.swa_num_query_groups = hf_config.swa_num_key_value_heads
        # base num_query_groups = full attention value (layer 0 is always full)
        provider.num_query_groups = provider.full_attn_num_query_groups

        # moe_layer_freq default on provider is int 1 — must override with list
        provider.moe_layer_freq = list(hf_config.moe_layer_freq)

        # noaux_tc: no auxiliary loss, learned expert bias
        provider.moe_router_load_balancing_type = "none"
        provider.moe_router_enable_expert_bias = True
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = True
        provider.moe_token_dispatcher_type = "alltoall"

        # Attention sink bias — learnable per-head offset on SWA layers
        # MiMoV2FlashTEDotProductAttention resets to "vanilla" for full attention layers
        provider.add_swa_attention_sink_bias = hf_config.add_swa_attention_sink_bias
        provider.add_full_attention_sink_bias = hf_config.add_full_attention_sink_bias

        # Attention value scale (0.707)
        provider.attention_value_scale = hf_config.attention_value_scale

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: MiMoV2FlashModelProvider) -> dict:
        """Convert Megatron provider config to HuggingFace config dict."""
        hf_cfg = super(MiMoV2FlashBridge, cls).megatron_to_hf_config(provider)

        # Dual RoPE bases
        if isinstance(provider.rotary_base, tuple):
            hf_cfg["swa_rope_theta"] = provider.rotary_base[0]
            hf_cfg["rope_theta"] = provider.rotary_base[1]

        # Hybrid attention pattern
        hf_cfg["hybrid_layer_pattern"] = provider.hybrid_attention_pattern

        # Sliding window
        window = provider.window_size
        hf_cfg["sliding_window_size"] = window
        hf_cfg["sliding_window"] = window

        # Per-layer KV heads
        hf_cfg["num_key_value_heads"] = provider.full_attn_num_query_groups
        hf_cfg["swa_num_key_value_heads"] = provider.swa_num_query_groups

        # MoE
        hf_cfg["moe_layer_freq"] = provider.moe_layer_freq

        # Attention sink bias
        hf_cfg["add_swa_attention_sink_bias"] = provider.add_swa_attention_sink_bias
        hf_cfg["add_full_attention_sink_bias"] = provider.add_full_attention_sink_bias

        # Attention value scale
        hf_cfg["attention_value_scale"] = provider.attention_value_scale

        # layernorm_epsilon: HF config uses this name, CONFIG_MAPPING maps rms_norm_eps so it's missed
        hf_cfg["layernorm_epsilon"] = provider.layernorm_epsilon

        return hf_cfg


    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []
        # TODO: How should I map first dense weight in first layer ?
        param_mappings = {
            # Embeddings
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Attention 
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.core_attention.softmax_offset": "model.layers.*.self_attn.attention_sink_bias",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # MoE
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
        }
        mtp_keys = {
            "enorm.weight" : "enorm.weight",
            "eh_proj.weight" : "eh_proj.weight",
            "final_layernorm.weight" : "final_layernorm.weight",
            "hnorm.weight" : "hnorm.weight",
            "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
            "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
            "self_attention.core_attention.softmax_offset": "self_attn.attention_sink_bias",
            "post_attention_layernorm.weight": "pre_mlp_layernorm.weight",
            "mlp.linear_fc2.weight": "mlp.down_proj.weight",
        }
        
        
        # TODO: Join logic for MTP and layers since the naming is the same


        # Support both naming conventions: Megatron-Core may expose MTP layers as
        # either "transformer_layer" or "mtp_model_layer" depending on configuration
        for layer_prefix in ("transformer_layer", "mtp_model_layer"):
            for megatron_mtp_key, hf_mtp_key in mtp_keys.items():
                megatron_param = f"mtp.layers.*.{layer_prefix}.{megatron_mtp_key}"
                hf_param = f"model.mtp.layers.*.{hf_mtp_key}"
                mapping_list.append(
                    AutoMapping(
                        megatron_param=megatron_param,
                        hf_param=hf_param,
                    )
                )
            layer_path = f"mtp.layers.*.{layer_prefix}"
            mapping_list.extend(
                [
                    QKVMapping(
                        megatron_param=f"{layer_path}.self_attention.linear_qkv.weight",
                        q="model.mtp.layers.*.self_attn.q_proj.weight",
                        k="model.mtp.layers.*.self_attn.k_proj.weight",
                        v="model.mtp.layers.*.self_attn.v_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"{layer_path}.mlp.linear_fc1.weight",
                        gate="model.mtp.layers.*.mlp.gate_proj.weight",
                        up="model.mtp.layers.*.mlp.up_proj.weight",
                    ),
                ]
            )


        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))
        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Dequantize FP8 weights during import.

        The released MiMo-V2-Flash checkpoint uses FP8 (e4m3) quantization with
        block-wise scaling (weight_block_size=[128, 128]). Weights are stored as
        float8_e4m3fn with accompanying *_scale_inv tensors. This hook dequantizes
        them to bfloat16 before loading into the Megatron model.
        """
        if isinstance(hf_param, dict):
            return {k: self._load_and_dequant(v, hf_state_dict) for k, v in hf_param.items()}
        return self._load_and_dequant(hf_param, hf_state_dict)

    def _load_and_dequant(self, key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        w = hf_state_dict[key]
        if not w.dtype == torch.float8_e4m3fn:
            return w
        sinv_key = key + "_scale_inv"
        if w.ndim == 2 and sinv_key in hf_state_dict:
            return _dequant_fp8_blockwise(w, hf_state_dict[sinv_key])
        return w.float().to(torch.bfloat16)
