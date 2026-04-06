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

from typing import Mapping

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    merge_qkv_weights,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_provider import MiMoV2FlashModelProvider


# softmax_offset (attention sink bias) is shape [num_heads], sharded along dim 0 by TP
# (see TE's sharded_state_dict: {'softmax_offset': 0})
AutoMapping.register_module_type("MiMoV2FlashTEDotProductAttention", "column")


class MiMoV2FlashQKVMapping(QKVMapping):
    """QKV mapping for MiMo-V2-Flash asymmetric head dims.

    MiMo-V2-Flash uses head_dim=192 for Q/K but v_head_dim=128 for V.
    Standard merge_qkv_weights uses kv_channels (192) for all three,
    causing a shape mismatch for V. We temporarily patch v_head_dim onto
    the config before merging.
    """

    def hf_to_megatron(self, hf_weights, megatron_module):
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            q, k, v = hf_weights["q"], hf_weights["k"], hf_weights["v"]
            if q.ndim == 2:
                num_heads = config.num_attention_heads
                num_qg = config.num_query_groups
                heads_per_group = num_heads // num_qg
                qk_ch = config.kv_channels
                v_ch = config.v_head_dim

                merged_rows = []
                for i in range(num_qg):
                    merged_rows.append(q[i * heads_per_group * qk_ch : (i + 1) * heads_per_group * qk_ch])
                    merged_rows.append(k[i * qk_ch : (i + 1) * qk_ch])
                    merged_rows.append(v[i * v_ch : (i + 1) * v_ch])
                merged = torch.cat(merged_rows, dim=0)
            else:
                merged = merge_qkv_weights(config, q, k, v)
        else:
            merged = None
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)


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

        # Dual RoPE bases: (SWA local theta, full attention global theta)
        provider.rotary_base = (hf_config.swa_rope_theta, hf_config.rope_theta)

        # Hybrid attention pattern
        provider.hybrid_attention_pattern = list(hf_config.hybrid_layer_pattern)

        # Sliding window size for SWA layers
        provider.window_size = hf_config.sliding_window_size

        # HF uses non-standard "layernorm_epsilon" name; CONFIG_MAPPING expects rms_norm_eps
        provider.layernorm_epsilon = hf_config.layernorm_epsilon

        # Asymmetric V head dimension (Q/K use kv_channels, V uses v_head_dim)
        provider.v_head_dim = hf_config.v_head_dim

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

        # Attention value scale
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

        # Asymmetric V head dim
        hf_cfg["v_head_dim"] = provider.v_head_dim

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

        # MCore renormalizes top-k probs for sigmoid scoring with topk > 1 (moe_utils.py:797)
        hf_cfg["norm_topk_prob"] = True

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
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.core_attention.softmax_offset": "model.layers.*.self_attn.attention_sink_bias",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # MoE
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
        }
        mapping_list.extend([
            # QKV projection — uses custom mapping to handle asymmetric v_head_dim
            MiMoV2FlashQKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            # Dense MLP gate+up
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            # MoE experts gate+up
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                up="model.layers.*.mlp.experts.*.up_proj.weight",
            ),
        ])
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
                    MiMoV2FlashQKVMapping(
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
