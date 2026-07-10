# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from functools import partial

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
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


__all__ = ["HYV3Bridge"]


@MegatronModelBridge.register_bridge(
    source="HYV3ForCausalLM",
    target=GPTModel,
    provider=GPTModelProvider,
    model_type="hy_v3",
)
class HYV3Bridge(MegatronModelBridge):
    """Megatron Bridge for Hy V3 MoE Causal LM.

    This bridge handles the conversion between HuggingFace HYV3ForCausalLM
    and Megatron-Core GPTModel formats. Hy V3 is a fine-grained MoE decoder
    with GQA + QK layernorm, dense-first layer freq, sigmoid router with
    per-expert bias, shared experts, and grouped-GEMM routed experts.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("tencent/Hy3-preview-Base")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace Hy V3 config to GPTModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False  # Hy V3 does NOT have QKV bias
        provider.hidden_dropout = 0.0
        provider.qk_layernorm = True  # Hy V3 uses QK layernorm
        provider.attention_softmax_in_fp32 = False
        provider.autocast_dtype = torch.bfloat16
        provider.fp16 = False
        provider.bf16 = True
        provider.params_dtype = torch.bfloat16
        provider.moe_grouped_gemm = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "none"
        provider.moe_router_pre_softmax = False
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = 0
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True
        provider.moe_shared_expert_overlap = False
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_router_topk_scaling_factor = float(hf_config.router_scaling_factor)

        # Shared experts hidden dim = moe_intermediate_size * num_shared_experts.
        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * hf_config.num_shared_experts

        # Dense-first layer pattern.
        provider.moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: GPTModelProvider) -> dict:
        """Convert Megatron provider config to HuggingFace HYV3Config dict."""
        hf_config = super().megatron_to_hf_config(provider)
        moe_ffn = provider.moe_ffn_hidden_size
        hf_config["moe_intermediate_size"] = moe_ffn
        hf_config["expert_hidden_dim"] = moe_ffn  # legacy alias

        # Reconstruct num_shared_experts from moe_shared_expert_intermediate_size / moe_ffn_hidden_size
        shared_size = getattr(provider, "moe_shared_expert_intermediate_size", None)
        moe_ffn_hidden = getattr(provider, "moe_ffn_hidden_size", None)
        if shared_size is not None and moe_ffn_hidden is not None and moe_ffn_hidden > 0:
            hf_config["num_shared_experts"] = shared_size // moe_ffn_hidden

        hf_config["moe_router_use_sigmoid"] = True
        hf_config["moe_router_enable_expert_bias"] = True
        hf_config["qk_norm"] = True
        hf_config["route_norm"] = True
        hf_config["router_scaling_factor"] = provider.moe_router_topk_scaling_factor

        # Reconstruct first_k_dense_replace from moe_layer_freq (count leading dense layers)
        moe_layer_freq = getattr(provider, "moe_layer_freq", None)
        if moe_layer_freq is not None and isinstance(moe_layer_freq, list):
            first_k_dense_replace = 0
            for val in moe_layer_freq:
                if val == 0:
                    first_k_dense_replace += 1
                else:
                    break
            hf_config["first_k_dense_replace"] = first_k_dense_replace

        # Megatron uses None="not set/disabled", but HF expects integers
        hf_config["num_nextn_predict_layers"] = hf_config.get("num_nextn_predict_layers") or 0
        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format.
        # First create simple 1:1 parameter mappings using a dictionary for readability.
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # TE spec fuses input_layernorm into linear_qkv even when qk_layernorm=True,
            # so the standalone input_layernorm.weight param does not exist. Same as Qwen3-MoE.
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": ("model.layers.*.input_layernorm.weight"),
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Post-attention norm — MoE layers use pre_mlp_layernorm; dense layers
            # use the fused variant on linear_fc1. Same HF weight for both.
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # Router.
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.router.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.expert_bias",
            # Dense MLP down_proj (first_k_dense_replace layers).
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # Grouped-GEMM expert down_proj + sequential fallback (used when quantized).
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": (
                "model.layers.*.mlp.experts.*.down_proj.weight"
            ),
            # Shared expert down_proj.
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": (
                "model.layers.*.mlp.shared_mlp.down_proj.weight"
            ),
        }

        mapping_list = [AutoMapping(megatron_param=m, hf_param=h) for m, h in param_mappings.items()]

        # Add special mappings that require parameter concatenation/transformation.
        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # Dense MLP gate+up (first_k_dense_replace layers).
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                # Expert mappings for TEGroupedMLP.
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                # Expert mappings for SequentialMLP (used by quantization).
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                # Shared experts gate+up.
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_mlp.gate_proj.weight",
                    up="model.layers.*.mlp.shared_mlp.up_proj.weight",
                ),
            ]
        )

        # MTP layer mappings (layer num_hidden_layers in the HF checkpoint).
        hf_config = self.hf_config
        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)
        num_transformer_layers = hf_config.num_hidden_layers
        for mtp_layer in range(num_mtp_layers):
            hf_layer_idx = mtp_layer + num_transformer_layers
            # Reuse per-layer 1:1 mappings for the MTP layer's inner transformer.
            for megatron_param, hf_param in param_mappings.items():
                if ".layers.*." not in megatron_param:
                    continue  # skip non-layer entries (embed, final norm, lm_head)
                megatron_param_mtp = (
                    megatron_param.replace(".*", ".*.mtp_model_layer", 1)
                    .replace("decoder", "mtp")
                    .replace(".*", f".{mtp_layer}", 1)
                )
                hf_param_mtp = hf_param.replace("layers.*", f"layers.{hf_layer_idx}")
                mapping_list.append(AutoMapping(megatron_param=megatron_param_mtp, hf_param=hf_param_mtp))

            # MTP-specific heads: enorm / hnorm / eh_proj / final_layernorm.
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.enorm.weight",
                        hf_param=f"model.layers.{hf_layer_idx}.enorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.hnorm.weight",
                        hf_param=f"model.layers.{hf_layer_idx}.hnorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.eh_proj.weight",
                        hf_param=f"model.layers.{hf_layer_idx}.eh_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                        hf_param=f"model.layers.{hf_layer_idx}.final_layernorm.weight",
                    ),
                ]
            )

            # MTP inner transformer QKV and gated MLP mappings.
            mapping_list.extend(
                [
                    QKVMapping(
                        megatron_param=(f"mtp.layers.{mtp_layer}.mtp_model_layer.self_attention.linear_qkv.weight"),
                        q=f"model.layers.{hf_layer_idx}.self_attn.q_proj.weight",
                        k=f"model.layers.{hf_layer_idx}.self_attn.k_proj.weight",
                        v=f"model.layers.{hf_layer_idx}.self_attn.v_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.linear_fc1.weight",
                        gate=f"model.layers.{hf_layer_idx}.mlp.gate_proj.weight",
                        up=f"model.layers.{hf_layer_idx}.mlp.up_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=(f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.experts.linear_fc1.weight*"),
                        gate=f"model.layers.{hf_layer_idx}.mlp.experts.*.gate_proj.weight",
                        up=f"model.layers.{hf_layer_idx}.mlp.experts.*.up_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=(
                            f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.experts.local_experts.*.linear_fc1.weight"
                        ),
                        gate=f"model.layers.{hf_layer_idx}.mlp.experts.*.gate_proj.weight",
                        up=f"model.layers.{hf_layer_idx}.mlp.experts.*.up_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=(
                            f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.shared_experts.linear_fc1.weight"
                        ),
                        gate=f"model.layers.{hf_layer_idx}.mlp.shared_mlp.gate_proj.weight",
                        up=f"model.layers.{hf_layer_idx}.mlp.shared_mlp.up_proj.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)
