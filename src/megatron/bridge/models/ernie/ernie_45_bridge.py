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

"""
Megatron Bridge for ERNIE 4.5 text-only MoE model.

Maps HuggingFace Ernie4_5_MoeForCausalLM weights and config to
Megatron-Core GPTModel with single-pool MoE (64 experts, top-6 routing,
shared experts, expert bias for aux-free load balancing).
"""

from typing import Any

import torch.nn.functional as F
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.gpt_provider import GPTModelProvider


def _ernie45_decoder_block_spec(config: "GPTModelProvider", vp_stage: int | None = None):
    """Create a decoder block spec that respects ``moe_layer_freq``.

    The default ``GPTModelProvider.transformer_layer_spec`` calls
    ``get_gpt_layer_with_transformer_engine_spec`` which returns a single
    MoE layer spec applied uniformly to ALL layers, ignoring
    ``moe_layer_freq``.

    ERNIE 4.5 has mixed dense/MoE layers (layer 0 is dense, layers 1-N
    are MoE).  This function uses ``get_gpt_decoder_block_spec`` which
    calls ``get_gpt_decoder_layer_specs`` — the code path that parses
    ``config.moe_layer_freq`` and creates per-layer specs (dense for
    pattern=0, MoE for pattern=1).
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

    return get_gpt_decoder_block_spec(
        config=config,
        use_transformer_engine=True,
        vp_stage=vp_stage,
    )


# HF class name string; avoids requiring the HF modeling module at import time.
_ERNIE45_MOE_HF_CLASS_NAME = "Ernie4_5_MoeForCausalLM"


class _PPSafeMixin:
    """Mixin that makes ``megatron_to_hf`` safe for PP export of MoE-only params.

    When ``moe_layer_freq`` makes some layers dense and others MoE,
    MoE-only parameters (router weight, expert bias, shared/routed expert
    weights) do not exist on dense layers.  With PP > 1,
    ``broadcast_from_pp_rank`` raises ``ValueError`` because no PP rank
    owns the tensor.

    This mixin catches that error and returns ``{}`` so the conversion
    loop simply omits the parameter from the output.

    **Must be listed before the base mapping class in the MRO** so that
    ``super().megatron_to_hf`` resolves to the concrete mapping's method.
    """

    def megatron_to_hf(self, megatron_weights, megatron_module):
        try:
            return super().megatron_to_hf(megatron_weights, megatron_module)
        except ValueError:
            # Parameter doesn't exist on any PP rank (dense layer).
            return {}


class _PPSafeAutoMapping(_PPSafeMixin, AutoMapping):
    """AutoMapping that skips export for missing parameters."""

    pass


class _PPSafeReplicatedMapping(_PPSafeMixin, ReplicatedMapping):
    """ReplicatedMapping that skips export for missing parameters."""

    pass


class _PPSafeGatedMLPMapping(_PPSafeMixin, GatedMLPMapping):
    """GatedMLPMapping that skips export for missing parameters."""

    pass


class _SqueezeBiasMapping(_PPSafeReplicatedMapping):
    """Mapping for the single-pool expert bias tensor.

    The HF text-only model stores ``moe_statics.e_score_correction_bias``
    with shape ``[1, num_experts]`` (1 expert group for text-only).
    Megatron stores ``router.expert_bias`` as a flat ``[num_experts]`` tensor.

    This mapping squeezes dim-0 on import and unsqueezes on export.

    Inherits from ``_PPSafeReplicatedMapping`` to gracefully skip dense
    layers during PP export.
    """

    def hf_to_megatron(self, hf_weights, megatron_module):
        # [1, N] -> [N]
        if hf_weights.ndim == 2 and hf_weights.shape[0] == 1:
            hf_weights = hf_weights.squeeze(0)
        return super().hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        result = super().megatron_to_hf(megatron_weights, megatron_module)
        if result:
            # [N] -> [1, N]
            out = {}
            for k, v in result.items():
                out[k] = v.unsqueeze(0) if v.ndim == 1 else v
            return out
        return result


@MegatronModelBridge.register_bridge(
    source=_ERNIE45_MOE_HF_CLASS_NAME,
    target=GPTModel,
    provider=GPTModelProvider,
    model_type="ernie4_5_moe",
)
class Ernie45Bridge(MegatronModelBridge):
    """
    Megatron Bridge for ERNIE 4.5 text-only MoE Causal LM.

    This bridge handles the conversion between HuggingFace Ernie4_5_MoeForCausalLM
    and Megatron-Core GPTModel formats with single-pool MoE architecture.

    Key architectural features:
    - Single-pool MoE: 64 experts, top-6 routing, shared experts
    - Softmax routing with expert bias for aux-free load balancing
    - Interleaved RoPE (base=500000)
    - GQA with 20 query heads, 4 KV heads, kv_channels=128
    - RMSNorm, SiLU-gated MLP
    - Router gate weight stored as [H, E] in HF (transposed for Megatron [E, H])

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("baidu/ERNIE-4.5-0.3B-PT")
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = BridgeGPTModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    @staticmethod
    def _get_num_experts(hf_config) -> int:
        """Extract num_experts as an int.

        The config may store moe_num_experts as a plain int or as a list
        ``[N]`` (single pool) or ``[N, N]`` (dual pool -- take first).
        """
        raw = getattr(hf_config, "moe_num_experts", 64)
        if isinstance(raw, (list, tuple)):
            return raw[0]
        return int(raw)

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face ERNIE 4.5 MoE config to builder config kwargs.

        Starts from the shared ``CONFIG_MAPPING`` fields (hidden_size, num_layers,
        tie_word_embeddings, etc.) and adds the ERNIE-specific settings.
        """
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)

        # Expert intermediate size: may be int or list (text-only uses first/only).
        moe_intermediate = getattr(hf_config, "moe_intermediate_size", None)
        if isinstance(moe_intermediate, (list, tuple)):
            moe_ffn_hidden_size = moe_intermediate[0]
        elif moe_intermediate is not None:
            moe_ffn_hidden_size = moe_intermediate
        else:
            moe_ffn_hidden_size = getattr(hf_config, "intermediate_size", 5120)

        config_kwargs.update(
            # --- Architecture ---
            normalization="RMSNorm",
            activation_func=F.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            add_qkv_bias=False,
            hidden_dropout=0.0,
            position_embedding_type="rope",
            rotary_base=500000.0,
            rotary_interleaved=True,
            masked_softmax_fusion=True,
            rope_scaling=False,
            rope_scaling_factor=1.0,
            # --- MoE (ERNIE uses non-standard HF config field names) ---
            num_moe_experts=self._get_num_experts(hf_config),
            moe_router_topk=getattr(hf_config, "moe_k", 6),
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            moe_shared_expert_intermediate_size=moe_ffn_hidden_size * getattr(hf_config, "moe_num_shared_experts", 2),
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=getattr(hf_config, "router_aux_loss_coef", 0.001),
            # Same MoE runtime settings as DeepSeek V3 (sigmoid routing + expert bias)
            moe_grouped_gemm=True,
            moe_router_pre_softmax=False,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_dtype="fp32",
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            # The HF config stores num_nextn_predict_layers, but the checkpoints ship
            # no MTP weights, so Megatron must not create MTP layers.
            mtp_num_layers=None,
        )

        # Determine which layers are dense vs MoE.
        mlp_layer_types = getattr(hf_config, "mlp_layer_types", None)
        if mlp_layer_types is not None:
            config_kwargs["moe_layer_freq"] = [0 if t == "dense" else 1 for t in mlp_layer_types]
        else:
            num_layers = hf_config.num_hidden_layers
            moe_start = getattr(hf_config, "moe_layer_start_index", None)
            if moe_start is not None:
                start = moe_start[0] if isinstance(moe_start, (list, tuple)) else moe_start
                config_kwargs["moe_layer_freq"] = [0] * start + [1] * (num_layers - start)
            else:
                # Default: layer 0 dense, rest MoE
                config_kwargs["moe_layer_freq"] = [0] + [1] * (num_layers - 1)
        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Adapt the canonical builder mapping to the deprecated provider path.

        The builder's default MoE spec already honors ``moe_layer_freq``; the
        provider's default spec does not, so the provider path keeps the
        decoder block spec explicitly.
        """
        provider_kwargs = self.hf_config_to_model_config_kwargs(hf_config)
        provider_kwargs["transformer_layer_spec"] = _ernie45_decoder_block_spec
        return provider_kwargs

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return MegatronMappingRegistry with parameter mappings for ERNIE 4.5 MoE."""
        # Simple 1:1 parameter mappings
        param_mappings = {
            # Embeddings & output
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Attention
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": ("model.layers.*.input_layernorm.weight"),
            "decoder.layers.*.self_attention.linear_proj.weight": ("model.layers.*.self_attn.o_proj.weight"),
            # Dense MLP layernorm (layer 0 -- fused into linear_fc1)
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": ("model.layers.*.post_attention_layernorm.weight"),
            # Dense MLP down_proj (layer 0)
            "decoder.layers.*.mlp.linear_fc2.weight": ("model.layers.*.mlp.down_proj.weight"),
            # MoE layers: pre_mlp_layernorm
            "decoder.layers.*.pre_mlp_layernorm.weight": ("model.layers.*.post_attention_layernorm.weight"),
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # MoE-only parameters use PP-safe variants that gracefully return {}
        # when the parameter doesn't exist on any PP rank (dense layers
        # created by moe_layer_freq).

        # Shared experts: down_proj (MoE-only)
        mapping_list.append(
            _PPSafeAutoMapping(
                megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                hf_param="model.layers.*.mlp.shared_experts.down_proj.weight",
            )
        )

        mapping_list.extend(
            [
                # =============================================================
                # QKV: Combine separate Q, K, V into fused QKV
                # =============================================================
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # =============================================================
                # Dense MLP (layer 0): gate_proj + up_proj -> fused linear_fc1
                # =============================================================
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                # =============================================================
                # Router weight: HF text-only model stores [E, H] (same as
                # Megatron), so no transpose needed.  Note: the VL model
                # stores the gate weight transposed as [H, E] and needs
                # permute_dims=(1, 0); the text-only model does not.
                #
                # Uses ``_PPSafeReplicatedMapping`` because TopKRouter.weight
                # is replicated across TP ranks, and dense layers (created
                # by ``moe_layer_freq``) have no router — the PP-safe
                # variant gracefully returns ``{}`` for those layers.
                # =============================================================
                _PPSafeReplicatedMapping(
                    megatron_param="decoder.layers.*.mlp.router.weight",
                    hf_param="model.layers.*.mlp.gate.weight",
                ),
                # =============================================================
                # MoE expert mappings for TEGroupedMLP (fused 3D tensors)
                # =============================================================
                _PPSafeGatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                _PPSafeAutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # =============================================================
                # MoE expert mappings for SequentialMLP (per-expert, for quantization)
                # =============================================================
                _PPSafeGatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                _PPSafeAutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # =============================================================
                # Shared experts: gate+up -> fused linear_fc1
                # =============================================================
                _PPSafeGatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
                # =============================================================
                # Expert bias: [1, N] on disk -> [N] in Megatron
                # =============================================================
                _SqueezeBiasMapping(
                    megatron_param="decoder.layers.*.mlp.router.expert_bias",
                    hf_param="model.layers.*.mlp.moe_statics.e_score_correction_bias",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
