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
Megatron Bridge for Gemma 4 text-only (CausalLM).

Gemma 4 is a MoE model with hybrid sliding/global attention. The dense MLP
is mapped to Megatron-Core's shared expert mechanism, and routed experts
use fused tensor format ``[num_experts, 2*intermediate, hidden]``.

Key architecture-specific handling:
- K=V on global attention layers: ``v_proj`` is absent; K weights are copied to V.
- Dual pre-norms: separate norms for dense MLP vs routed experts.
- Router scale/per_expert_scale: loaded as replicated buffers.
- layer_scalar: per-layer scaling buffer.

Supported models
----------------
- ``google/gemma-4-26B-A4B`` (MoE, ``enable_moe_block=True``) — fully supported.

NOT supported
-------------
- Dense Gemma 4 models (``enable_moe_block=False``, e.g. a hypothetical text-only e2b).
  The currently available dense model is ``google/gemma-4-e2b-it``, which is a VLM and
  is handled by ``gemma4_vl_bridge.py`` instead.

  Dense Gemma 4 models introduce two architectural features that are disabled (but present)
  in the MoE variant (26B-A4B has hidden_size_per_layer_input=0, use_double_wide_mlp=False):
  1. ``use_double_wide_mlp``: the last ``num_kv_shared_layers`` layers have 2× FFN hidden size.
     Supporting this requires per-layer ``ffn_hidden_size`` in MCore.
  2. **Per-Layer Embeddings (PLE)**: a second low-dimensional embedding pathway injected
     into every decoder layer after the FFN.  See ``gemma4_vl_bridge.py`` for details on
     what PLE does and what needs to be implemented in MCore to support it.

  To support dense Gemma 4 CausalLM, this bridge would need the same MoE/dense branching
  that ``gemma4_vl_bridge.py`` already has, plus full MCore implementations of PLE and
  variable-width FFN.
"""

import math
import re
from typing import Mapping

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.transformers_compat import (
    rope_local_base_freq_from_hf,
    rope_theta_from_hf,
)
from megatron.bridge.models.gemma.gemma4_provider import Gemma4ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# Register Gemma4 custom module types for AutoMapping
AutoMapping.register_module_type("Gemma4TEDotProductAttention", "replicated")
AutoMapping.register_module_type("Gemma4SelfAttention", "replicated")
AutoMapping.register_module_type("Gemma4TransformerLayer", "replicated")
AutoMapping.register_module_type("Gemma4TopKRouter", "replicated")
AutoMapping.register_module_type("Gemma4MoELayer", "replicated")
AutoMapping.register_module_type("SharedExpertMLP", "column")


class _Gemma4QKVMapping(QKVMapping):
    """QKV mapping that tolerates missing v_proj in the HF checkpoint.

    Gemma 4 global attention layers share K=V, so v_proj is absent.
    ``allow_hf_name_mismatch = True`` prevents the weight loader from
    skipping the entire QKV mapping; the V weights are synthesized from K
    in ``Gemma4Bridge.maybe_modify_loaded_hf_weight``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_hf_name_mismatch = True


@MegatronModelBridge.register_bridge(
    source="Gemma4ForCausalLM",
    target=GPTModel,
    provider=Gemma4ModelProvider,
    model_type="gemma4",
)
class Gemma4Bridge(MegatronModelBridge):
    """
    Megatron Bridge for Gemma 4 text-only (CausalLM).

    Handles conversion between HuggingFace Gemma4ForCausalLM and
    Megatron-Core GPTModel with MoE + shared experts.

    Architecture mapping:
    - Dense MLP → Megatron shared experts (``moe_shared_expert_overlap=False``)
    - Routed MoE → Megatron routed experts (fused expert format)
    - Sliding attention → standard kv_channels/num_query_groups
    - Global attention → overridden kv_channels/num_query_groups per layer

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("google/gemma-4-12B-A2B")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Gemma4ModelProvider:
        """Convert HuggingFace config to Gemma4ModelProvider."""
        hf_config = hf_pretrained.config

        # Use base class helper for common config conversion
        provider_kwargs = self.hf_config_to_provider_kwargs(hf_config)
        provider = Gemma4ModelProvider(**provider_kwargs)

        # Gemma 4 specific features not in CONFIG_MAPPING
        provider.window_size = getattr(hf_config, "sliding_window", 1024)

        # Dual RoPE bases: local (sliding) and global (full attention)
        provider.rotary_base = (
            rope_local_base_freq_from_hf(hf_config),
            rope_theta_from_hf(hf_config),
        )

        # Gemma 4 uses QK norm — no 1/sqrt(d) scaling on attention logits
        head_dim = getattr(hf_config, "head_dim", 256)
        provider.softmax_scale = 1.0
        provider.kv_channels = head_dim
        provider.qk_layernorm = True

        # Global attention overrides
        provider.global_head_dim = getattr(hf_config, "global_head_dim", 512)
        provider.num_global_key_value_heads = getattr(hf_config, "num_global_key_value_heads", 2)

        # Parse partial_rotary_factor from rope_parameters for global attention
        rope_params = getattr(hf_config, "rope_parameters", {})
        if isinstance(rope_params, dict):
            full_attn_rope = rope_params.get("full_attention", {})
            provider.global_rotary_percent = full_attn_rope.get("partial_rotary_factor", 0.25)

        # Sliding/global layer pattern
        layer_types = getattr(hf_config, "layer_types", None)
        if layer_types:
            provider.interleaved_attn_pattern = _infer_attn_pattern(layer_types)

        # MoE configuration
        provider.num_moe_experts = getattr(hf_config, "num_experts", 128)
        provider.moe_router_topk = getattr(hf_config, "top_k_experts", 8)
        provider.moe_ffn_hidden_size = getattr(hf_config, "moe_intermediate_size", 704)

        # Dense MLP intermediate → shared expert
        provider.moe_shared_expert_intermediate_size = getattr(hf_config, "intermediate_size", 2112)
        provider.moe_shared_expert_overlap = False  # Must be False: Gemma4 needs separate pre/post norms
        provider.moe_shared_expert_gate = False
        provider.moe_layer_freq = 1  # all layers are MoE

        # Logit softcapping
        provider.final_logit_softcapping = getattr(hf_config, "final_logit_softcapping", 30.0)

        # Override dtype and vocab settings
        provider.bf16 = True
        provider.params_dtype = torch.bfloat16
        provider.autocast_dtype = torch.bfloat16
        provider.make_vocab_size_divisible_by = 128

        return provider

    def maybe_modify_converted_hf_weight(
        self,
        task,
        converted_weights_dict,
        hf_state_dict,
    ):
        """Un-fuse fused weights and drop synthesized keys on export.

        On import, two non-trivial fusions are applied to the MoE layers:

        1. **Router fusion**: ``mg = hf * (scale * hidden^-0.5 / pffl2)``
        2. **Shared-expert gate/up fusion**: ``mg = hf * (pffl / pffl2)``

        This method inverts both fusions on export so the resulting HF weights
        exactly match the original checkpoint.  It also drops the synthesized
        ``v_proj`` key produced for K=V global-attention layers where ``v_proj``
        is absent in HF.
        """
        if not hf_state_dict:
            return converted_weights_dict

        result = {}
        for hf_name, tensor in converted_weights_dict.items():
            # Drop synthesized v_proj (absent for K=V global-attention layers)
            if hf_name not in hf_state_dict:
                continue

            # ── Router weight inverse: hf = mg * pffl2 / (scale * hidden^-0.5)
            if hf_name.endswith("router.proj.weight"):
                layer_match = re.search(r"layers\.(\d+)\.", hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    prefix = hf_name.rsplit("layers.", 1)[0]
                    scale_key = f"{prefix}layers.{layer_idx}.router.scale"
                    ln2_key = f"{prefix}layers.{layer_idx}.pre_feedforward_layernorm_2.weight"
                    if scale_key in hf_state_dict and ln2_key in hf_state_dict:
                        router_scale = hf_state_dict[scale_key].float().to(tensor.device)
                        ln2_weight = hf_state_dict[ln2_key].float().to(tensor.device)
                        hidden_size = tensor.shape[-1]
                        scalar_root_size = hidden_size ** -0.5
                        fusion_factor = router_scale * scalar_root_size / ln2_weight
                        tensor = (tensor.float() / fusion_factor.unsqueeze(0)).to(tensor.dtype)

            # ── Shared-expert gate/up inverse: hf = mg * (pffl2 / pffl)
            elif (
                hf_name.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight"))
                and "experts" not in hf_name
            ):
                layer_match = re.search(r"layers\.(\d+)\.", hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    prefix = hf_name.rsplit("layers.", 1)[0]
                    pffl_key = f"{prefix}layers.{layer_idx}.pre_feedforward_layernorm.weight"
                    pffl2_key = f"{prefix}layers.{layer_idx}.pre_feedforward_layernorm_2.weight"
                    if pffl_key in hf_state_dict and pffl2_key in hf_state_dict:
                        w_pffl = hf_state_dict[pffl_key].float().to(tensor.device)
                        w_pffl2 = hf_state_dict[pffl2_key].float().to(tensor.device)
                        correction = w_pffl / w_pffl2
                        tensor = (tensor.float() / correction.unsqueeze(0)).to(tensor.dtype)

            result[hf_name] = tensor

        return result

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Handle special weight loading for Gemma 4.

        1. K=V on global attention layers: synthesize ``v_proj`` from ``k_proj``.
        2. Router weight fusion: absorb ``router.scale * scalar_root_size / (1 + ln2_weight)``
           into ``router.proj.weight`` so MCore's router produces correct logits when
           receiving ``pre_feedforward_layernorm_2``-normed input.
        3. Shared expert pre-norm fusion: absorb the ratio
           ``(1 + pre_feedforward_layernorm) / (1 + pre_feedforward_layernorm_2)`` into
           shared expert gate/up weights so the shared expert effectively receives
           ``pre_feedforward_layernorm``-normed input even though MCore feeds it
           ``pre_feedforward_layernorm_2``-normed input.
        """
        # Handle K=V on global layers
        if isinstance(hf_param, dict) and "v" in hf_param:
            v_name = hf_param["v"]
            if v_name not in hf_state_dict:
                k_name = hf_param["k"]
                hf_weights = {}
                for role, name in hf_param.items():
                    if role == "v":
                        hf_weights[role] = hf_state_dict[k_name].clone()
                    else:
                        hf_weights[role] = hf_state_dict[name]
                return hf_weights

        # Fuse pre-norm correction into shared expert gate/up weights
        if isinstance(hf_param, dict) and "gate" in hf_param:
            gate_name = hf_param["gate"]
            if "mlp.gate_proj" in gate_name:
                return self._fuse_shared_expert_prenorm(hf_param, hf_state_dict)

        # Fuse router scaling into router.proj.weight
        if isinstance(hf_param, str) and hf_param.endswith("router.proj.weight"):
            return self._fuse_router_weight(hf_param, hf_state_dict)

        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

    def _fuse_router_weight(
        self, hf_param: str, hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Fuse router preprocessing into projection weight.

        HF router: logits = proj(rms_norm(x) * scale * scalar_root_size)
        MCore router: logits = weight @ pre_feedforward_layernorm_2(x)

        Since rms_norm(x) = pre_feedforward_layernorm_2(x) / ln2_weight
        (Gemma 4 uses standard gamma: x * w / rms(x)),
        we fuse: new_weight = proj.weight * (scale * scalar_root_size / ln2_weight)
        """
        proj_weight = hf_state_dict[hf_param]  # [num_experts, hidden]

        # Extract layer index from param name
        layer_match = re.search(r"layers\.(\d+)\.", hf_param)
        if layer_match is None:
            return proj_weight
        layer_idx = layer_match.group(1)

        # Get router.scale and pre_feedforward_layernorm_2.weight for this layer
        scale_key = f"model.layers.{layer_idx}.router.scale"
        ln2_key = f"model.layers.{layer_idx}.pre_feedforward_layernorm_2.weight"

        if scale_key not in hf_state_dict or ln2_key not in hf_state_dict:
            return proj_weight

        router_scale = hf_state_dict[scale_key].float()  # [hidden]
        ln2_weight = hf_state_dict[ln2_key].float()  # [hidden]
        hidden_size = proj_weight.shape[-1]
        scalar_root_size = hidden_size ** -0.5

        # Compute fusion factor: scale * scalar_root_size / ln2_weight
        # This corrects for the difference between parameter-free rms_norm
        # (used by HF router) and MCore's pre_mlp_layernorm (x * w / rms(x)).
        # Gemma 4 uses STANDARD gamma (not zero-centered), so the norm weight
        # directly multiplies: pre_mlp_ln(x) = x * w / rms(x).
        fusion_factor = router_scale * scalar_root_size / ln2_weight

        # Fuse into weight: new_weight[i, j] = proj_weight[i, j] * fusion_factor[j]
        fused_weight = proj_weight.float() * fusion_factor.unsqueeze(0)
        return fused_weight.to(proj_weight.dtype)

    def _fuse_shared_expert_prenorm(
        self, hf_param: dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Fuse pre-norm correction into shared expert gate/up weights.

        MCore feeds shared experts ``pre_feedforward_layernorm_2(x)`` but HF feeds them
        ``pre_feedforward_layernorm(x)``.  Since both norms are standard RMSNorm
        (``x * w / rms(x)``), the correction is element-wise:

            correction[j] = w_pffl[j] / w_pffl2[j]
            new_weight[i, j] = weight[i, j] * correction[j]
        """
        gate_name = hf_param["gate"]
        layer_match = re.search(r"layers\.(\d+)\.", gate_name)
        if layer_match is None:
            return {role: hf_state_dict[name] for role, name in hf_param.items()}

        layer_idx = layer_match.group(1)
        pffl_key = f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight"
        pffl2_key = f"model.layers.{layer_idx}.pre_feedforward_layernorm_2.weight"

        if pffl_key not in hf_state_dict or pffl2_key not in hf_state_dict:
            return {role: hf_state_dict[name] for role, name in hf_param.items()}

        w_pffl = hf_state_dict[pffl_key].float()
        w_pffl2 = hf_state_dict[pffl2_key].float()
        correction = w_pffl / w_pffl2  # [hidden_size]

        hf_weights = {}
        for role, name in hf_param.items():
            weight = hf_state_dict[name]
            # weight shape: [intermediate_size, hidden_size] — correct along hidden dim
            fused = weight.float() * correction.unsqueeze(0)
            hf_weights[role] = fused.to(weight.dtype)
        return hf_weights

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Define parameter mappings between Megatron and HF formats.

        HF param names use ``model.layers.*`` prefix (text-only CausalLM).
        The VLM bridge overrides this with ``model.language_model.layers.*``.
        """
        param_mappings = {
            # === Embeddings ===
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",

            # === Per-layer attention ===
            # TE backend: layernorm fused into QKV linear
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                "model.layers.*.input_layernorm.weight"
            ),
            # Local (non-TE) backend fallback
            "decoder.layers.*.input_layernorm.weight": (
                "model.layers.*.input_layernorm.weight"
            ),
            "decoder.layers.*.self_attention.q_layernorm.weight": (
                "model.layers.*.self_attn.q_norm.weight"
            ),
            "decoder.layers.*.self_attention.k_layernorm.weight": (
                "model.layers.*.self_attn.k_norm.weight"
            ),
            "decoder.layers.*.self_attention.linear_proj.weight": (
                "model.layers.*.self_attn.o_proj.weight"
            ),
            # Post-attention RMSNorm (Gemma 4 applies this after attention, before residual)
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),

            # === Pre-MLP layernorm ===
            # MCore uses a single pre_mlp_layernorm for both shared and routed experts.
            # Gemma 4 has separate pre-norms: pre_feedforward_layernorm (dense)
            # and pre_feedforward_layernorm_2 (MoE).  We map the MoE pre-norm since
            # MCore's router also receives the normed input.
            "decoder.layers.*.pre_mlp_layernorm.weight": (
                "model.layers.*.pre_feedforward_layernorm_2.weight"
            ),

            # === Dense MLP → Shared Expert ===
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": (
                "model.layers.*.mlp.down_proj.weight"
            ),
            # Post-dense-MLP RMSNorm (Gemma 4: post_feedforward_layernorm_1)
            "decoder.layers.*.mlp.shared_experts.linear_fc2.post_layernorm.weight": (
                "model.layers.*.post_feedforward_layernorm_1.weight"
            ),

            # === MoE Router ===
            "decoder.layers.*.mlp.router.weight": "model.layers.*.router.proj.weight",

            # === MoE Router ===
            # router.scale is fused into router.weight on import; stored as an inert buffer
            # (Gemma4TopKRouter.scale) so it round-trips on export without needing the
            # reference HF checkpoint.  Mapped via ReplicatedMapping below.
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend([
            # === QKV: Combine Q, K, V into single QKV matrix ===
            # Uses _Gemma4QKVMapping which sets allow_hf_name_mismatch=True so
            # the loader doesn't skip global layers where v_proj is absent (K=V).
            # V is synthesized from K in maybe_modify_loaded_hf_weight.
            _Gemma4QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),

            # === Dense MLP → Shared Expert gated FC1 ===
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),

            # === MoE Experts (fused format) ===
            # gate_up_proj: [num_experts, 2*moe_intermediate, hidden]
            FusedGatedExpertMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                hf_param="model.layers.*.experts.gate_up_proj",
            ),
            # down_proj: [num_experts, hidden, moe_intermediate]
            FusedExpertMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                hf_param="model.layers.*.experts.down_proj",
            ),

            # === Per-layer output scaling (buffer) ===
            ReplicatedMapping(
                megatron_param="decoder.layers.*.layer_scalar",
                hf_param="model.layers.*.layer_scalar",
            ),

            # === Router per-expert scaling (buffer on Gemma4TopKRouter) ===
            ReplicatedMapping(
                megatron_param="decoder.layers.*.mlp.router.per_expert_scale",
                hf_param="model.layers.*.router.per_expert_scale",
            ),

            # === Router input scale (fused into router weight on import; stored as buffer) ===
            ReplicatedMapping(
                megatron_param="decoder.layers.*.mlp.router.scale",
                hf_param="model.layers.*.router.scale",
            ),

            # === Dense/shared-expert pre-norm (fused into gate/up on import; stored as buffer) ===
            ReplicatedMapping(
                megatron_param="decoder.layers.*.pffl_weight",
                hf_param="model.layers.*.pre_feedforward_layernorm.weight",
            ),

            # === Post-MoE layernorm (applied to routed expert output before combining) ===
            ReplicatedMapping(
                megatron_param="decoder.layers.*.mlp.post_moe_layernorm.weight",
                hf_param="model.layers.*.post_feedforward_layernorm_2.weight",
            ),

            # === Post-feedforward layernorm (after combined dense+MoE, before residual) ===
            ReplicatedMapping(
                megatron_param="decoder.layers.*.post_ffn_layernorm.weight",
                hf_param="model.layers.*.post_feedforward_layernorm.weight",
            ),
        ])

        return MegatronMappingRegistry(*mapping_list)


def _infer_attn_pattern(layer_types: list[str]) -> tuple[int, int]:
    """Infer (sliding, global) interleaved attention pattern from layer_types list.

    E.g., ["sliding", "sliding", ..., "full", "sliding", ...] with 5 sliding + 1 full
    returns (5, 1).
    """
    # Find the first occurrence of "full_attention" to determine the pattern
    for i, lt in enumerate(layer_types):
        if lt == "full_attention":
            sliding_count = i
            # Count consecutive full attention layers
            full_count = 0
            for j in range(i, len(layer_types)):
                if layer_types[j] == "full_attention":
                    full_count += 1
                else:
                    break
            return (sliding_count, full_count)

    # Fallback: all sliding
    return (len(layer_types), 0)
