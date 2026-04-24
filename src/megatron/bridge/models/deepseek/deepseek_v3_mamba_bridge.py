# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""DeepSeek-V3 bridge variant that targets `MambaModel` instead of `GPTModel`.

The default `DeepSeekV3Bridge` instantiates a Megatron-Core `GPTModel`.
This module adds an opt-in subclass that swaps the base model class to
`MambaModel` (the hybrid Mamba/attention stack), so the MLA and MoE
configuration coming from the HF checkpoint can be combined with Mamba/hybrid
layer layouts.

Layout mapping (HF → Megatron decoder layers):

    Each HF DSv3 block `i` is `[MLA, MLP-or-MoE]`. In the Mamba
    hybrid stack that block is unrolled into two positions:

        - Megatron decoder layer `2*i`     – MLA (`+`)
        - Megatron decoder layer `2*i + 1` – MLP (`-`) or MoE (`E`)

    The hybrid pattern is therefore `+-` for the first
    `first_k_dense_replace` blocks and `+E` thereafter. The HF→Megatron
    weight mapping uses the same index split.

Note:
    This bridge is *not* registered via `@MegatronModelBridge.register_bridge`
    because the auto-dispatch table in `model_bridge.py` is keyed by the HF
    source class name alone – registering a second bridge for
    `DeepseekV3ForCausalLM` would overwrite the default GPT-based bridge.
    Use this class explicitly, e.g.:

        bridge = DeepSeekV3MambaBridge()
        bridge.hf_config = hf_pretrained.config
        provider = bridge.provider_bridge(hf_pretrained)
        provider.finalize()
        model = provider.provide()
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.mamba import MambaModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.models.mla_provider import MLAModelProvider


@dataclass
class DeepSeekV3MambaProvider(MambaModelProvider, MLAModelProvider):
    """Provider that combines MLA config with the Mamba model backbone.

    Method-resolution order places `MambaModelProvider` first, so `provide()`
    returns an `MCoreMambaModel` (the hybrid stack that can host Mamba,
    attention, MLA, and DSA layers). `MLAModelProvider` contributes the MLA
    dataclass fields (`q_lora_rank`, `kv_lora_rank`, `qk_head_dim`,
    `qk_pos_emb_head_dim`, `v_head_dim`, ...) and – importantly – lets
    `issubclass(..., MLAModelProvider)` return True so the base bridge's
    MLA rope-scaling extraction path runs for DSv3 YaRN parameters.

    `finalize()` chains through both parents via C3 linearization: Mamba
    pattern processing runs first (deriving `num_layers` from the hybrid
    pattern), then MLA post-init computes the derived MLA head dims.
    """

    # MLA needs "rope" positional embeddings.
    position_embedding_type: str = "rope"


class DeepSeekV3MambaBridge(DeepSeekV3Bridge):
    """DeepSeek-V3 bridge that instantiates `MambaModel` instead of `GPTModel`.

    Inherits HF→Megatron config translation from `DeepSeekV3Bridge` (MLA
    params, MoE params, vocab sizing, MTP, YaRN rope scaling, ...) and swaps
    the target model class to `MambaModel` via `PROVIDER_CLASS`.

    `provider_bridge` picks up DSv3 defaults from the parent bridge, then:
    - Clears `transformer_layer_spec` (GPT-only).
    - Auto-populates `hybrid_layer_pattern` from `first_k_dense_replace`.
    - Resizes `moe_layer_freq` to match the doubled hybrid layer count.

    `mapping_registry` rewrites the DSv3 common mappings so attention
    params land on even Megatron decoder indices and MLP/MoE params land on
    odd ones, matching the `+-` / `+E` hybrid pattern. MTP mappings are
    carried over unchanged.
    """

    PROVIDER_CLASS = DeepSeekV3MambaProvider

    # Advertised target so downstream tooling that inspects the bridge knows
    # which MCore model class this variant produces.
    TARGET_MODEL = MambaModel

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DeepSeekV3MambaProvider:
        """Build a `DeepSeekV3MambaProvider` from an HF DSv3 checkpoint.

        Reuses the DSv3 MLA/MoE configuration produced by the parent
        `DeepSeekV3Bridge`, then adjusts for the Mamba backbone: drops
        GPT-only wiring, builds a hybrid layer pattern, and doubles
        `moe_layer_freq` to match the pattern length.
        """
        provider: DeepSeekV3MambaProvider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # GPT-specific decoder spec; MambaModel is built from `mamba_stack_spec`
        # instead. Clearing avoids confusion for downstream tooling that
        # introspects provider fields.
        if hasattr(provider, "transformer_layer_spec"):
            try:
                provider.transformer_layer_spec = None
            except (AttributeError, TypeError):
                pass

        # MLA requires "rope"; the Mamba default of "none" disables it.
        provider.position_embedding_type = "rope"

        # Build and install the hybrid layer pattern (`+-` dense / `+E` MoE).
        pattern = self._build_hybrid_layer_pattern(hf_config)
        provider.hybrid_layer_pattern = pattern
        # `finalize()` will derive num_layers from the pattern; clear any prior
        # value so the pattern-based derivation doesn't fail the equality check.
        provider.num_layers = None

        # moe_layer_freq must match the final num_layers (i.e., pattern length)
        # for MCore validation. Dense hybrid positions use `-` (ignored), MoE
        # positions use `E`; reflect this per-position freq.
        provider.moe_layer_freq = [
            1 if ch == Symbols.MOE else 0 for ch in pattern
        ]

        return provider

    @staticmethod
    def _build_hybrid_layer_pattern(hf_config) -> str:
        """Translate the DSv3 HF layout into a hybrid layer pattern.

        Each HF transformer block expands to `+-` (dense) or `+E` (MoE)
        in the hybrid stack.
        """
        num_layers = hf_config.num_hidden_layers
        first_k_dense = getattr(hf_config, "first_k_dense_replace", 0) or 0
        parts = []
        for layer_idx in range(num_layers):
            if layer_idx < first_k_dense:
                parts.append(Symbols.MLA + Symbols.MLP)
            else:
                parts.append(Symbols.MLA + Symbols.MOE)
        return "".join(parts)

    # Pattern templates categorized by which Megatron hybrid-position they land on.
    # `.*` is the (single) layer wildcard; trailing `*` (no dot) is the expert wildcard.
    _ATTENTION_TEMPLATES: List[Tuple[str, str]] = [
        ("decoder.layers.*.input_layernorm.weight", "model.layers.*.input_layernorm.weight"),
        ("decoder.layers.*.self_attention.linear_proj.weight", "model.layers.*.self_attn.o_proj.weight"),
        (
            "decoder.layers.*.self_attention.linear_kv_down_proj.weight",
            "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
        ),
        (
            "decoder.layers.*.self_attention.linear_kv_up_proj.weight",
            "model.layers.*.self_attn.kv_b_proj.weight",
        ),
        (
            "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight",
            "model.layers.*.self_attn.kv_a_layernorm.weight",
        ),
        (
            "decoder.layers.*.self_attention.kv_layernorm.weight",
            "model.layers.*.self_attn.kv_a_layernorm.weight",
        ),
        (
            "decoder.layers.*.self_attention.linear_q_down_proj.weight",
            "model.layers.*.self_attn.q_a_proj.weight",
        ),
        (
            "decoder.layers.*.self_attention.linear_q_up_proj.weight",
            "model.layers.*.self_attn.q_b_proj.weight",
        ),
        (
            "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight",
            "model.layers.*.self_attn.q_a_layernorm.weight",
        ),
        (
            "decoder.layers.*.self_attention.q_layernorm.weight",
            "model.layers.*.self_attn.q_a_layernorm.weight",
        ),
        # Fallback for DSv3 variants without LoRA on Q.
        (
            "decoder.layers.*.self_attention.linear_q_proj.weight",
            "model.layers.*.self_attn.q_proj.weight",
        ),
    ]

    _MLP_TEMPLATES: List[Tuple[str, str]] = [
        ("decoder.layers.*.pre_mlp_layernorm.weight", "model.layers.*.post_attention_layernorm.weight"),
        (
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight",
        ),
        ("decoder.layers.*.mlp.linear_fc2.weight", "model.layers.*.mlp.down_proj.weight"),
        ("decoder.layers.*.mlp.router.weight", "model.layers.*.mlp.gate.weight"),
        (
            "decoder.layers.*.mlp.router.expert_bias",
            "model.layers.*.mlp.gate.e_score_correction_bias",
        ),
        (
            "decoder.layers.*.mlp.experts.linear_fc2.weight*",
            "model.layers.*.mlp.experts.*.down_proj.weight",
        ),
        (
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
            "model.layers.*.mlp.shared_experts.down_proj.weight",
        ),
    ]

    # (megatron, gate_hf, up_hf) – always on the MLP/MoE hybrid position.
    _MLP_GATED_TEMPLATES: List[Tuple[str, str, str]] = [
        (
            "decoder.layers.*.mlp.linear_fc1.weight",
            "model.layers.*.mlp.gate_proj.weight",
            "model.layers.*.mlp.up_proj.weight",
        ),
        (
            "decoder.layers.*.mlp.experts.linear_fc1.weight*",
            "model.layers.*.mlp.experts.*.gate_proj.weight",
            "model.layers.*.mlp.experts.*.up_proj.weight",
        ),
        (
            "decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
            "model.layers.*.mlp.shared_experts.gate_proj.weight",
            "model.layers.*.mlp.shared_experts.up_proj.weight",
        ),
    ]

    # Top-level (non-layer) mappings – identical on both sides.
    _STATIC_MAPPINGS: List[Tuple[str, str]] = [
        ("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
        ("decoder.final_layernorm.weight", "model.norm.weight"),
        ("output_layer.weight", "lm_head.weight"),
    ]

    @staticmethod
    def _specialize(pattern: str, hf_idx: int, meg_idx: int) -> str:
        """Substitute the first `.*.` with the Megatron index and any later
        `.*.` with the HF index. Trailing `*` (no dot – expert wildcard)
        is preserved.
        """
        # First wildcard = layer index
        out = pattern.replace(".*.", f".{meg_idx}.", 1)
        # Any remaining `.*.` is the HF *layer* wildcard (only on HF side).
        # On Megatron side this won't match because only one layer wildcard
        # exists in our templates.
        return out

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Build mappings that split HF layer i into Megatron layers 2i and 2i+1.

        Attention-side templates are emitted at Megatron index `2*i`; MLP/MoE
        templates at `2*i + 1`. MTP mappings are reused from the DSv3 common
        list unchanged because MTP layers form a separate decoder namespace
        (`mtp.layers.*`) whose internal structure has no hybrid split.
        """
        hf_config = self.hf_config
        if hf_config is None:
            # Fall back to the GPT-layout mappings (still useful for tooling
            # that only inspects the registry without running conversion).
            return super().mapping_registry()

        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if num_layers is None:
            return super().mapping_registry()

        mapping_list = []

        # Top-level, non-layer params are untouched.
        for megatron_param, hf_param in self._STATIC_MAPPINGS:
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Per-layer unrolling.
        for hf_layer_idx in range(num_layers):
            attn_meg_idx = 2 * hf_layer_idx
            mlp_meg_idx = 2 * hf_layer_idx + 1

            for meg_tpl, hf_tpl in self._ATTENTION_TEMPLATES:
                mapping_list.append(
                    AutoMapping(
                        megatron_param=self._specialize(meg_tpl, hf_layer_idx, attn_meg_idx),
                        hf_param=hf_tpl.replace(".*.", f".{hf_layer_idx}.", 1),
                    )
                )

            for meg_tpl, hf_tpl in self._MLP_TEMPLATES:
                mapping_list.append(
                    AutoMapping(
                        megatron_param=self._specialize(meg_tpl, hf_layer_idx, mlp_meg_idx),
                        hf_param=hf_tpl.replace(".*.", f".{hf_layer_idx}.", 1),
                    )
                )

            for meg_tpl, gate_tpl, up_tpl in self._MLP_GATED_TEMPLATES:
                mapping_list.append(
                    GatedMLPMapping(
                        megatron_param=self._specialize(meg_tpl, hf_layer_idx, mlp_meg_idx),
                        gate=gate_tpl.replace(".*.", f".{hf_layer_idx}.", 1),
                        up=up_tpl.replace(".*.", f".{hf_layer_idx}.", 1),
                    )
                )

        # MTP mappings – delegate to the shared DSv3 helper. It emits MTP-only
        # entries (under `mtp.layers.*` / `model.layers.{N + k}.*`) that are
        # unaffected by the hybrid layer split.
        mtp_mappings = self._mtp_mappings_only(hf_config)
        mapping_list.extend(mtp_mappings)

        # DSv3-specific router bias mapping inherited from the default bridge.
        mapping_list.append(
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.router.expert_bias",
                hf_param="model.layers.*.mlp.gate.e_score_correction_bias",
            )
        )

        return MegatronMappingRegistry(*mapping_list)

    @staticmethod
    def _mtp_mappings_only(hf_config) -> list:
        """Extract only the MTP mappings from the DSv3 common mapping list.

        The shared `get_common_mapping_list` builds both transformer-layer
        mappings (for the GPT layout) and MTP mappings. We want only the MTP
        half for the hybrid layout – filter by the Megatron param prefix.
        """
        common = get_common_mapping_list(hf_config=hf_config)
        mtp_only = []
        for mapping in common:
            megatron_param = getattr(mapping, "megatron_param", "") or ""
            if isinstance(megatron_param, str) and megatron_param.startswith("mtp."):
                mtp_only.append(mapping)
        return mtp_only

    def maybe_modify_converted_hf_weight(
        self,
        task,
        converted_weights_dict: Dict[str, "torch.Tensor"],  # noqa: F821 (runtime torch)
        hf_state_dict,
    ):
        """Add rotary inv_freq based on Megatron input_layernorm → HF layer.

        Mirrors the parent bridge's logic but resolves the HF layer index from
        the doubled Megatron index: HF layer `i` corresponds to Megatron
        attention layer `2*i`.
        """
        import torch

        from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf

        global_name = task.global_param_name
        if not global_name.startswith("decoder.layers.") or not global_name.endswith(".input_layernorm.weight"):
            return converted_weights_dict

        parts = global_name.split(".")
        if len(parts) < 4 or not parts[2].isdigit():
            return converted_weights_dict

        megatron_layer_idx = int(parts[2])
        # Only even megatron layers correspond to HF attention layers in the
        # hybrid `+-` / `+E` layout. Odd layers are MLP/MoE and have no inv_freq.
        if megatron_layer_idx % 2 != 0:
            return converted_weights_dict
        hf_layer_idx = megatron_layer_idx // 2

        inv_freq_key = f"model.layers.{hf_layer_idx}.self_attn.rotary_emb.inv_freq"
        if inv_freq_key in converted_weights_dict:
            return converted_weights_dict

        has_inv_freq = getattr(self, "_deepseek_has_inv_freq", None)
        if has_inv_freq is None:
            has_inv_freq = False
            for key in hf_state_dict.keys():
                if key.startswith("model.layers.") and key.endswith(".self_attn.rotary_emb.inv_freq"):
                    has_inv_freq = True
                    break
            self._deepseek_has_inv_freq = has_inv_freq
        if not has_inv_freq:
            return converted_weights_dict

        inv_freq = getattr(self, "_deepseek_inv_freq", None)
        if inv_freq is None:
            rotary_dim = self.hf_config.qk_rope_head_dim
            rotary_base = rope_theta_from_hf(self.hf_config)
            inv_freq = 1.0 / (rotary_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
            self._deepseek_inv_freq = inv_freq

        if converted_weights_dict:
            reference_tensor = next(iter(converted_weights_dict.values()))
            if inv_freq.device != reference_tensor.device:
                inv_freq = inv_freq.to(device=reference_tensor.device)
                self._deepseek_inv_freq = inv_freq

        converted_weights_dict[inv_freq_key] = inv_freq
        return converted_weights_dict
