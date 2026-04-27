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
from typing import Dict

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.mamba import MambaModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, MegatronParamMapping
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

    `mapping_registry` reuses `get_common_mapping_list` and rewrites each
    `decoder.layers.*.*` template so attention params land on even Megatron
    decoder indices and MLP/MoE params land on odd ones, matching the
    `+-` / `+E` hybrid pattern. Everything else (embeddings, final
    layernorm, LM head, MTP) passes through unchanged.
    """

    PROVIDER_CLASS = DeepSeekV3MambaProvider

    # Advertised target so downstream tooling that inspects the bridge knows
    # which MCore model class this variant produces.
    TARGET_MODEL = MambaModel

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DeepSeekV3MambaProvider:
        """Build a `DeepSeekV3MambaProvider` from an HF DSv3 checkpoint.

        Reuses the DSv3 MLA/MoE configuration produced by the parent
        `DeepSeekV3Bridge`, then adjusts for the Mamba backbone: drops
        GPT-only wiring, builds a hybrid layer pattern, and resizes
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
        provider.moe_layer_freq = [1 if ch == Symbols.MOE else 0 for ch in pattern]

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

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Reuse the shared DSv3 mappings, splitting HF layer `i` into
        Megatron layers `2*i` (attention) and `2*i + 1` (MLP/MoE).

        Non-layer mappings (embeddings, final layernorm, LM head, and the
        `mtp.layers.*` MTP subtree) pass through unchanged.
        """
        hf_config = self.hf_config
        if hf_config is None:
            return super().mapping_registry()

        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if num_layers is None:
            return super().mapping_registry()

        # Shared DSv3 mappings (wildcard transformer templates + already-
        # concretized MTP entries) plus the DSv3-specific router expert bias
        # mapping that the parent bridge appends in `super().mapping_registry()`.
        templated = list(get_common_mapping_list(hf_config=hf_config))
        templated.append(
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.router.expert_bias",
                hf_param="model.layers.*.mlp.gate.e_score_correction_bias",
            )
        )

        remapped = []
        for mapping in templated:
            meg_param = getattr(mapping, "megatron_param", "") or ""
            if not (isinstance(meg_param, str) and meg_param.startswith("decoder.layers.")):
                # Non-layer or MTP mapping – pass through unchanged.
                remapped.append(mapping)
                continue

            is_attention = "self_attention" in meg_param or ".input_layernorm." in meg_param
            for hf_layer_idx in range(num_layers):
                meg_idx = (2 * hf_layer_idx) if is_attention else (2 * hf_layer_idx + 1)
                remapped.append(self._concretize_layer_mapping(mapping, hf_layer_idx, meg_idx))

        return MegatronMappingRegistry(*remapped)

    @staticmethod
    def _concretize_layer_mapping(
        mapping: MegatronParamMapping, hf_layer_idx: int, meg_idx: int
    ) -> MegatronParamMapping:
        """Replace the first `.*.` layer wildcard with concrete indices.

        Preserves any trailing `*` expert wildcard and uses `type(mapping)`
        so subclasses (AutoMapping, GatedMLPMapping, ...) round-trip cleanly.
        """
        meg_sub = f".{meg_idx}."
        hf_sub = f".{hf_layer_idx}."

        if isinstance(mapping, GatedMLPMapping):
            return GatedMLPMapping(
                megatron_param=mapping.megatron_param.replace(".*.", meg_sub, 1),
                gate=mapping.gate.replace(".*.", hf_sub, 1),
                up=mapping.up.replace(".*.", hf_sub, 1),
            )
        return type(mapping)(
            megatron_param=mapping.megatron_param.replace(".*.", meg_sub, 1),
            hf_param=mapping.hf_param.replace(".*.", hf_sub, 1),
        )

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
