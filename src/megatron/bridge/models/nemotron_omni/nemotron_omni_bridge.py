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

"""Nemotron Omni conversion bridges.

Standalone bridge for the Nemotron-3 Omni family (HF architecture
``NemotronH_Nano_Omni_Reasoning_V3``). Inherits the language / vision /
mamba parameter mappings from :class:`NemotronVLBridge` and adds:

- Omni-specific ``CONFIG_MAPPING`` entries (Mamba shape fields used by the
  hybrid LLM and the MoE shared-expert intermediate size).
- An overridden :meth:`provider_bridge` that produces a
  :class:`NemotronOmniModelProvider` (MoE language model + RADIO ViT vision
  + optional Parakeet sound encoder) instead of the dense VL provider.
- A :meth:`mapping_registry` override that adds the temporal
  ``video_embedder`` parameter and the sound projection / sound encoder
  parameters (the latter via a single ``**`` wildcard, since the Megatron
  sound encoder is HF transformers' ``ParakeetEncoder`` and the parameter
  names line up 1:1 with ``sound_encoder.encoder.*``).
- ``ADDITIONAL_FILE_PATTERNS`` covering the bespoke Omni HF modeling /
  processing / audio files that need to be copied during HF export.
"""

import copy
from dataclasses import fields

from megatron.core.activations import squared_relu

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
    NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT,
    NEMOTRON_OMNI_LLAVA_CONTRACT,
    NemotronOmniLlavaModelProvider,
    NemotronOmniModelProvider,
)
from megatron.bridge.models.nemotron_vl.nemotron_vl_bridge import NemotronVLBridge
from megatron.bridge.models.nemotronh.nemotron_h_bridge import NemotronHBridge


def _copy_mapping_with_prefixes(mapping, *, megatron_prefix: str, hf_prefix: str):
    """Copy a mapping while preserving its conversion implementation."""

    copied = copy.copy(mapping)
    copied.megatron_param = megatron_prefix + mapping.megatron_param
    if isinstance(mapping.hf_param, str):
        copied.hf_param = hf_prefix + mapping.hf_param
    else:
        copied.hf_param = {key: hf_prefix + value for key, value in mapping.hf_param.items()}
    return copied


@MegatronModelBridge.register_bridge(
    source="NemotronH_Nano_Omni_Reasoning_V3",
    target=NemotronOmniModel,
    provider=NemotronOmniModelProvider,
    model_type="NemotronH_Nano_Omni_Reasoning_V3",
)
@MegatronModelBridge.register_bridge(
    source="NemotronH_Super_Omni_Reasoning_V3",
    target=NemotronOmniModel,
    provider=NemotronOmniModelProvider,
    model_type="NemotronH_Super_Omni_Reasoning_V3",
)
class NemotronOmniBridge(NemotronVLBridge):
    """Bridge for the canonical expanded-sequence Nemotron-3 Omni model."""

    CONFIG_MAPPING = NemotronVLBridge.CONFIG_MAPPING + [
        # HF public Omni config uses layer_norm_epsilon instead of rms_norm_eps.
        ("layer_norm_epsilon", "layernorm_epsilon"),
        # Mamba-specific (same as NemotronHBridge)
        ("mamba_head_dim", "mamba_head_dim"),
        ("mamba_num_heads", "mamba_num_heads"),
        ("n_groups", "mamba_num_groups"),
        ("ssm_state_size", "mamba_state_dim"),
        ("residual_in_fp32", "fp32_residual_connection"),
        # MoE-specific (only present in Omni configs)
        ("moe_latent_size", "moe_latent_size"),
        ("moe_shared_expert_intermediate_size", "moe_shared_expert_intermediate_size"),
    ]

    # Custom modeling/processing/audio files to copy during HF export.
    ADDITIONAL_FILE_PATTERNS = [
        "modeling*.py",
        "configuration*.py",
        "processing*.py",
        "processing_utils.py",
        "image_processing*.py",
        "video_processing*.py",
        "video_io.py",
        "audio_model.py",
        "evs.py",
    ]

    # ------------------------------------------------------------------
    # Provider translation
    # ------------------------------------------------------------------

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> NemotronOmniModelProvider:  # type: ignore[override]
        """Create a NemotronOmniModelProvider from the HF Omni config.

        Always returns an Omni provider (MoE language model + RADIO ViT
        vision + optional Parakeet sound encoder). When ``sound_config`` is
        absent on the HF config, ``has_sound=False`` and the sound branch
        is skipped at construction time.
        """
        hf_config = hf_pretrained.config
        llm_config = hf_config.llm_config

        provider_kwargs = self.hf_config_to_provider_kwargs(llm_config)

        provider_kwargs["num_layers"] = None
        provider_kwargs["make_vocab_size_divisible_by"] = self.make_vocab_size_divisible_by(llm_config.vocab_size)

        if hasattr(hf_config, "projector_hidden_size"):
            provider_kwargs["vision_proj_ffn_hidden_size"] = hf_config.projector_hidden_size

        has_sound = hasattr(hf_config, "sound_config") and hf_config.sound_config is not None
        if has_sound:
            sc = hf_config.sound_config
            provider_kwargs["has_sound"] = True
            provider_kwargs["sound_model_type"] = getattr(sc, "model_type", "parakeet")
            provider_kwargs["sound_hidden_size"] = sc.hidden_size
            provider_kwargs["sound_projection_hidden_size"] = sc.projection_hidden_size
            provider_kwargs["sound_context_token_id"] = hf_config.sound_context_token_id
            provider_kwargs["sound_config"] = sc.to_dict() if hasattr(sc, "to_dict") else dict(sc)

        provider_kwargs["language_model_type"] = "nemotron6-moe"
        provider_kwargs["image_token_index"] = getattr(hf_config, "img_context_token_id", 18)
        provider_kwargs["img_start_token_id"] = 21
        provider_kwargs["img_end_token_id"] = 22
        provider_kwargs["tokenizer_type"] = "nemotron6-moe"
        provider_kwargs["use_vision_backbone_fp8_arch"] = False
        provider_kwargs["vision_class_token_len"] = 10
        # Match C-RADIO's eval-time position embedding behavior: interpolate
        # to a square grid covering the longest image dimension, then crop to
        # the requested aspect ratio. Keep the provider's historical default
        # unchanged for serialized configurations that explicitly select it.
        provider_kwargs["radio_interpolate_only_cpe"] = False
        provider_kwargs["nemotron_omni_contract"] = NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT

        # NemotronH uses squared_relu for MLP layers (HF config: mlp_hidden_act="relu2").
        # The base hf_config_to_provider_kwargs reads "hidden_act" which doesn't exist on
        # this config, causing it to fall back to silu. Override explicitly.
        provider_kwargs["activation_func"] = squared_relu

        # Temporal video embedder: pull settings from HF vision_config when the
        # checkpoint was trained with a separate video patch embedder.
        vision_cfg = getattr(hf_config, "vision_config", None)
        if vision_cfg is not None and getattr(vision_cfg, "separate_video_embedder", False):
            provider_kwargs["separate_video_embedder"] = True
            provider_kwargs["temporal_patch_dim"] = getattr(vision_cfg, "video_temporal_patch_size", 2)
            provider_kwargs["temporal_ckpt_compat"] = True

        provider = NemotronOmniModelProvider(**provider_kwargs)
        provider.mtp_hybrid_override_pattern = getattr(llm_config, "mtp_hybrid_override_pattern", None)
        return provider

    # ------------------------------------------------------------------
    # Parameter mapping
    # ------------------------------------------------------------------

    def _llava_mapping_registry(self) -> MegatronMappingRegistry:
        """Build mappings for the historical LLaVA wrapper namespace."""
        vl_registry = super().mapping_registry()
        mapping_list = list(vl_registry.mappings)

        # MoE language decoder (not present in the dense VL variant).
        for megatron_param, hf_param in {
            "llava_model.language_model.decoder.layers.*.mlp.router.weight": "language_model.backbone.layers.*.mixer.gate.weight",
            "llava_model.language_model.decoder.layers.*.mlp.router.expert_bias": "language_model.backbone.layers.*.mixer.gate.e_score_correction_bias",
            "llava_model.language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*": "language_model.backbone.layers.*.mixer.experts.*.up_proj.weight",
            "llava_model.language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*": "language_model.backbone.layers.*.mixer.experts.*.down_proj.weight",
            "llava_model.language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight": "language_model.backbone.layers.*.mixer.shared_experts.up_proj.weight",
            "llava_model.language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "language_model.backbone.layers.*.mixer.shared_experts.down_proj.weight",
        }.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Temporal video embedder (only present in Omni checkpoints trained
        # with a separate video patch embedder).
        mapping_list.append(
            AutoMapping(
                megatron_param="llava_model.vision_model.video_embedder.weight",
                hf_param="vision_model.radio_model.model.patch_generator.video_embedder.weight",
            )
        )

        # Sound projection (same MultimodalProjector structure as vision projection).
        for megatron_param, hf_param in {
            "llava_model.sound_projection.encoder.linear_fc1.layer_norm_weight": "sound_projection.norm.weight",
            "llava_model.sound_projection.encoder.linear_fc1.weight": "sound_projection.linear1.weight",
            "llava_model.sound_projection.encoder.linear_fc2.weight": "sound_projection.linear2.weight",
        }.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Sound encoder: the Megatron sound encoder is HF transformers'
        # ``ParakeetEncoder``, so its parameter names line up 1:1 with the
        # ``sound_encoder.encoder.*`` keys in the Nemotron-Omni HF
        # checkpoint. A single wildcard mapping handles the whole subtree
        # (conformer layers, subsampling convs, subsampling linear).
        # Feature extractor buffers (``feature_extractor.featurizer.fb``,
        # ``.window``) live outside the encoder and are intentionally
        # unmapped -- they're skipped on import and regenerated from config
        # on export.
        mapping_list.append(
            ReplicatedMapping(
                megatron_param="llava_model.sound_model.encoder.**",
                hf_param="sound_encoder.encoder.**",
            )
        )

        return MegatronMappingRegistry(*mapping_list)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return top-level media mappings plus prefixed NemotronH mappings."""

        mappings = []

        # Reuse the media mappings, removing LLaVA's wrapper namespace.
        # Language mappings come from NemotronHBridge so MTP and supported MoE
        # layouts remain one source of truth.
        for mapping in self._llava_mapping_registry().mappings:
            if mapping.megatron_param.startswith("llava_model.language_model."):
                continue
            copied = copy.copy(mapping)
            copied.megatron_param = mapping.megatron_param.removeprefix("llava_model.")
            mappings.append(copied)

        hf_config = getattr(self, "hf_config", None)
        llm_config = getattr(hf_config, "llm_config", None)
        mtp_pattern = getattr(llm_config, "mtp_hybrid_override_pattern", None)

        language_bridge = NemotronHBridge()
        language_bridge._mtp_layers_per_block = len(mtp_pattern) if mtp_pattern else 0
        for mapping in language_bridge.mapping_registry().mappings:
            is_mtp = mapping.megatron_param.startswith("mtp.")
            mappings.append(
                _copy_mapping_with_prefixes(
                    mapping,
                    megatron_prefix="language_model.",
                    # The public Omni checkpoint keeps MTP at the top level,
                    # while the rest of NemotronH lives under language_model.
                    hf_prefix="" if is_mtp else "language_model.",
                )
            )

        return MegatronMappingRegistry(*mappings)


class NemotronOmniLlavaBridge(NemotronOmniBridge):
    """Explicit fallback bridge for the historical collapse/expand model."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> NemotronOmniLlavaModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        provider_kwargs = {
            field.name: getattr(provider, field.name)
            for field in fields(NemotronOmniLlavaModelProvider)
            if field.init and hasattr(provider, field.name)
        }
        provider_kwargs["nemotron_omni_contract"] = NEMOTRON_OMNI_LLAVA_CONTRACT
        return NemotronOmniLlavaModelProvider(**provider_kwargs)

    def mapping_registry(self) -> MegatronMappingRegistry:
        return self._llava_mapping_registry()
