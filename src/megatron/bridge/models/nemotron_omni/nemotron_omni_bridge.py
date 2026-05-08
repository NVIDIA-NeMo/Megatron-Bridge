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

"""Nemotron Omni bridge.

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

from megatron.core.activations import squared_relu

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
    NemotronOmniModelProvider,
)
from megatron.bridge.models.nemotron_vl.nemotron_vl_bridge import NemotronVLBridge


@MegatronModelBridge.register_bridge(
    source="NemotronH_Nano_Omni_Reasoning_V3",
    target=NemotronOmniModel,
    provider=NemotronOmniModelProvider,
    model_type="NemotronH_Nano_Omni_Reasoning_V3",
)
class NemotronOmniBridge(NemotronVLBridge):
    """Bridge for Nemotron-3 Omni (MoE LLM + vision + optional sound) models."""

    CONFIG_MAPPING = NemotronVLBridge.CONFIG_MAPPING + [
        # Mamba-specific (same as NemotronHBridge)
        ("mamba_head_dim", "mamba_head_dim"),
        ("mamba_num_heads", "mamba_num_heads"),
        ("n_groups", "mamba_num_groups"),
        ("ssm_state_size", "mamba_state_dim"),
        ("residual_in_fp32", "fp32_residual_connection"),
        # MoE-specific (only present in Omni configs)
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

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> NemotronOmniModelProvider:  # type: ignore[override]
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
        provider_kwargs["dynamic_resolution"] = True
        provider_kwargs["vision_class_token_len"] = 10

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

        return NemotronOmniModelProvider(**provider_kwargs)

    # ------------------------------------------------------------------
    # Parameter mapping
    # ------------------------------------------------------------------

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Inherit VL mappings and add temporal video, sound projection, and sound encoder mappings."""
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
