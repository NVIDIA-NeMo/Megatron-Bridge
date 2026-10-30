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
- A :meth:`hf_config_to_model_config_kwargs` mapping to the builder-backed
  :class:`NemotronOmniModelConfig` (MoE language model + RADIO ViT vision
  + optional Parakeet sound encoder). The deprecated provider path reuses it.
- A :meth:`mapping_registry` override that adds the temporal
  ``video_embedder`` parameter and the sound projection / sound encoder
  parameters (the latter via a single ``**`` wildcard, since the Megatron
  sound encoder is HF transformers' ``ParakeetEncoder`` and the parameter
  names line up 1:1 with ``sound_encoder.encoder.*``).
- ``ADDITIONAL_FILE_PATTERNS`` covering the bespoke Omni HF modeling /
  processing / audio files that need to be copied during HF export.
"""

import copy
import json
import logging
import warnings
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import transformers
from megatron.core.activations import squared_relu
from megatron.core.transformer.enums import AttnBackend
from packaging.version import Version
from safetensors.torch import save_file
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import (
    HFSourcedWeightTuple,
    HFWeightTuple,
    MegatronModelBridge,
    WeightConversionTask,
)
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict
from megatron.bridge.models.megatron_mimo.conversion import MIMOComponent, register_mimo_conversion_spec
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import (
    NemotronOmniMimoRadioEncoder,
    NemotronOmniModel,
)
from megatron.bridge.models.nemotron_omni.nemotron_omni_config import NemotronOmniModelConfig
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
    source="NemotronH_Nano_VL_V2",
    target=NemotronOmniModel,
    provider=NemotronOmniModelProvider,
    model_type="nemotron_vl",
)
class _LegacyNemotronOmniBridge(NemotronVLBridge):
    """Load MoE Nano Omni checkpoints carrying the historical V2 label."""

    MODEL_CONFIG_CLASS = NemotronOmniModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Map a MoE config with the V2 label like the canonical bridge; dense V2 configs are rejected."""
        return self._legacy_omni_bridge(hf_config).hf_config_to_model_config_kwargs(hf_config)


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

    MODEL_CONFIG_CLASS = NemotronOmniModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    _HF_DYNAMIC_MODULE_IMPORTS = """

# Transformers copies only direct relative imports into its local dynamic-module cache.
# Keep these transitive configuration dependencies visible from this auto_map entrypoint.
from .configuration_nemotron_h import NemotronHConfig as _NemotronHConfig
from .configuration_radio import RADIOConfig as _RADIOConfig
"""

    _HF_PASSTHROUGH_KEYS = (
        "sound_encoder.encoder.feature_extractor.featurizer.fb",
        "sound_encoder.encoder.feature_extractor.featurizer.window",
        "vision_model.radio_model.input_conditioner.norm_mean",
        "vision_model.radio_model.input_conditioner.norm_std",
    )
    _HF_VIDEO_EMBEDDER_KEY = "vision_model.radio_model.model.patch_generator.video_embedder.weight"

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
        ("moe_shared_expert_overlap", "moe_shared_expert_overlap"),
    ]

    # Custom modeling/processing/audio files and reasoning parsers for HF export.
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
        "*reasoning_parser.py",
    ]

    def postprocess_hf_export_artifacts(self, path: Path) -> None:
        """Make transitive Omni configuration modules discoverable on local reload.

        The pinned Nemotron Omni Hugging Face repositories expose ``modeling.py``
        as their ``auto_map`` entrypoint. Fail explicitly if that required export
        artifact is absent so an incomplete checkpoint is never reported as saved.
        """
        modeling_path = path / "modeling.py"
        if not modeling_path.is_file():
            raise FileNotFoundError(f"Nemotron Omni export is missing required artifact: {modeling_path}")

        modeling_source = modeling_path.read_text()
        if self._HF_DYNAMIC_MODULE_IMPORTS.strip() not in modeling_source:
            modeling_path.write_text(modeling_source.rstrip() + self._HF_DYNAMIC_MODULE_IMPORTS + "\n")

    # ------------------------------------------------------------------
    # Model config translation
    # ------------------------------------------------------------------

    @staticmethod
    def _hf_language_config(hf_config):
        return getattr(hf_config, "llm_config", None)

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face Omni config to builder config kwargs.

        The language fields are read from the checkpoint's language config. The
        presence of ``sound_config`` is the Hugging Face checkpoint's sound
        capability.
        """
        llm_config = self._hf_language_config(hf_config)
        config_kwargs = super().hf_config_to_model_config_kwargs(llm_config)

        hybrid_pattern = NemotronHBridge._hf_hybrid_pattern(llm_config)
        if hybrid_pattern is None:
            raise ValueError("Nemotron Omni requires hybrid_override_pattern or layers_block_type in llm_config.")
        config_kwargs.update(
            num_layers=None,
            hybrid_layer_pattern=hybrid_pattern,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(llm_config.vocab_size),
            # Nemotron-H attention layers use no position embedding.
            position_embedding_type="none",
            # NemotronH uses squared_relu for MLP layers (HF config: mlp_hidden_act="relu2").
            # The base mapping reads "hidden_act" which doesn't exist on this config, causing
            # it to fall back to silu. Override explicitly.
            activation_func=squared_relu,
            # The vision-language provider default, where the text mapping uses False.
            attention_softmax_in_fp32=True,
            image_token_index=getattr(hf_config, "img_context_token_id", 18),
            use_vision_backbone_fp8_arch=False,
            vision_class_token_len=10,
            # Match C-RADIO's eval-time position embedding behavior: interpolate
            # to a square grid covering the longest image dimension, then crop to
            # the requested aspect ratio.
            radio_interpolate_only_cpe=False,
            **NemotronHBridge._hf_mtp_settings(llm_config),
        )

        if hasattr(hf_config, "projector_hidden_size"):
            config_kwargs["vision_proj_ffn_hidden_size"] = hf_config.projector_hidden_size

        sc = getattr(hf_config, "sound_config", None)
        config_kwargs["has_sound"] = sc is not None
        if sc is not None:
            config_kwargs["sound_model_type"] = getattr(sc, "model_type", "parakeet")
            config_kwargs["sound_hidden_size"] = sc.hidden_size
            config_kwargs["sound_projection_hidden_size"] = sc.projection_hidden_size
            config_kwargs["sound_context_token_id"] = hf_config.sound_context_token_id
            config_kwargs["sound_config"] = sc.to_dict() if hasattr(sc, "to_dict") else dict(sc)

        # Temporal video embedder: pull settings from HF vision_config when the
        # checkpoint was trained with a separate video patch embedder.
        vision_cfg = getattr(hf_config, "vision_config", None)
        if vision_cfg is not None and getattr(vision_cfg, "separate_video_embedder", False):
            config_kwargs["separate_video_embedder"] = True
            config_kwargs["temporal_patch_dim"] = getattr(vision_cfg, "video_temporal_patch_size", 2)
            config_kwargs["temporal_ckpt_compat"] = True

        # NemotronOmniModelProvider defaults that differ from the builder config's, for
        # fields the Hugging Face config does not set.
        for name, default in (
            ("normalization", "RMSNorm"),
            ("add_bias_linear", False),
            ("hidden_dropout", 0.0),
            ("attention_dropout", 0.0),
            ("num_attention_heads", 1),
            ("num_query_groups", 8),
            ("masked_softmax_fusion", True),
            ("persist_layer_norm", True),
            ("first_last_layers_bf16", True),
            ("is_hybrid_model", True),
            ("apply_rope_fusion", True),
            ("attention_backend", AttnBackend.flash),
            ("bias_dropout_fusion", True),
            ("moe_aux_loss_coeff", 0.0001),
            ("moe_router_score_function", "sigmoid"),
            ("moe_router_enable_expert_bias", True),
            ("moe_router_load_balancing_type", "seq_aux_loss"),
            ("moe_router_dtype", "fp32"),
            ("moe_grouped_gemm", True),
            ("moe_token_dispatcher_type", "alltoall"),
            ("moe_permute_fusion", True),
            ("moe_shared_expert_overlap", True),
        ):
            config_kwargs.setdefault(name, default)
        return config_kwargs

    def _provider_only_kwargs(self, hf_config: PretrainedConfig, *, mtp_num_layers: int) -> dict[str, Any]:
        """Return the provider fields that the builder config does not declare.

        The canonical model reads none of them: the tokenizer and image boundary
        fields serve the LLaVA provider, the contract guards provider checkpoints,
        and nothing reads ``keep_mtp_spec_in_bf16``.
        """
        llm_config = self._hf_language_config(hf_config)
        return dict(
            language_model_type="nemotron6-moe",
            img_start_token_id=21,
            img_end_token_id=22,
            tokenizer_type="nemotron6-moe",
            nemotron_omni_contract=NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT,
            keep_mtp_spec_in_bf16=NemotronHBridge._keep_mtp_spec_in_bf16(llm_config, mtp_num_layers),
        )

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Adapt the canonical builder mapping to the deprecated provider path."""
        provider_kwargs = self.hf_config_to_model_config_kwargs(hf_config)
        provider_kwargs.update(self._provider_only_kwargs(hf_config, mtp_num_layers=provider_kwargs["mtp_num_layers"]))
        return provider_kwargs

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> NemotronOmniModelProvider:  # type: ignore[override]
        """Create a deprecated NemotronOmniModelProvider from the HF Omni config.

        This bridge maps its own configs, so it skips ``NemotronVLBridge``'s
        routing of the historical V2 label.
        """
        return MegatronModelBridge.provider_bridge(self, hf_pretrained)

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Export sound capability consistently with model construction."""
        hf_config = super().megatron_to_hf_config(provider)
        if provider.has_sound:
            hf_config["sound_config"] = provider.sound_config
            hf_config["sound_context_token_id"] = provider.sound_context_token_id
        else:
            # Config synthesis fills missing keys from the reference HF config.
            # Keep an explicit None so an image-text checkpoint stays sound-free.
            hf_config["sound_config"] = None
            hf_config["sound_context_token_id"] = None
        return hf_config

    # ------------------------------------------------------------------
    # Parameter mapping
    # ------------------------------------------------------------------

    def _mtp_hf_prefix(self) -> str:
        """Return the HF prefix applied to Nemotron-H MTP weights."""
        return ""

    def _llava_mapping_registry(self) -> MegatronMappingRegistry:
        """Build mappings for the historical LLaVA wrapper namespace."""
        # Call the explicit legacy implementation. NemotronVLBridge.mapping_registry
        # can route V2-labeled MoE checkpoints back to this canonical bridge.
        vl_registry = self._legacy_mapping_registry()
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
        # unmapped. They are preserved directly from the source checkpoint
        # during export.
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
        llm_config = self._hf_language_config(hf_config)

        language_bridge = NemotronHBridge()
        language_bridge.hf_config = llm_config
        for mapping in language_bridge.mapping_registry().mappings:
            is_mtp = mapping.megatron_param.startswith("mtp.")
            mappings.append(
                _copy_mapping_with_prefixes(
                    mapping,
                    megatron_prefix="language_model.",
                    # The public Omni checkpoint keeps MTP at the top level,
                    # while the rest of NemotronH lives under language_model.
                    hf_prefix=self._mtp_hf_prefix() if is_mtp else "language_model.",
                )
            )

        return MegatronMappingRegistry(*mappings)

    @staticmethod
    def _hf_requires_video_embedder(config) -> bool:
        vision_config = getattr(config, "vision_config", None)
        separate_video_embedder = (
            vision_config.get("separate_video_embedder", False)
            if isinstance(vision_config, dict)
            else getattr(vision_config, "separate_video_embedder", False)
        )
        return separate_video_embedder is True

    @torch.no_grad()
    def stream_weights_megatron_to_hf(
        self,
        megatron_model: NemotronOmniModel | list[NemotronOmniModel],
        hf_pretrained: PreTrainedCausalLM,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: list[WeightConversionTask] | None = None,
        merge_adapter_weights: bool = True,
        weight_dtype: torch.dtype | None = None,
        with_megatron_names: bool = False,
        current_pp_stage_only: bool = False,
    ) -> Iterable[HFWeightTuple | HFSourcedWeightTuple]:
        """Export model weights, recovering the unused image-SFT video weight from HF."""
        video_exported = False
        for weight in super().stream_weights_megatron_to_hf(
            megatron_model,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            merge_adapter_weights=merge_adapter_weights,
            weight_dtype=weight_dtype,
            with_megatron_names=with_megatron_names,
            current_pp_stage_only=current_pp_stage_only,
        ):
            video_exported |= weight.param_name == self._HF_VIDEO_EMBEDDER_KEY
            yield weight
        if not self._should_emit_hf_passthrough(megatron_model, current_pp_stage_only=current_pp_stage_only):
            return
        # A stage-local stream need not contain a video weight owned by another
        # pipeline stage. Never replace that trained parameter with the reference.
        video_owned_elsewhere = current_pp_stage_only and any(
            name.endswith("vision_model.video_embedder.weight") for name in getattr(self, "_cached_param_owners", {})
        )
        # MIMO calls this stream once per route. Only the vision route may
        # recover video weights: an earlier language-route fallback would win
        # deduplication over the trained weight emitted by the vision route.
        video_in_registry = any(
            mapping.hf_param == self._HF_VIDEO_EMBEDDER_KEY for mapping in self.mapping_registry().mappings
        )
        recover_video = video_in_registry and not video_exported and not video_owned_elsewhere
        config = getattr(hf_pretrained, "config", hf_pretrained)
        video_required = recover_video and self._hf_requires_video_embedder(config)
        missing_video_error = (
            f"Cannot export Nemotron Omni: {self._HF_VIDEO_EMBEDDER_KEY} is missing from the Megatron model "
            "and cannot be recovered from the reference HF checkpoint. Provide the original HF checkpoint "
            "and revision used for training; exporting without this weight would randomly initialize the "
            "video embedder on HF reload, even with strict=False."
        )
        # Passthrough tensors are copied straight from the HF checkpoint and have no
        # Megatron counterpart, so with ``with_megatron_names`` they carry zero sources.
        passthrough_sources = () if with_megatron_names else None

        state = getattr(hf_pretrained, "state", None)
        source = getattr(state, "source", None)
        if source is None:
            source_path = getattr(hf_pretrained, "name_or_path", None)
            if not source_path:
                if video_required:
                    raise ValueError(missing_video_error)
                return
            source = SafeTensorsStateSource(source_path, revision=getattr(config, "_commit_hash", None))
            state = StateDict(source)
        try:
            source_keys = set(source.get_all_keys())
        except (OSError, ValueError) as error:
            if video_required:
                raise ValueError(missing_video_error) from error
            raise
        passthrough_keys = list(self._HF_PASSTHROUGH_KEYS)
        if recover_video and self._HF_VIDEO_EMBEDDER_KEY in source_keys:
            passthrough_keys.append(self._HF_VIDEO_EMBEDDER_KEY)
            logging.getLogger(__name__).warning(
                "Recovering %s from the reference HF checkpoint because the Megatron export has no video embedder. "
                "Use the original HF checkpoint and revision used for training.",
                self._HF_VIDEO_EMBEDDER_KEY,
            )
        elif video_required:
            raise ValueError(missing_video_error)
        for name in passthrough_keys:
            if name not in source_keys:
                continue
            try:
                tensor = state[name]
            except (KeyError, OSError, ValueError) as error:
                if name == self._HF_VIDEO_EMBEDDER_KEY:
                    raise ValueError(missing_video_error) from error
                raise
            if name == self._HF_VIDEO_EMBEDDER_KEY and weight_dtype is not None:
                tensor = tensor.to(dtype=weight_dtype)
            # These come straight off disk, so they are on CPU while every
            # mapped tensor in the stream above is on the current device when
            # cpu=False. Consumers that batch the whole stream together (RL
            # refit packs it into one buffer) cannot mix devices.
            if not cpu and tensor.device.type == "cpu" and torch.cuda.is_available():
                tensor = tensor.to(device=torch.cuda.current_device())
            yield from HFWeightTuple(name, tensor).iter_finalized(cpu=cpu, megatron_param_names=passthrough_sources)


@MegatronModelBridge.register_bridge(
    source="NemotronH_Omni_Reasoning_V3",
    target=NemotronOmniModel,
    provider=NemotronOmniModelProvider,
    model_type="nemotron_h_omni",
)
class Nemotron35SuperVLBridge(NemotronOmniBridge):
    """Bridge for Nemotron 3.5 Super VL using the shared Omni media stack."""

    _HF_SUMMARY_IDXS_BUFFER = "vision_model.summary_idxs"
    # A previous export includes this derived buffer in its source index. Keep
    # it in the stream so strict re-export succeeds before postprocessing runs.
    _HF_PASSTHROUGH_KEYS = (*NemotronOmniBridge._HF_PASSTHROUGH_KEYS, _HF_SUMMARY_IDXS_BUFFER)
    _HF_SHARED_MTP_BLOCKS = 1
    _MCORE_MTP_PREDICTION_DEPTHS = 2

    @staticmethod
    def _hf_requires_video_embedder(config) -> bool:
        # Super VL uses the top-level temporal size, overriding nested RADIO flags.
        return int(getattr(config, "video_temporal_patch_size", 1) or 1) > 1

    @staticmethod
    def _hf_language_config(hf_config):
        # Remote-code checkpoints retain llm_config, including on older HF
        # versions. Do not force those customers onto the native implementation.
        legacy_config = getattr(hf_config, "llm_config", None)
        if legacy_config is not None:
            return legacy_config
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None and Version(transformers.__version__) < Version("5.18.0"):
            raise ValueError(
                "Native Nemotron 3.5 Super VL conversion requires Transformers >=5.18.0; "
                "use trust_remote_code=True with the original checkpoint on older versions."
            )
        return text_config

    @classmethod
    def _validate_shared_mtp_config(cls, llm_config) -> None:
        """Validate the serialized shared block before choosing training depths."""
        blocks, pattern = NemotronHBridge._hf_mtp_config(llm_config)
        if blocks != cls._HF_SHARED_MTP_BLOCKS:
            raise ValueError(f"Nemotron 3.5 Super VL requires exactly one serialized shared MTP block; got {blocks}.")
        if pattern != "*E" or not getattr(llm_config, "mtp_use_repeated_layer", True):
            raise ValueError("Nemotron 3.5 Super VL requires a repeated attention+MoE MTP block.")

    def text_only_pretrained(self, hf_pretrained: PreTrainedCausalLM) -> PreTrainedCausalLM:
        """Select the native Nemotron-H language checkpoint, excluding all media.

        Preserve the HF count of one serialized shared MTP block. Super text
        recipes explicitly set two training prediction depths, independently
        of this checkpoint representation.
        """
        config = copy.deepcopy(self._hf_language_config(hf_pretrained.config))
        self._validate_shared_mtp_config(config)
        config.architectures = ["NemotronHForCausalLM"]
        if hasattr(config, "auto_map"):
            del config.auto_map
        config.mtp_use_repeated_layer = True
        kwargs = dict(hf_pretrained.init_kwargs)
        if kwargs.get("subfolder"):
            raise ValueError(
                "text_only=True does not yet support HF subfolder checkpoints; use a local model directory."
            )
        revision = getattr(hf_pretrained.config, "_commit_hash", None) or kwargs.get("revision")
        if revision is not None:
            kwargs["revision"] = revision
        text = PreTrainedCausalLM(
            hf_pretrained.model_name_or_path,
            device=hf_pretrained.device,
            torch_dtype=hf_pretrained.torch_dtype,
            trust_remote_code=hf_pretrained.trust_remote_code,
            **kwargs,
        )
        text.config = config
        text._text_only = True
        # Reuse the native text bridge with a namespace-local checkpoint view,
        # as MIMO reuses component bridges with namespace-local registries.
        text._state_dict_accessor = StateDict(
            SafeTensorsStateSource(
                hf_pretrained.model_name_or_path,
                key_prefix="language_model.",
                revision=revision,
                hub_kwargs={
                    key: kwargs[key]
                    for key in ("token", "cache_dir", "local_files_only", "force_download")
                    if key in kwargs
                },
            )
        )
        text._processor = None
        text._image_processor = None
        text.custom_file_patterns = []
        return text

    def postprocess_hf_export_artifacts(self, path: Path) -> None:
        """Require remote code only for exports that actually reference it."""
        config_path = path / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            if "text_config" in config and not config.get("auto_map"):
                return
        modeling_path = path / "modeling_nemotron_h_omni.py"
        if not modeling_path.is_file():
            raise FileNotFoundError(f"Nemotron 3.5 Super VL export is missing required artifact: {modeling_path}")

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Map the shared Omni fields with Super-VL checkpoint features enabled."""
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        temporal_patch_dim = int(getattr(hf_config, "video_temporal_patch_size", 1) or 1)

        # Super-VL serializes one shared MTP block in HF. Megatron training applies
        # that block at two prediction depths, with the attention+MoE parameters
        # shared across both applications.
        self._validate_shared_mtp_config(self._hf_language_config(hf_config))
        config_kwargs["mtp_num_layers"] = self._MCORE_MTP_PREDICTION_DEPTHS

        if getattr(hf_config, "image_token_id", None) is not None:
            config_kwargs["image_token_index"] = hf_config.image_token_id

        config_kwargs["temporal_patch_dim"] = temporal_patch_dim
        config_kwargs["separate_video_embedder"] = temporal_patch_dim > 1
        # The Super-VL checkpoint carries a trained video embedder. Do not
        # synthesize one from image weights if that parameter is missing.
        config_kwargs["temporal_ckpt_compat"] = False
        config_kwargs["vision_final_layernorm"] = bool(config_kwargs["mtp_num_layers"])
        return config_kwargs

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Preserve HF's single-block serialization for the repeated MTP head."""
        if (
            provider.mtp_num_layers != cls._MCORE_MTP_PREDICTION_DEPTHS
            or provider.mtp_hybrid_override_pattern != "*E"
            or not provider.mtp_use_repeated_layer
        ):
            raise ValueError("Nemotron 3.5 Super VL export requires two repeated attention+MoE MTP depths.")

        hf_config = super().megatron_to_hf_config(provider)
        hf_config.pop("num_nextn_predict_layers", None)
        hf_config["llm_config"] = {"num_nextn_predict_layers": cls._HF_SHARED_MTP_BLOCKS}
        # Config synthesis retains only keys present in the reference schema.
        # Supply both spellings so native and remote-code exports preserve the
        # same shared-block count without requiring an HF dependency upgrade.
        hf_config["text_config"] = {"num_nextn_predict_layers": cls._HF_SHARED_MTP_BLOCKS}
        return hf_config

    def postprocess_hf_export_weights(self, path: Path) -> None:
        """Add the deterministic RADIO summary buffer omitted by the source index."""
        index_path = path / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Nemotron 3.5 Super VL export is missing its weight index: {index_path}")

        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Nemotron 3.5 Super VL export has an invalid weight map: {index_path}")
        if self._HF_SUMMARY_IDXS_BUFFER in weight_map:
            return

        config_path = path / "config.json"
        config = json.loads(config_path.read_text())
        summary_idxs = config.get("vision_config", {}).get("summary_idxs")
        if (
            not isinstance(summary_idxs, list)
            or not summary_idxs
            or not all(isinstance(value, int) for value in summary_idxs)
        ):
            raise ValueError(f"Nemotron 3.5 Super VL export has invalid vision summary indexes: {config_path}")

        summary_tensor = torch.tensor(summary_idxs, dtype=torch.long)
        shard_name = "model-summary-idxs.safetensors"
        shard_path = path / shard_name
        temporary_shard_path = path / f".{shard_name}.tmp"
        save_file({self._HF_SUMMARY_IDXS_BUFFER: summary_tensor}, temporary_shard_path)
        temporary_shard_path.replace(shard_path)

        weight_map[self._HF_SUMMARY_IDXS_BUFFER] = shard_name
        metadata = index.setdefault("metadata", {})
        metadata["total_size"] = (
            int(metadata.get("total_size", 0)) + summary_tensor.numel() * summary_tensor.element_size()
        )
        temporary_index_path = path / ".model.safetensors.index.json.tmp"
        temporary_index_path.write_text(json.dumps(index, indent=4) + "\n")
        temporary_index_path.replace(index_path)

    def _mtp_hf_prefix(self) -> str:
        """Nemotron 3.5 Super VL nests MTP below ``language_model``."""
        return "language_model."

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Add the Super-VL vision final norm to the shared Omni mappings."""
        mappings = list(super().mapping_registry().mappings)
        mappings.extend(
            [
                AutoMapping(
                    megatron_param="vision_model.decoder.final_layernorm.weight",
                    hf_param="vision_projector.vision_final_layernorm.weight",
                ),
                AutoMapping(
                    megatron_param="vision_model.decoder.final_layernorm.bias",
                    hf_param="vision_projector.vision_final_layernorm.bias",
                ),
            ]
        )
        return MegatronMappingRegistry(*mappings)


class NemotronOmniLlavaBridge(NemotronOmniBridge):
    """Deprecated fallback bridge for the historical collapse/expand model.

    Use :class:`NemotronOmniBridge`, which is the canonical AutoBridge
    registration and consumes processor-expanded media-token sequences.
    The LLaVA model still builds through its provider, so this bridge opts
    out of the builder-backed config that ``NemotronOmniBridge`` uses.
    """

    # Restore the base rejection instead of inheriting the Nemotron Omni builder mapping.
    MODEL_CONFIG_CLASS = None
    USE_MODEL_CONFIG_FOR_CONVERSION = False
    hf_config_to_model_config_kwargs = MegatronModelBridge.hf_config_to_model_config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Map the canonical Omni fields for the provider this bridge converts to LLaVA."""
        provider_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        provider_kwargs.update(self._provider_only_kwargs(hf_config, mtp_num_layers=provider_kwargs["mtp_num_layers"]))
        return provider_kwargs

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> NemotronOmniLlavaModelProvider:
        warnings.warn(
            "NemotronOmniLlavaBridge is deprecated; use NemotronOmniBridge with the canonical "
            "processor-expanded sequence contract.",
            FutureWarning,
            stacklevel=2,
        )
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


# RADIO's own parameters (class token, position embeddings, patch/video
# embedders) are replicated across TP ranks, exactly like ``RADIOViTModel``.
AutoMapping.register_module_type(NemotronOmniMimoRadioEncoder.__name__, "replicated")


@register_mimo_conversion_spec(NemotronOmniBridge)
@register_mimo_conversion_spec(Nemotron35SuperVLBridge)
def nemotron_omni_mimo_conversion_spec(
    source_bridge: NemotronOmniBridge,
    hf_pretrained: PreTrainedCausalLM,
    parallelism_config: MegatronMIMOParallelismConfig,
) -> tuple[MegatronMIMOProvider, list[MIMOComponent]]:
    """Reuse Omni weight mappings with separate language and image grids.

    The RADIO encoder and projector have separate routes sharing the image
    component's process groups.
    """
    standard_provider = source_bridge.provider_bridge(hf_pretrained)
    provider = MegatronMIMOProvider.from_standard_provider(standard_provider, parallelism_config)
    routes = [
        MIMOComponent("language", "language_model.", "language_model"),
        MIMOComponent("images", "vision_model.", "modality_submodules.images.encoders.radio"),
        MIMOComponent(
            "projector",
            "vision_projection.",
            "modality_submodules.images.input_projections.0",
            component_name="images",
        ),
    ]
    return provider, routes
