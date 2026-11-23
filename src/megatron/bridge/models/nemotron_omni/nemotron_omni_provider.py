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

import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Callable, ClassVar, Literal, Optional

from megatron.core import parallel_state
from megatron.core.activations import squared_relu
from megatron.core.models.hybrid.hybrid_model import HybridModel as MCoreHybridModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import (
    NemotronOmniMimoRadioEncoder,
    NemotronOmniModel,
)
from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni_llava import NemotronOmniLlavaModel
from megatron.bridge.models.nemotron_omni.nemotron_omni_builder import (
    _get_transformer_engine_projection_submodules,
    _make_omni_vision_config,
    _make_omni_vision_projection_config,
    _make_radio_projection_config,
    _make_radio_vision_config,
    _make_sound_encoder,
    _make_sound_projection_config,
)
from megatron.bridge.models.nemotron_omni.nemotron_omni_config import _validate_nemotron_omni_settings
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT = "expanded_sequence_v1"
NEMOTRON_OMNI_LLAVA_CONTRACT = "llava_collapse_expand_v1"

# MegatronMIMO component naming. The modality key doubles as the parallelism
# component name and the conversion route name; the encoder key is the
# ``modality_submodules.<modality>.encoders.<key>`` attribute.
_IMAGES_MODALITY_KEY = "images"
_RADIO_ENCODER_KEY = "radio"


@dataclass
class NemotronVLModelProvider(HybridModelProvider, ABC):
    """Abstract base provider for Nemotron VL model variants.

    Provides common VL fields, RADIO ViT-H vision config building methods, and
    vision projection config building methods shared by dense and MoE variants.
    Concrete subclasses set LLM-specific defaults (hidden_size, hybrid pattern,
    etc.) and may override ``provide()`` for variant-specific assembly.
    """

    # NemotronH base defaults
    mamba_num_groups: int = 8
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128
    activation_func: Callable = squared_relu
    masked_softmax_fusion: bool = True
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    first_last_layers_bf16: bool = True
    is_hybrid_model: bool = True

    # MoE defaults (shared across Nemotron VL variants)
    moe_aux_loss_coeff: float = 0.0001
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_router_dtype: str = "fp32"
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_shared_expert_overlap: bool = True

    # VL common overrides
    scatter_embedding_sequence_parallel: bool = False
    attention_softmax_in_fp32: bool = True

    vision_model_type: str = "radio"
    language_model_type: str = ""

    # Token IDs (overridden by concrete subclasses)
    image_token_index: int = 0
    img_start_token_id: int = 0
    img_end_token_id: int = 0
    tokenizer_type: str = ""

    # Vision backbone control
    dynamic_resolution: bool = False
    use_vision_backbone_fp8_arch: bool = True
    radio_force_eval_mode: bool = True
    radio_force_cpe_eval_mode: bool = True
    radio_interpolate_only_cpe: bool = True
    radio_cpe_aspect_ratio_select: bool = False
    radio_disable_cpe: bool = False
    recompute_vision: bool = False
    vision_recompute_granularity: Literal["full", "selective"] | None = None
    vision_recompute_modules: list[str] | None = None
    vision_recompute_method: Literal["uniform", "block"] | None = None
    vision_recompute_num_layers: int | None = None
    vision_proj_ffn_hidden_size: int = 20480
    vision_class_token_len: Optional[int] = None

    # Freeze control
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def _build_vision_config(self, language_cfg):
        """Build RADIO ViT-H vision encoder config from a language config copy."""
        return _make_radio_vision_config(self, language_cfg)

    def _build_vision_projection_config(self, language_cfg):
        """Build vision projection MLP config from a language config copy."""
        return _make_radio_projection_config(self, language_cfg)

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None):
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


@dataclass
class _NemotronOmniModelProviderBase(NemotronVLModelProvider):
    """Shared configuration and builders for Nemotron Omni models.

    Extends NemotronVLModelProvider with sound-specific fields and fixes RADIO
    to the public model's dynamic-resolution input contract.
    """

    # The public Nemotron-3 Omni processor emits variable-resolution images
    # which Bridge pre-patchifies for RADIO. The inherited static RADIO mode
    # accepts a different 4D tensor contract and is not an Omni configuration.
    dynamic_resolution: Literal[True] = True

    # This is the single source of truth for sound checkpoint capability:
    # disabling it omits both the encoder and its dependent projector.
    has_sound: bool = False
    sound_model_type: str = "parakeet"
    sound_hidden_size: int = 1024
    sound_projection_hidden_size: int = 4096
    sound_context_token_id: int = 0
    sound_config: Optional[dict] = None
    freeze_sound_encoder: bool = False
    freeze_sound_projection: bool = False

    # Temporal video embedder
    temporal_patch_dim: int = 1
    separate_video_embedder: bool = False
    temporal_ckpt_compat: bool = False  # formerly allow_checkpoint_without_temporal_compression
    vision_final_layernorm: bool = False

    # Shard images (or tubelets, when temporal compression is on) across the
    # context-parallel group instead of encoding every image on every CP rank.
    # The vision tower is replicated, so this is data parallelism borrowing the
    # CP group, not sequence sharding: RADIO already attends per image.
    # No-op at CP=1, so defaulting it on only changes CP>1 runs.
    vision_dp_over_cp: bool = True

    # This field is serialized in run_config.yaml. It prevents an older
    # checkpoint whose provider had the same class name but LLaVA semantics
    # from being loaded as the canonical expanded-sequence implementation.
    nemotron_omni_contract: Optional[str] = None

    def _validate_omni_config(self) -> None:
        _validate_nemotron_omni_settings(self)

    def finalize(self) -> None:
        """Finalize a dynamic-resolution Nemotron Omni provider."""
        self._validate_omni_config()
        super().finalize()

    def _require_model_contract(self, expected: str, implementation: str) -> None:
        actual = self.nemotron_omni_contract
        if actual == expected:
            return

        if actual is None:
            raise RuntimeError(
                f"Cannot construct {implementation}: the Nemotron Omni model contract is missing. "
                "Older branches used the name NemotronOmniModelProvider for the LLaVAModel "
                "collapse/expand implementation, while this branch uses that name for the "
                "expanded-sequence implementation. Refusing to guess which checkpoint layout "
                "was intended. Reconvert the checkpoint with this branch, or use the explicit "
                "NemotronOmniLlavaModelProvider/NemotronOmniLlavaBridge with a compatible legacy "
                "Megatron-LM implementation."
            )

        raise RuntimeError(
            f"Cannot construct {implementation}: run_config.yaml declares Nemotron Omni contract "
            f"{actual!r}, but this implementation requires {expected!r}. Use the provider/bridge "
            "that matches the checkpoint contract instead of reusing a provider class name across "
            "the LLaVA collapse/expand and expanded-sequence implementations."
        )

    def _build_vision_config(self, language_cfg):
        """Pin vision encoder to PP=1 (Omni training uses PP>1 on the LLM).

        The dense Nemotron-VL recipe runs with PP=1 everywhere, so the base
        VL provider doesn't pin this; for Omni we always co-locate the
        vision encoder with the first PP stage.
        """
        return _make_omni_vision_config(self, language_cfg)

    def _build_vision_projection_config(self, language_cfg):
        """Build the vision projection MLP config.

        Nemotron Omni applies squared ReLU in the vision projection MLP,
        matching the HF checkpoint and vLLM implementation. Also pin the
        projector to PP=1 (see :meth:`_build_vision_config`).
        """
        return _make_omni_vision_projection_config(self, language_cfg)

    def _build_sound_projection_config(self, language_cfg):
        """Build sound projection config (mirrors _build_vision_projection_config)."""
        return _make_sound_projection_config(self, language_cfg)

    def _build_sound_encoder(self):
        """Build BridgeSoundEncoder from sound_config dict."""
        return _make_sound_encoder(self)

    def _build_sound_modules(self, language_cfg, *, add_encoder: bool):
        """Build optional sound modules on the encoder pipeline stage."""
        if not (self.has_sound and add_encoder):
            return None, None

        sound_model = self._build_sound_encoder()
        sound_projection = MultimodalProjector(
            config=self._build_sound_projection_config(language_cfg),
            submodules=_get_transformer_engine_projection_submodules(),
            projector_type="mlp",
            input_size=self.sound_hidden_size,
        )
        return sound_model, sound_projection

    def _provide_llava(self, pre_process=None, post_process=None, vp_stage=None):
        """Assemble the legacy LLaVA collapse/expand implementation.

        Duplicates the VL provide() logic because LLaVAModel requires sound kwargs
        at construction time -- they can't be added after. This is intentional to
        maintain zero changes to nemotron_vl/.
        """
        self._validate_omni_config()
        language_cfg = self._copy_config_without_runtime_process_groups(deep=True)

        vision_cfg = self._build_vision_config(language_cfg)
        # Nano Omni checkpoints were trained with RADIO's ten class tokens.
        # This MCore LLaVAModel consumes the setting from the vision config;
        # its generic RADIO fallback otherwise assumes eight.
        vision_cfg.class_token_len = self.vision_class_token_len or 10
        vision_proj_cfg = self._build_vision_projection_config(language_cfg)

        # LLM decoder spec, which can be TE or Megatron inference-optimized.
        language_spec = self._resolve_hybrid_stack_spec()
        # ViT spec for the vision component of this model.
        vision_spec = get_vit_layer_with_transformer_engine_spec()
        # Vision projection spec that maps vision embeddings to language embeddings.
        vision_proj_spec = _get_transformer_engine_projection_submodules()

        add_encoder_flag = parallel_state.is_pipeline_first_stage() if self.pipeline_model_parallel_size > 1 else True
        add_decoder_flag = True

        sound_token_index = self.sound_context_token_id
        sound_model, sound_projection = self._build_sound_modules(
            language_cfg,
            add_encoder=add_encoder_flag,
        )

        llava_model = LLaVAModel(
            language_transformer_config=language_cfg,
            language_transformer_layer_spec=language_spec,
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=self.seq_length,
            **logit_dtype_kwarg(LLaVAModel, self.logit_dtype),
            vision_transformer_config=vision_cfg,
            vision_transformer_layer_spec=vision_spec,
            drop_vision_class_token=True,
            vision_projection_config=vision_proj_cfg,
            vision_projection_layer_spec=vision_proj_spec,
            vision_projection_type="mlp",
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            language_position_embedding_type=self.position_embedding_type,
            pre_process=pre_process if pre_process is not None else True,
            post_process=post_process if post_process is not None else True,
            add_encoder=add_encoder_flag,
            add_decoder=add_decoder_flag,
            img_h=512,
            img_w=512,
            patch_dim=16,
            hybrid_layer_pattern=self.hybrid_layer_pattern,
            image_token_index=self.image_token_index,
            pixel_shuffle=True,
            max_num_tiles=12,
            tokenizer_type=self.tokenizer_type,
            use_vision_backbone_fp8_arch=self.use_vision_backbone_fp8_arch,
            dynamic_resolution=self.dynamic_resolution,
            radio_force_eval_mode=self.radio_force_eval_mode,
            radio_force_cpe_eval_mode=self.radio_force_cpe_eval_mode,
            radio_interpolate_only_cpe=self.radio_interpolate_only_cpe,
            radio_cpe_aspect_ratio_select=self.radio_cpe_aspect_ratio_select,
            radio_disable_cpe=self.radio_disable_cpe,
            sound_model=sound_model,
            sound_projection=sound_projection,
            sound_token_index=sound_token_index,
            temporal_patch_dim=self.temporal_patch_dim,
            separate_video_embedder=self.separate_video_embedder,
            temporal_ckpt_compat=self.temporal_ckpt_compat,
        )

        self._configure_llava_preprocess_contract(llava_model)

        model = NemotronOmniLlavaModel(llava_model=llava_model)

        llava_model.img_start_token_id = self.img_start_token_id
        llava_model.img_end_token_id = self.img_end_token_id

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        if self.freeze_sound_encoder or self.freeze_sound_projection:
            model.freeze(
                freeze_sound_model=self.freeze_sound_encoder,
                freeze_sound_projection=self.freeze_sound_projection,
            )

        return model

    def _configure_llava_preprocess_contract(self, llava_model: LLaVAModel) -> None:
        """Align MCore preprocessing with the legacy collator's replacement counts."""
        if self.temporal_patch_dim == 1:
            # The legacy collator stores exact replacement-token counts in
            # num_image_tiles rather than physical tile counts. Keep MCore on
            # that contract even when imgs_sizes is present: dynamic-resolution
            # regrouping interprets num_image_tiles as physical tiles.
            llava_model._dynamic_resolution = False
            llava_model.img_seq_len = 1


@dataclass
class NemotronOmniModelProvider(_NemotronOmniModelProviderBase):
    """Provider for the canonical expanded-sequence Nemotron Omni model."""

    # The MIMO language spec wires the Nemotron-H MTP block, so MIMO
    # conversion keeps MTP weights instead of dropping them.
    mimo_supports_mtp: ClassVar[bool] = True

    # Match the RADIO position-embedding behavior used by the canonical
    # processor-expanded implementation without changing shared VL defaults.
    radio_interpolate_only_cpe: bool = False

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return the MIMO modality placeholder token ids."""
        return {_IMAGES_MODALITY_KEY: self.image_token_index}

    # ------------------------------------------------------------------
    # MegatronMIMO spec builders
    # ------------------------------------------------------------------

    def build_language_model_spec(self, pp_rank: Optional[int] = 0) -> ModuleSpec:
        """Build the MIMO language module spec (Nemotron-H hybrid decoder with MTP).

        Mirrors :meth:`HybridModelProvider.provide`; ``pre_process``,
        ``post_process`` and ``pg_collection`` are injected by the MIMO provider.
        """
        del pp_rank  # The hybrid stack spec is stage-independent.
        # Derive the unified hybrid+MTP pattern and layer counts before the spec
        # captures them; finalize() is idempotent.
        self.finalize()
        if self.should_pad_vocab:
            vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            vocab_size = self.vocab_size
        return ModuleSpec(
            module=MCoreHybridModel,
            params={
                "config": self,
                "hybrid_stack_spec": self._resolve_hybrid_stack_spec(),
                "vocab_size": vocab_size,
                "max_sequence_length": self.seq_length,
                "hybrid_layer_pattern": self.hybrid_layer_pattern,
                "fp16_lm_cross_entropy": self.fp16_lm_cross_entropy,
                **logit_dtype_kwarg(MCoreHybridModel, self.logit_dtype),
                "parallel_output": self.parallel_output,
                "share_embeddings_and_output_weights": self.share_embeddings_and_output_weights,
                "position_embedding_type": self.position_embedding_type,
                "rotary_percent": self.rotary_percent,
                "rotary_base": self.rotary_base,
                # MimoModel merges media into the embeddings before the decoder,
                # matching NemotronOmniModel's external-embedding contract.
                "scatter_embedding_sequence_parallel": False,
                "seq_len_interpolation_factor": self.seq_len_interpolation_factor,
            },
        )

    def _build_mimo_media_config(self, builder: Callable):
        """Derive an encoder/projector config copy that survives MIMO re-finalization.

        The standard model derives these copies from the finalized language config
        and never finalizes them again. The MIMO provider finalizes every config it
        finds in a spec, so the hybrid/MTP pattern fields that would re-derive the
        language layer count are cleared and expert parallelism is made dense.
        """
        self.finalize()
        language_cfg = self._copy_config_without_runtime_process_groups(deep=True)
        media_cfg = builder(language_cfg)
        media_cfg.hybrid_layer_pattern = None
        media_cfg.hybrid_override_pattern = None
        media_cfg.mtp_hybrid_override_pattern = None
        media_cfg.mtp_use_repeated_layer = False
        media_cfg.expert_model_parallel_size = 1
        media_cfg.expert_tensor_parallel_size = 1
        return media_cfg

    def build_vision_encoder_spec(self) -> ModuleSpec:
        """Build the RADIO encoder spec for the MIMO ``images`` modality."""
        vision_cfg = self._build_mimo_media_config(self._build_vision_config)
        return ModuleSpec(
            module=NemotronOmniMimoRadioEncoder,
            params={
                "transformer_config": vision_cfg,
                "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
                "img_h": 512,
                "img_w": 512,
                "max_img_h": 2048,
                "max_img_w": 2048,
                "class_token_len": self.vision_class_token_len or 10,
                "patch_dim": 16,
                "add_class_token": True,
                "embedder_bias": False,
                "dynamic_resolution": self.dynamic_resolution,
                "force_eval_mode": self.radio_force_eval_mode,
                "force_cpe_eval_mode": self.radio_force_cpe_eval_mode,
                "interpolate_only_cpe": self.radio_interpolate_only_cpe,
                "cpe_aspect_ratio_select": self.radio_cpe_aspect_ratio_select,
                "has_cpe": not self.radio_disable_cpe,
                "temporal_patch_dim": self.temporal_patch_dim,
                "separate_video_embedder": self.separate_video_embedder,
                "temporal_ckpt_compat": self.temporal_ckpt_compat,
            },
        )

    def build_vision_input_projection_spec(self) -> ModuleSpec:
        """Build the pixel-shuffled RADIO feature to language hidden projector spec."""
        vision_proj_cfg = self._build_mimo_media_config(self._build_vision_projection_config)
        vision_proj_cfg.mtp_num_layers = None
        vision_hidden_size = self._build_mimo_media_config(self._build_vision_config).hidden_size
        return ModuleSpec(
            module=MultimodalProjector,
            params={
                "config": vision_proj_cfg,
                "submodules": _get_transformer_engine_projection_submodules(),
                "projector_type": "mlp",
                "input_size": vision_hidden_size * 4,
            },
        )

    def build_mimo_modality_submodules_spec(self) -> dict[str, ModuleSpec]:
        """Build the MIMO modality specs: RADIO encoder plus vision projector."""
        if self.has_sound:
            raise NotImplementedError(
                "MegatronMIMO conversion of Nemotron Omni does not support sound-enabled checkpoints yet."
            )
        encoder_spec = self.build_vision_encoder_spec()

        return {
            _IMAGES_MODALITY_KEY: ModuleSpec(
                module=VisionModalitySubmodules,
                submodules={
                    "encoders": {_RADIO_ENCODER_KEY: encoder_spec},
                    "input_projections": [self.build_vision_input_projection_spec()],
                },
            )
        }

    def validate_model_contract(self) -> None:
        """Reject ambiguous or legacy serialized provider configurations."""
        self._require_model_contract(
            NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT,
            "NemotronOmniModel",
        )

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Build the top-level model from the shared Nano/Super fields."""

        self.validate_model_contract()

        language_cfg = self._copy_config_without_runtime_process_groups(deep=True)
        vision_cfg = self._build_vision_config(language_cfg)
        vision_proj_cfg = self._build_vision_projection_config(language_cfg)

        # LLM decoder spec, which can be TE or Megatron inference-optimized.
        language_spec = self._resolve_hybrid_stack_spec()
        # ViT spec for the vision component of this model.
        vision_spec = get_vit_layer_with_transformer_engine_spec()
        # Vision projection spec that maps vision embeddings to language embeddings.
        vision_proj_spec = _get_transformer_engine_projection_submodules()

        add_encoder = parallel_state.is_pipeline_first_stage() if self.pipeline_model_parallel_size > 1 else True

        sound_model, sound_projection = self._build_sound_modules(
            language_cfg,
            add_encoder=add_encoder,
        )

        model = NemotronOmniModel(
            language_transformer_config=language_cfg,
            language_transformer_layer_spec=language_spec,
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=self.seq_length,
            vision_transformer_config=vision_cfg,
            vision_transformer_layer_spec=vision_spec,
            vision_projection_config=vision_proj_cfg,
            vision_projection_layer_spec=vision_proj_spec,
            image_token_index=self.image_token_index,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            language_position_embedding_type=self.position_embedding_type,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            add_encoder=add_encoder,
            add_decoder=True,
            hybrid_layer_pattern=self.hybrid_layer_pattern,
            dynamic_resolution=self.dynamic_resolution,
            vision_class_token_len=self.vision_class_token_len or 10,
            radio_force_eval_mode=self.radio_force_eval_mode,
            radio_force_cpe_eval_mode=self.radio_force_cpe_eval_mode,
            radio_interpolate_only_cpe=self.radio_interpolate_only_cpe,
            radio_cpe_aspect_ratio_select=self.radio_cpe_aspect_ratio_select,
            radio_disable_cpe=self.radio_disable_cpe,
            temporal_patch_dim=self.temporal_patch_dim,
            separate_video_embedder=self.separate_video_embedder,
            temporal_ckpt_compat=self.temporal_ckpt_compat,
            vision_dp_over_cp=self.vision_dp_over_cp,
            sound_model=sound_model,
            sound_projection=sound_projection,
            sound_token_index=self.sound_context_token_id,
            vp_stage=vp_stage,
        )

        if any(
            (
                self.freeze_language_model,
                self.freeze_vision_model,
                self.freeze_vision_projection,
                self.freeze_sound_encoder,
                self.freeze_sound_projection,
            )
        ):
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
                freeze_sound_model=self.freeze_sound_encoder,
                freeze_sound_projection=self.freeze_sound_projection,
            )

        return model


@dataclass
class NemotronOmniLlavaModelProvider(NemotronOmniModelProvider):
    """Deprecated fallback provider for the historical collapse/expand path.

    Use :class:`NemotronOmniModelProvider`, which constructs the canonical
    processor-expanded model with collator-owned packing.
    """

    # Preserve the existing LLaVA provider default for compatibility.
    radio_interpolate_only_cpe: bool = True

    def validate_model_contract(self) -> None:
        """Require an explicitly selected legacy LLaVA contract."""
        self._require_model_contract(
            NEMOTRON_OMNI_LLAVA_CONTRACT,
            "NemotronOmniLlavaModel",
        )

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        warnings.warn(
            "NemotronOmniLlavaModelProvider is deprecated; use NemotronOmniModelProvider with the canonical "
            "processor-expanded sequence contract.",
            FutureWarning,
            stacklevel=2,
        )
        self.validate_model_contract()
        return self._provide_llava(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
