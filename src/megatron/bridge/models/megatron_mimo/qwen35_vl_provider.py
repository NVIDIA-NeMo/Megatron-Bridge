# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Qwen3.5-VL MegatronMIMO provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider


_QWEN_VISUAL_ENCODER_KEY = "qwen_visual"
_IMAGES_MODALITY_KEY = "images"


@dataclass
class Qwen35VLMegatronMIMOProvider(MegatronMIMOProvider):
    """MegatronMIMO provider for Qwen3.5-VL dense models.

    Wraps a ``Qwen35VLModelProvider`` to produce a ``MimoModel`` with the
    language module under ``language_model`` and the Qwen3VL vision encoder
    under the ``"images"`` modality (encoder name ``"qwen_visual"``).
    """

    # Underlying Qwen3.5-VL provider — carries all language + vision config.
    # Optional only so dataclass field ordering works; ``__post_init__`` rejects None.
    language_provider: Optional[Qwen35VLModelProvider] = None

    def __post_init__(self) -> None:
        """Validate inputs and build the MIMO modality spec eagerly.

        ``language_model_spec`` is deferred to ``provide()`` because the
        hybrid block-spec helper queries ``parallel_state`` and needs
        ``torch.distributed`` initialised.
        """
        if self.language_provider is None:
            raise ValueError(
                "language_provider must be provided. "
                "Example: Qwen35VLMegatronMIMOProvider(language_provider=Qwen35VLModelProvider(...))"
            )

        # MIMO v1 does not support MTP — reject and canonicalise to None below.
        if self.language_provider.mtp_num_layers:
            raise ValueError(
                "Qwen3.5-VL MegatronMIMO does not support MTP layers (v1). "
                f"Got mtp_num_layers={self.language_provider.mtp_num_layers}. "
                "Set language_provider.mtp_num_layers=None."
            )
        self.language_provider.mtp_num_layers = None

        self.modality_submodules_spec = {
            _IMAGES_MODALITY_KEY: self._build_vision_modality_spec(),
        }
        self.special_token_ids = {_IMAGES_MODALITY_KEY: self.language_provider.image_token_id}

    def provide(self, *args, **kwargs) -> MimoModel:
        """Build the MimoModel, lazily filling in ``language_model_spec``."""
        if self.language_model_spec is None:
            self.language_model_spec = self._build_language_model_spec()
        return super().provide(*args, **kwargs)

    def _build_language_model_spec(self) -> ModuleSpec:
        """ModuleSpec for ``Qwen3VLGPTModel``.

        ``pp_rank=0`` is hard-coded because MIMO v1 has PP=1 per component
        and the hybrid block helper otherwise reads global parallel_state.
        """
        lp = self.language_provider
        return ModuleSpec(
            module=Qwen3VLGPTModel,
            params={
                "config": lp,
                "transformer_layer_spec": lp.build_language_spec(pp_rank=0),
                "vocab_size": lp.vocab_size,
                "max_sequence_length": lp.language_max_sequence_length,
                "fp16_lm_cross_entropy": lp.fp16_lm_cross_entropy,
                "parallel_output": True,
                "share_embeddings_and_output_weights": lp.share_embeddings_and_output_weights,
                "position_embedding_type": "mrope",
                "rotary_percent": lp.rotary_percent,
                "rotary_base": lp.rotary_base,
                "scatter_embedding_sequence_parallel": False,
                "mtp_block_spec": None,
                # pre_process / post_process / pg_collection injected per rank
                # by MegatronMIMOProvider._inject_pg_collection_into_language_spec.
            },
        )

    def _build_vision_modality_spec(self) -> ModuleSpec:
        """ModuleSpec for the ``"images"`` modality container."""
        return ModuleSpec(
            module=VisionModalitySubmodules,
            params={},
            submodules={
                "encoders": {
                    _QWEN_VISUAL_ENCODER_KEY: self.language_provider.build_vision_encoder_spec(),
                },
                "input_projections": [],
            },
        )
