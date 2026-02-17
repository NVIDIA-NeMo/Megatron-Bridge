# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""LLaVA-style Vision-Language Model provider."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch.nn.functional as F
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.mimo.mimo_provider import MimoModelProvider
from megatron.bridge.models.transformer_config import TransformerConfig


@dataclass
class LlavaMimoProvider(MimoModelProvider):
    """LLaVA-style Vision-Language Model provider.

    Preconfigures specs for:
    - Vicuna-7B style language model (Llama-based)
    - CLIP-style vision encoder
    - 2-layer MLP projector

    Example:
        >>> from my_encoders import HFCLIPEncoder
        >>> provider = LlavaMimoProvider(
        ...     vision_encoder_module=HFCLIPEncoder,
        ...     mimo_parallelism_config=mimo_parallelism_config,
        ... )
        >>> result = provider.provide()
    """

    # Vision encoder (user must provide)
    vision_encoder_module: Optional[Type] = None
    vision_encoder_params: Dict = field(default_factory=dict)
    vision_projector_input_size: int = 1024  # CLIP ViT-L/14 output size

    # Override defaults
    image_special_token_id: int = 32000
    vocab_size: int = 32256  # Vicuna vocab size

    # Optional custom configs
    language_config: Optional[TransformerConfig] = None

    # Make parent's required field optional (we build it in __post_init__)
    language_model_spec: Optional[ModuleSpec] = None

    def __post_init__(self):
        """Build specs after initialization."""
        if self.vision_encoder_module is None:
            raise ValueError(
                "vision_encoder_module must be provided. "
                "Example: LlavaMimoProvider(vision_encoder_module=HFCLIPEncoder, ...)"
            )

        # Create default language config if not provided
        if self.language_config is None:
            self.language_config = self._get_default_language_config()

        # Build language model spec
        self.language_model_spec = ModuleSpec(
            module=GPTModel,
            params={
                "config": self.language_config,
                "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
                "vocab_size": self.vocab_size,
                "max_sequence_length": 4096,
                "pre_process": True,
                "post_process": True,
                "position_embedding_type": "rope",
            },
        )

        # Build vision modality spec
        self.modality_submodules_spec = {"images": self._build_vision_submodule_spec()}

        # Set special token IDs
        self.special_token_ids = {"images": self.image_special_token_id}

    def _get_default_language_config(self) -> TransformerConfig:
        """Create default Vicuna-7B language model config."""
        return TransformerConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=32,
            ffn_hidden_size=11008,
            normalization="RMSNorm",
            activation_func=F.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )

    def _build_vision_submodule_spec(self) -> ModuleSpec:
        """Build vision modality specification."""
        # Vision encoder
        vision_encoder = ModuleSpec(
            module=self.vision_encoder_module,
            params=self.vision_encoder_params,
        )

        # Vision projector (2-layer MLP)
        projection_config = TransformerConfig(
            num_layers=2,
            hidden_size=self.language_config.hidden_size,
            num_attention_heads=1,
        )

        projection_layer_spec = ModuleSpec(
            module=None,
            submodules=MLPSubmodules(linear_fc1=None, linear_fc2=None),
        )

        vision_projection = ModuleSpec(
            module=MultimodalProjector,
            params={
                "config": projection_config,
                "submodules": projection_layer_spec.submodules,
                "projector_type": "mlp",
                "input_size": self.vision_projector_input_size,
            },
        )

        return ModuleSpec(
            module=VisionModalitySubmodules,
            params={},
            submodules={
                "encoders": {"clip_encoder": vision_encoder},
                "input_projections": [vision_projection],
            },
        )
