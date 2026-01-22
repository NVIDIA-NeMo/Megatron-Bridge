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

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch.nn.functional as F
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.mimo.mimo_provider import MimoModelProvider
from megatron.bridge.models.transformer_config import TransformerConfig


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class LlavaMimoProvider(MimoModelProvider):
    """LLaVA-style Vision-Language Model using MIMO architecture.

    This provider creates a multimodal model with:
    - Vicuna-7B style language model (Llama-based)
    - CLIP-style vision encoder
    - 2-layer MLP projector

    Example:
        >>> provider = LlavaMimoProvider()
        >>> model = provider.provide()
    """

    # Component configurations
    language_config: Optional[TransformerConfig] = None  # Defaults to Vicuna-7B
    vision_projection_config: Optional[TransformerConfig] = None  # Defaults to 2-layer MLP

    # Vision encoder (must be provided by user)
    vision_encoder_module: Optional[type] = None
    vision_encoder_params: Dict[str, any] = field(default_factory=dict)
    vision_projector_input_size: int = 1024  # CLIP ViT-L/14 output size

    # MIMO-specific
    image_special_token_id: int = 32000
    vocab_size: int = 32256  # Vicuna vocab size

    def __post_init__(self) -> None:
        """Configure MIMO specs after initialization."""
        # Create default configs if not provided
        if self.language_config is None:
            self.language_config = self._get_default_language_config()

        if self.vision_projection_config is None:
            self.vision_projection_config = self._get_default_vision_projection_config()

        # Create language model spec (Vicuna-7B style GPT)
        self.language_model_spec = ModuleSpec(
            module=GPTModel,
            params={
                "config": self.language_config,
                "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
                "vocab_size": self.vocab_size,
                "max_sequence_length": 4096,  # Vicuna default
                "position_embedding_type": "rope",
                "rotary_base": 10000,  # Vicuna default
                "rotary_percent": 1.0,
            },
        )

        # Create vision modality spec (CLIP encoder + MLP projector)
        self.modality_submodules_spec = {"images": self._get_vision_submodule_spec()}

        # Set special token IDs
        self.special_token_ids = {"images": self.image_special_token_id}

    def _get_default_language_config(self) -> TransformerConfig:
        """Create default Vicuna-7B language model config.

        Returns:
            TransformerConfig for the language model
        """
        config = TransformerConfig(
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
        return config

    def _get_default_vision_projection_config(self) -> TransformerConfig:
        """Create default vision projection config.

        Returns:
            TransformerConfig for the vision projector
        """
        return TransformerConfig(
            num_layers=2,  # 2-layer MLP
            hidden_size=4096,  # Match language model hidden size
            num_attention_heads=1,  # Not used in projector
        )

    def _get_vision_projection_spec(self) -> ModuleSpec:
        """Create vision projection spec (MLP).

        Returns:
            ModuleSpec for the vision→language projector
        """
        projection_layer_spec = ModuleSpec(module=None, submodules=MLPSubmodules(linear_fc1=None, linear_fc2=None))

        return ModuleSpec(
            module=MultimodalProjector,
            params={
                "config": self.vision_projection_config,
                "submodules": projection_layer_spec.submodules,
                "projector_type": "mlp",
                "input_size": self.vision_projector_input_size,
            },
        )

    def _get_vision_encoder_spec(self) -> ModuleSpec:
        """Create vision encoder spec.

        Returns:
            ModuleSpec for the vision encoder

        Raises:
            ValueError: If vision_encoder_module is not provided
        """
        if self.vision_encoder_module is None:
            raise ValueError(
                "vision_encoder_module must be provided. "
                "LlavaMimoProvider requires a vision encoder module. "
                "Example: provider = LlavaMimoProvider(vision_encoder_module=HFCLIPEncoder, ...)"
            )

        return ModuleSpec(module=self.vision_encoder_module, params=self.vision_encoder_params)

    def _get_vision_submodule_spec(self) -> ModuleSpec:
        """Create complete vision modality specification.

        Returns:
            ModuleSpec for the vision modality submodule
        """
        return ModuleSpec(
            module=VisionModalitySubmodules,
            params={},
            submodules={
                "encoders": {"clip_encoder": self._get_vision_encoder_spec()},
                "input_projections": [self._get_vision_projection_spec()],
            },
        )
