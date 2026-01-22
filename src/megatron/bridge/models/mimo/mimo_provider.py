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

from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.model_provider import ModelProviderMixin


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MimoModelProvider(ModelProviderMixin[MimoModel]):
    """Base provider for MIMO multimodal models.

    This provider creates MIMO models by composing:
    - A language model specification
    - Modality submodule specifications (vision, audio, etc.)
    - Special token IDs for each modality

    Subclasses should configure:
    - language_model_spec: Specification for the language model
    - modality_submodules_spec: Dict mapping modality names to their specs
    - special_token_ids: Dict mapping modality names to special token IDs

    Example:
        >>> provider = MimoModelProvider(
        ...     language_model_spec=gpt_spec,
        ...     modality_submodules_spec={"images": vision_spec},
        ...     special_token_ids={"images": 32000},
        ... )
        >>> model = provider.provide()
    """

    # Language model configuration
    language_model_spec: Optional[ModuleSpec] = None

    # Modality configurations
    modality_submodules_spec: Dict[str, ModuleSpec] = field(default_factory=dict)
    special_token_ids: Dict[str, int] = field(default_factory=dict)

    # Optional: Freeze options
    freeze_language_model: bool = False
    freeze_modality_encoders: Dict[str, bool] = field(default_factory=dict)
    freeze_modality_projections: Dict[str, bool] = field(default_factory=dict)

    def provide(
        self, pre_process: Optional[bool] = None, post_process: Optional[bool] = None, vp_stage: Optional[int] = None
    ) -> MimoModel:
        """Provide a configured MIMO model instance.

        Args:
            pre_process: Whether to include pre-processing (embedding layer)
            post_process: Whether to include post-processing (output layer)
            vp_stage: Virtual pipeline stage

        Returns:
            MimoModel: Configured MIMO model instance
        """
        if self.language_model_spec is None:
            raise ValueError("language_model_spec must be configured before calling provide()")

        if not self.modality_submodules_spec:
            logger.warning("No modality submodules configured. Creating MIMO model with language model only.")

        # Create MIMO model config
        mimo_config = MimoModelConfig(
            language_model_spec=self.language_model_spec,
            modality_submodules_spec=self.modality_submodules_spec,
            special_token_ids=self.special_token_ids,
        )

        # Instantiate MIMO model
        model = MimoModel(mimo_config)

        # Apply freezing if configured
        if self.freeze_language_model:
            logger.info("Freezing language model parameters")
            for param in model.language_model.parameters():
                param.requires_grad = False

        # Freeze modality encoders if specified
        for modality, should_freeze in self.freeze_modality_encoders.items():
            if should_freeze and modality in model.modality_submodules:
                logger.info(f"Freezing {modality} encoder parameters")
                for param in model.modality_submodules[modality].encoders.parameters():
                    param.requires_grad = False

        # Freeze modality projections if specified
        for modality, should_freeze in self.freeze_modality_projections.items():
            if should_freeze and modality in model.modality_submodules:
                logger.info(f"Freezing {modality} projection parameters")
                for param in model.modality_submodules[modality].input_projections.parameters():
                    param.requires_grad = False

        return model
