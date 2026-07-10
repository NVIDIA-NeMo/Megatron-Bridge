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

"""Provider-neutral EXAONE 4 model configuration."""

from dataclasses import dataclass, field
from typing import Callable, ClassVar

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.exaone.layer_specs import exaone4_layer_spec
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


@dataclass(kw_only=True)
class Exaone4ModelConfig(BridgeGPTModelConfig):
    """GPT build config preserving EXAONE's post-LN and Llama 3 RoPE settings."""

    builder: ClassVar[str] = "megatron.bridge.models.exaone.model_config.Exaone4ModelBuilder"
    transformer_layer_spec: Callable[..., ModuleSpec] = field(default_factory=lambda: exaone4_layer_spec)
    rope_scaling_low_freq_factor: float = 1.0
    rope_scaling_high_freq_factor: float = 4.0
    rope_scaling_original_max_position_embeddings: int = 8192


class Exaone4ModelBuilder(GPTModelBuilder):
    """GPT builder applying all configurable Llama 3 frequency bands."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build EXAONE and replace the default fixed-band Llama 3 frequencies."""
        model = super().build_model(
            pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        if not self._model_config.rope_scaling or not hasattr(model, "rotary_pos_emb"):
            return model

        rotary = model.rotary_pos_emb
        dim = self._model_config.kv_channels
        if self._model_config.rotary_percent < 1.0:
            dim = int(dim * self._model_config.rotary_percent)
        unscaled_inv_freq = 1.0 / (
            self._model_config.rotary_base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=rotary.inv_freq.device) / dim)
        )
        rotary.inv_freq = rotary._apply_scaling(
            unscaled_inv_freq,
            factor=self._model_config.rope_scaling_factor,
            low_freq_factor=self._model_config.rope_scaling_low_freq_factor,
            high_freq_factor=self._model_config.rope_scaling_high_freq_factor,
            original_max_position_embeddings=self._model_config.rope_scaling_original_max_position_embeddings,
        )
        return model


__all__ = ["Exaone4ModelBuilder", "Exaone4ModelConfig"]
