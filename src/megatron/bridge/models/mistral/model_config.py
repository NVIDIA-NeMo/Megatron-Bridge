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

"""Builder-backed model configuration for Mistral models."""

from dataclasses import dataclass
from typing import ClassVar

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


@dataclass(kw_only=True)
class MistralModelConfig(BridgeGPTModelConfig):
    """Serializable Mistral GPT build configuration."""

    builder: ClassVar[str] = "megatron.training.models.gpt.GPTModelBuilder"

    yarn_rotary_scaling_factor: float | None = None
    yarn_original_max_position_embeddings: int | None = None
    yarn_beta_fast: float | None = None
    yarn_beta_slow: float | None = None
    yarn_mscale: float | None = None
    yarn_mscale_all_dim: float | None = None
    yarn_correction_range_round_to_int: bool | None = None


__all__ = ["MistralModelConfig"]
