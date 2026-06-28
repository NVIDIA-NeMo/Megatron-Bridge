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

"""Pure model configuration and standalone builder for FLUX."""

from dataclasses import dataclass, field
from typing import ClassVar

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.diffusion.models.flux.flux_model import Flux
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


@dataclass(kw_only=True)
class FluxModelConfig(BridgeGPTModelConfig):
    """Serializable FLUX architecture and exact MCore transformer settings."""

    builder: ClassVar[str] = "megatron.bridge.diffusion.models.flux.model_config.FluxModelBuilder"
    num_joint_layers: int = 19
    num_single_layers: int = 38
    in_channels: int = 64
    context_dim: int = 4096
    model_channels: int = 256
    axes_dims_rope: list[int] = field(default_factory=lambda: [16, 56, 56])
    patch_size: int = 1
    guidance_embed: bool = False
    vec_in_dim: int = 768
    guidance_scale: float = 3.5


class FluxModelBuilder(GPTModelBuilder):
    """Build FLUX stages without a model provider."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Flux:
        """Build one FLUX model stage."""
        return Flux(
            config=self._model_config.transformer,
            architecture_config=self._model_config,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
        )


__all__ = ["FluxModelBuilder", "FluxModelConfig"]
