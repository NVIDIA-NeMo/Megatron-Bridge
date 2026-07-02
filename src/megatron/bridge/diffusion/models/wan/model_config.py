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

"""Pure model configuration and standalone builder for Wan diffusion."""

from dataclasses import dataclass
from typing import ClassVar

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.diffusion.models.wan.wan_model import WanModel
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


@dataclass(kw_only=True)
class WanModelConfig(BridgeGPTModelConfig):
    """Serializable Wan architecture and exact MCore transformer settings."""

    builder: ClassVar[str] = "megatron.bridge.diffusion.models.wan.model_config.WanModelBuilder"
    crossattn_emb_size: int = 1536
    in_channels: int = 16
    out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    freq_dim: int = 256
    text_len: int = 512
    text_dim: int = 4096
    layernorm_across_heads: bool = True
    qkv_format: str = "thd"


class WanModelBuilder(GPTModelBuilder):
    """Build Wan stages without a model provider."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> WanModel:
        """Build one Wan pipeline stage."""
        if pre_process is None:
            pre_process = parallel_state.is_pipeline_first_stage()
        if post_process is None:
            post_process = parallel_state.is_pipeline_last_stage()
        return WanModel(
            self._model_config.transformer,
            crossattn_emb_size=self._model_config.crossattn_emb_size,
            in_channels=self._model_config.in_channels,
            out_channels=self._model_config.out_channels,
            patch_spatial=self._model_config.patch_spatial,
            patch_temporal=self._model_config.patch_temporal,
            freq_dim=self._model_config.freq_dim,
            text_dim=self._model_config.text_dim,
            layernorm_across_heads=self._model_config.layernorm_across_heads,
            qkv_format=self._model_config.qkv_format,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=self._model_config.fp16_lm_cross_entropy,
            parallel_output=self._model_config.parallel_output,
        )


__all__ = ["WanModelBuilder", "WanModelConfig"]
