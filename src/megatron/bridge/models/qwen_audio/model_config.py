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

"""Pure model configuration and builder for Qwen2-Audio."""

from copy import copy
from dataclasses import dataclass, field
from typing import Any, ClassVar

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder
from transformers import Qwen2AudioConfig

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.qwen_audio.modeling_qwen2_audio import Qwen2AudioModel


@dataclass(kw_only=True)
class Qwen2AudioModelConfig(BridgeGPTModelConfig):
    """Serializable builder input for Qwen2-Audio."""

    builder: ClassVar[str] = "megatron.bridge.models.qwen_audio.model_config.Qwen2AudioModelBuilder"
    hf_config: dict[str, Any] = field(default_factory=dict)
    audio_token_id: int = 151646
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    freeze_language_model: bool = False
    freeze_audio_model: bool = False
    freeze_audio_projection: bool = False
    gradient_accumulation_fusion: bool = False


class Qwen2AudioModelBuilder(GPTModelBuilder):
    """Build Qwen2-Audio without a model provider."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Qwen2AudioModel:
        """Build one Qwen2-Audio pipeline stage.

        Args:
            pg_collection: Process groups for distributed construction.
            pre_process: Whether this stage owns input processing.
            post_process: Whether this stage owns output processing.
            vp_stage: Virtual pipeline stage index.

        Returns:
            Constructed Qwen2-Audio stage.
        """
        config = self._model_config
        transformer = copy(config.transformer)
        transformer.audio_token_id = config.audio_token_id
        transformer.pad_token_id = config.pad_token_id
        runtime_config = copy(config)
        runtime_config.transformer = transformer
        language_model = GPTModelBuilder(runtime_config).build_model(
            pg_collection, pre_process, post_process, vp_stage
        )
        if pre_process is None:
            pre_process = language_model.pre_process
        if post_process is None:
            post_process = language_model.post_process
        hf_config = Qwen2AudioConfig(**config.hf_config)
        model = Qwen2AudioModel(
            runtime_config,
            pre_process,
            post_process,
            vp_stage,
            language_model=language_model,
            hf_config=hf_config,
        )
        if config.freeze_language_model or config.freeze_audio_model or config.freeze_audio_projection:
            model.freeze(config.freeze_language_model, config.freeze_audio_model, config.freeze_audio_projection)
        return model


__all__ = ["Qwen2AudioModelBuilder", "Qwen2AudioModelConfig"]
