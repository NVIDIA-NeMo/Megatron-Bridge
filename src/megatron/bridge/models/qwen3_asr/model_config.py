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

"""Pure model configuration and builder for Qwen3-ASR."""

from copy import copy
from dataclasses import dataclass, field
from typing import Any, ClassVar

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.qwen3_asr.hf_qwen3_asr.configuration_qwen3_asr import Qwen3ASRThinkerConfig
from megatron.bridge.models.qwen3_asr.modeling_qwen3_asr.model import Qwen3ASRModel


@dataclass(kw_only=True)
class Qwen3ASRModelConfig(BridgeGPTModelConfig):
    """Serializable builder input for Qwen3-ASR."""

    builder: ClassVar[str] = "megatron.bridge.models.qwen3_asr.model_config.Qwen3ASRModelBuilder"
    thinker_config: dict[str, Any] = field(default_factory=dict)
    audio_token_id: int = 151646
    audio_start_token_id: int = 151647
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    language_max_sequence_length: int = 2048
    freeze_language_model: bool = False
    freeze_audio_model: bool = False


class Qwen3ASRModelBuilder(GPTModelBuilder):
    """Build Qwen3-ASR without a model provider."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Qwen3ASRModel:
        """Build one Qwen3-ASR pipeline stage.

        Args:
            pg_collection: Process groups for distributed construction.
            pre_process: Whether this stage owns input processing.
            post_process: Whether this stage owns output processing.
            vp_stage: Virtual pipeline stage index.

        Returns:
            Constructed Qwen3-ASR stage.
        """
        config = self._model_config
        transformer = copy(config.transformer)
        vp_size = config.transformer.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)
        for name in ("audio_token_id", "audio_start_token_id", "mrope_section", "language_max_sequence_length"):
            setattr(transformer, name, getattr(config, name))
        transformer.vocab_size = config.vocab_size
        transformer.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        transformer.rotary_base = config.rotary_base
        thinker = Qwen3ASRThinkerConfig(**config.thinker_config)
        spec = get_gpt_layer_with_transformer_engine_spec(None, False, transformer.qk_layernorm, fp8=False)
        model = Qwen3ASRModel(
            transformer,
            spec,
            thinker,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
        )
        if config.freeze_language_model or config.freeze_audio_model:
            model.freeze(config.freeze_language_model, config.freeze_audio_model)
        return model


__all__ = ["Qwen3ASRModelBuilder", "Qwen3ASRModelConfig"]
