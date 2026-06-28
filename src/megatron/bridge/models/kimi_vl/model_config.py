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

"""Serializable Kimi K2.5 VL config and builder."""

from dataclasses import dataclass, field
from typing import ClassVar

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.kimi.kimi_bridge import KimiK2ModelConfig
from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel


@dataclass(kw_only=True)
class KimiK25VLModelConfig(KimiK2ModelConfig):
    """Pure-data Kimi K2.5 multimodal build configuration."""

    builder: ClassVar[str] = "megatron.bridge.models.kimi_vl.model_config.KimiK25VLModelBuilder"
    hf_model_id: str = ""
    trust_remote_code: bool = False
    vision_config: dict[str, object] = field(default_factory=dict)
    generation_config: dict[str, object] | None = None
    bos_token_id: int = 163584
    eos_token_id: int = 163585
    image_token_id: int = 163605
    media_placeholder_token_id: int = 163605
    pad_token_id: int = 163839
    ignore_index: int = -100
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False


class KimiK25VLModelBuilder(GPTModelBuilder):
    """Build the Kimi MLA language stage and dynamic HF vision wrapper."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> KimiK25VLModel:
        """Build one Kimi K2.5 VL pipeline stage."""
        language_model = super().build_model(pg_collection, pre_process, post_process, vp_stage)
        config = self._model_config
        model = KimiK25VLModel(
            config=config,
            language_model=language_model,
            pg_collection=pg_collection,
            pre_process=pre_process if pre_process is not None else True,
            post_process=post_process if post_process is not None else True,
            vp_stage=vp_stage,
        )
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_vision_projection:
            model.freeze(
                freeze_language_model=config.freeze_language_model,
                freeze_vision_model=config.freeze_vision_model,
                freeze_vision_projection=config.freeze_vision_projection,
            )
        return model


__all__ = ["KimiK25VLModelBuilder", "KimiK25VLModelConfig"]
