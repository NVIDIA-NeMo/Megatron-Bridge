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

"""Serializable Ministral 3 model config and builder."""

from dataclasses import dataclass, field
from typing import Callable, ClassVar

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.ministral3.layer_specs import ministral_layer_spec
from megatron.bridge.models.ministral3.modeling_ministral3 import Ministral3Model


@dataclass(kw_only=True)
class Ministral3ModelConfig(BridgeGPTModelConfig):
    """Pure-data build configuration for Ministral 3 VLMs."""

    builder: ClassVar[str] = "megatron.bridge.models.ministral3.model_config.Ministral3ModelBuilder"
    transformer_layer_spec: Callable[..., ModuleSpec] = field(default_factory=lambda: ministral_layer_spec)
    hf_config: dict[str, object] = field(default_factory=dict)
    image_token_id: int = 10
    spatial_merge_size: int = 2
    vision_feature_layer: int = -1
    return_dict: bool = True
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    yarn_rotary_scaling_factor: float = 16.0
    yarn_original_max_position_embeddings: int = 16384
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_correction_range_round_to_int: bool = False
    yarn_mscale: float | None = 1.0
    yarn_mscale_all_dim: float | None = 1.0
    llama_4_scaling_beta: float = 0.0
    llama_4_original_max_position_embeddings: int = 16384


class Ministral3ModelBuilder(GPTModelBuilder):
    """Build the MCore language model and HF-vision Ministral wrapper."""

    _TRANSIENT_FIELDS = (
        "yarn_rotary_scaling_factor",
        "yarn_original_max_position_embeddings",
        "yarn_beta_fast",
        "yarn_beta_slow",
        "yarn_correction_range_round_to_int",
        "yarn_mscale",
        "yarn_mscale_all_dim",
        "llama_4_scaling_beta",
        "llama_4_original_max_position_embeddings",
    )

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Ministral3Model:
        """Build one Ministral pipeline stage."""
        config = self._model_config
        transformer = config.transformer
        missing = object()
        previous = {name: getattr(transformer, name, missing) for name in self._TRANSIENT_FIELDS}
        try:
            for name in self._TRANSIENT_FIELDS:
                setattr(transformer, name, getattr(config, name))
            language_model: GPTModel = super().build_model(
                pg_collection,
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            )
        finally:
            for name in self._TRANSIENT_FIELDS:
                if previous[name] is missing:
                    delattr(transformer, name)
                else:
                    setattr(transformer, name, previous[name])

        model = Ministral3Model(
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


__all__ = ["Ministral3ModelBuilder", "Ministral3ModelConfig"]
