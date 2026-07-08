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

"""Pure Step3.7 build configuration and standalone builder."""

from dataclasses import dataclass, field
from typing import Callable, ClassVar

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.stepfun.configuration_step37 import Step37VisionConfig
from megatron.bridge.models.stepfun.modelling_step37.model import Step37Model
from megatron.bridge.models.stepfun.step35_modeling import build_step35_layer_spec


@dataclass(kw_only=True)
class Step37ModelConfig(BridgeGPTModelConfig):
    """Serializable Step3.7 config with family fields outside MCore config."""

    builder: ClassVar[str] = "megatron.bridge.models.stepfun.step37_model_config.Step37ModelBuilder"
    transformer_layer_spec: Callable[..., TransformerBlockSubmodules] = field(
        default_factory=lambda: build_step35_layer_spec
    )
    layer_types: list[str] | None = None
    attention_other_setting: dict[str, object] | None = None
    sliding_attention_setting: dict[str, object] | None = None
    rotary_base_per_layer: list[float] | None = None
    rotary_percents: list[float] | None = None
    swiglu_limits: list[float | None] | None = None
    swiglu_limits_shared: list[float | None] | None = None
    head_wise_attn_gate: bool = False
    vision_config: dict[str, object] = field(default_factory=dict)
    image_token_id: int = 128001
    understand_projector_stride: int = 2
    projector_bias: bool = False
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    add_encoder: bool = True
    add_decoder: bool = True


class Step37ModelBuilder(GPTModelBuilder):
    """Build Step3.7 without importing or instantiating a provider."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Step37Model:
        """Build one Step3.7 pipeline stage."""
        config = self._model_config
        language_model = super().build_model(
            pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        model = Step37Model(
            language_transformer_config=config.transformer,
            language_model=language_model,
            vision_transformer_config=Step37VisionConfig(**config.vision_config),
            image_token_id=config.image_token_id,
            projector_bias=config.projector_bias,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            add_encoder=config.add_encoder,
            add_decoder=config.add_decoder,
            pg_collection=pg_collection,
        )
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_vision_projection:
            model.freeze(
                config.freeze_language_model,
                config.freeze_vision_model,
                config.freeze_vision_projection,
            )
        return model


__all__ = ["Step37ModelBuilder", "Step37ModelConfig"]
