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

from dataclasses import dataclass, field

from megatron.core.models.gpt import GPTModel as MCoreGPTModel

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.step_vl.modeling_step3_vl.configuration import (
    StepRoboticsVisionEncoderConfig,
)


@dataclass
class Step3VLModelProvider(GPTModelProvider):
    """
    Model provider for Step3-VL (Step3-VL-10B and variants).

    Language-model parameters are populated by ``Step3VLBridge.provider_bridge``
    via ``hf_config_to_provider_kwargs(text_config)``.  Only VLM-specific fields
    that are NOT part of the language model are defined here.
    """

    # VLMs must not scatter embeddings across SP regions because image tokens
    # are inserted into the language embedding sequence.
    scatter_embedding_sequence_parallel: bool = False

    # Vision encoder configuration
    vision_config: StepRoboticsVisionEncoderConfig = field(
        default_factory=StepRoboticsVisionEncoderConfig
    )
    projector_bias: bool = False

    # Image placeholder token in the vocabulary
    image_token_id: int = 151679

    # Freeze options for fine-tuning
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Instantiate Step3VLModel (vision + language)."""
        from megatron.bridge.models.step_vl.modeling_step3_vl.model import Step3VLModel

        model = Step3VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Instantiate only the Megatron GPT language model (no vision)."""
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
