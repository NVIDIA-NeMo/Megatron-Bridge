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

"""Step3.7 multimodal model package — mirrors ``modelling_qwen3_vl``.

Re-exports the public classes consumed by the bridge / provider so they can
be imported from ``megatron.bridge.models.stepfun.modelling_step37`` as a
flat namespace.
"""

from megatron.bridge.models.stepfun.modelling_step37.model import Step37Model
from megatron.bridge.models.stepfun.modelling_step37.text_model import Step37GPTModel
from megatron.bridge.models.stepfun.modelling_step37.transformer_config import (
    Step37TransformerConfig,
    get_vision_model_config,
)
from megatron.bridge.models.stepfun.modelling_step37.vision_model import Step37VisionModel


__all__ = [
    "Step37Model",
    "Step37GPTModel",
    "Step37TransformerConfig",
    "Step37VisionModel",
    "get_vision_model_config",
]
