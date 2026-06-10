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

"""Step3.7 transformer config and vision-config helper.

Mirrors ``qwen_vl/modelling_qwen3_vl/transformer_config.py``: the text-side
config is the standard Megatron ``TransformerConfig`` already used by Step-3.5,
extended with vision-tower fields. The HF ``StepRoboticsVisionEncoderConfig``
is passed straight through to the Megatron vision module — no separate
Megatron-side ``TransformerConfig`` is constructed for the vision tower, since
the PE-G/14 trunk does not use any Megatron tensor-parallel primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class Step37TransformerConfig(TransformerConfig):
    """Step3.7 transformer config.

    Extends the Step-3.5 text-decoder ``TransformerConfig`` with the multimodal
    fields that ``Step37Model`` reads at construction time. All Step-3.5
    per-layer fields (``layer_types``, ``rotary_percents``,
    ``rotary_base_per_layer``, ``swiglu_limits``, ``swiglu_limits_shared``,
    ``attention_other_setting``, ``sliding_attention_setting``,
    ``head_wise_attn_gate``) are inherited from the Step-3.5 model provider —
    this class only adds the vision-side fields.
    """

    vision_config: Any = None
    image_token_id: int = 128001
    understand_projector_stride: int = 2
    projector_bias: bool = False
    language_max_sequence_length: int = 262144


def get_vision_model_config(vision_cfg: Any) -> Any:
    """Return the HF vision config unchanged.

    ``Step37VisionModel`` consumes the HF ``StepRoboticsVisionEncoderConfig``
    directly (it never uses Megatron tensor-parallel primitives), so this
    function is just a structural mirror of
    ``qwen_vl/modelling_qwen3_vl/transformer_config.get_vision_model_config``
    for parity with the Qwen3-VL package shape. It is intentionally a no-op.
    """
    return vision_cfg
