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
# ruff: noqa: F401
"""Compatibility aliases for legacy recipe names."""

from __future__ import annotations

from megatron.bridge.recipes.glm_vl.h100.glm_45v import (
    glm_45v_peft_32gpu_h100_bf16_config as glm_45v_peft_config,
)
from megatron.bridge.recipes.glm_vl.h100.glm_45v import (
    glm_45v_sft_128gpu_h100_bf16_config as glm_45v_sft_config,
)
from megatron.bridge.recipes.glm_vl.h100.glm_45v import (
    set_glm_45v_pipeline_model_parallel_layout,
)


__all__ = [
    "glm_45v_peft_config",
    "glm_45v_sft_config",
    "set_glm_45v_pipeline_model_parallel_layout",
]
