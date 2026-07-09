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

from megatron.bridge.recipes.deepseek.h100.deepseek_v3 import (
    _build_standalone_mtp_layout,
    set_deepseek_v3_pipeline_model_parallel_layout,
)
from megatron.bridge.recipes.deepseek.h100.deepseek_v3 import (
    deepseek_v3_pretrain_256gpu_h100_bf16_32nodes_config as deepseek_v3_pretrain_config_32nodes,
)
from megatron.bridge.recipes.deepseek.h100.deepseek_v3 import (
    deepseek_v3_pretrain_1024gpu_h100_bf16_config as deepseek_v3_pretrain_config,
)


__all__ = [
    "deepseek_v3_pretrain_config",
    "deepseek_v3_pretrain_config_32nodes",
    "set_deepseek_v3_pipeline_model_parallel_layout",
]
