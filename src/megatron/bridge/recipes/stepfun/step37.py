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

from megatron.bridge.recipes.stepfun.h100.step37 import (
    step37_sft_4gpu_h100_bf16_flickr8k_smoke_config as step37_sft_flickr8k_smoke_config,
)
from megatron.bridge.recipes.stepfun.h100.step37 import (
    step37_sft_64gpu_h100_bf16_flickr8k_config as step37_sft_flickr8k_config,
)


__all__ = [
    "step37_sft_flickr8k_config",
    "step37_sft_flickr8k_smoke_config",
]
