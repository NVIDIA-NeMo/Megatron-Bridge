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
"""Common helpers for kimi performance recipes."""

from megatron.bridge.perf_recipes._common import _benchmark_common, _perf_precision
from megatron.bridge.recipes.kimi.kimi_k2 import (
    _get_kimi_k2_pipeline_layout,
    kimi_k2_pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer
