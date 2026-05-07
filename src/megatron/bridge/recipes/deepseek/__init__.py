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

"""DeepSeek recipe exports.

This module re-exports AutoBridge-based pretrain config helpers for DeepSeek
models (V2, V2-Lite, V3).
"""

# DeepSeek V2/V2-Lite
from .deepseek_v2 import (
    deepseek_v2_lite_pretrain_config,
    deepseek_v2_pretrain_config,
)

# DeepSeek V3
from .deepseek_v3 import (
    deepseek_v3_pretrain_config,
    deepseek_v3_pretrain_config_32nodes,
)


__all__ = [
    # DeepSeek V2/V2-Lite
    "deepseek_v2_pretrain_config",
    "deepseek_v2_lite_pretrain_config",
    # DeepSeek V3
    "deepseek_v3_pretrain_config",
    "deepseek_v3_pretrain_config_32nodes",
]
