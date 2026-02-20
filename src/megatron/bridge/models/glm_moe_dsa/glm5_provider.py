# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from dataclasses import dataclass

from megatron.bridge.models.deepseek.deepseek_provider import DeepSeekV3ModelProvider


logger = logging.getLogger(__name__)


@dataclass
class GLM5ModelProvider(DeepSeekV3ModelProvider):
    """GLM5 models share DeepSeek-V3.2 architecture defaults."""

    moe_aux_loss_coeff: float = 0.001
    sparse_attention_type: str = "dsa"
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 2048
