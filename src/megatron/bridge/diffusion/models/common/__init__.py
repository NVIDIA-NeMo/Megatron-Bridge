# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Common modules for diffusion models in DFM."""

from megatron.bridge.diffusion.models.common.dit_attention import (
    DiTCrossAttention,
    DiTCrossAttentionSubmodules,
    DiTSelfAttention,
)
from megatron.bridge.diffusion.models.common.dit_embeddings import (
    FactorizedLearnable3DEmbedding,
    ParallelTimestepEmbedding,
    SinCosPosEmb3D,
)
from megatron.bridge.diffusion.models.common.normalization import RMSNorm


__all__ = [
    # Attention modules
    "DiTCrossAttention",
    "DiTCrossAttentionSubmodules",
    "DiTSelfAttention",
    # Embeddings
    "FactorizedLearnable3DEmbedding",
    "ParallelTimestepEmbedding",
    "SinCosPosEmb3D",
    # Normalization
    "RMSNorm",
]
