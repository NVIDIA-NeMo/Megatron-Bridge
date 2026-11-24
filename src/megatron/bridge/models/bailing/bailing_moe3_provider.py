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

"""HybridModel provider for Ling 3.0."""

from dataclasses import dataclass

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig


@dataclass
class BailingMoe3HybridProvider(HybridModelProvider, MLATransformerConfig):
    """Combine HybridModel construction with MLA configuration for Ling 3.0."""

    multi_latent_attention: bool = True
