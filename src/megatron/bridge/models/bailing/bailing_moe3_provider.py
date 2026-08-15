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

"""Serializable provider fields needed by the temporary Ling MCore path."""

from dataclasses import dataclass

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider


@dataclass
class BailingMoe3HybridProvider(HybridModelProvider):
    """Hybrid provider shared by the public Ling 3.0 model variants.

    The temporary MCore checkout used for Ling 3.0 exposes these attributes only
    dynamically on ``TransformerConfig``.  Declaring them here makes the provider
    self-contained and ensures that Bridge's ``run_config.yaml`` can reconstruct
    the exact model for DCP reload.  Variant selection remains config-driven in
    the bridge; there is deliberately no Tiny-specific or Flash-specific provider
    class.
    """

    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_head_dim: int | None = None
    qk_pos_emb_head_dim: int | None = None
    v_head_dim: int | None = None
    rope_type: str = "rope"
    rotary_scaling_factor: float = 1.0
    original_max_position_embeddings: int | None = None
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    mscale_all_dim: float = 0.0
    cache_mla_latents: bool = False
    mla_down_proj_fusion: bool = False


# Keep previously generated temporary DCP run_config.yaml files loadable after
# the provider was renamed to an architecture-level class name.
BailingMoe3TinyHybridProvider = BailingMoe3HybridProvider
