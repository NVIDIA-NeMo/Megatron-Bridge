# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Bridge support for AllenAI's OLMo-2 dense causal LM family.

OLMo-2 is the second-generation fully-open language model from the Allen
Institute. Compared to OLMo-1 and OLMoE, OLMo-2 introduces two architectural
changes that motivate this bridge:

* **Pure post-norm placement.** The decoder block is::

      x = x + post_attention_layernorm(self_attn(x))
      x = x + post_feedforward_layernorm(mlp(x))

  with no `input_layernorm` or `pre_feedforward_layernorm` (vs. Llama/Qwen3
  which pre-norm both sub-blocks, or Gemma2 which sandwiches both pre and
  post). See `_olmo2_layer_spec` in `olmo2_provider.py`.

* **QK-RMSNorm** applied to the per-head Q and K projections inside attention.
  Megatron-Core supports this via `qk_layernorm=True` plus mappings for the
  `q_layernorm` / `k_layernorm` weights — same as Qwen3 and OLMoE.

Reference: Yang et al., 2024, *2 OLMo 2 Furious* (https://arxiv.org/abs/2501.00656).
"""

from megatron.bridge.models.olmo2.olmo2_bridge import Olmo2Bridge
from megatron.bridge.models.olmo2.olmo2_provider import (
    Olmo2ModelProvider,
    Olmo2ModelProvider1B,
    Olmo2ModelProvider7B,
    Olmo2ModelProvider13B,
    olmo2_layer_spec,
)


__all__ = [
    "Olmo2Bridge",
    "Olmo2ModelProvider",
    "Olmo2ModelProvider1B",
    "Olmo2ModelProvider7B",
    "Olmo2ModelProvider13B",
    "olmo2_layer_spec",
]
