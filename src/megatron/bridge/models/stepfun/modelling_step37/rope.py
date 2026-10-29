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

"""Step3.7 RoPE — per-layer partial rotary applied by ``Step37DecoderLayer``.

The text decoder uses Step-3.5's per-layer ``rotary_percents`` /
``rotary_base_per_layer`` mechanism (see
``stepfun/step35_provider.Step35DecoderLayer.__init__``). The vision tower's
own 2D RoPE lives next to the attention block, in
``modelling_step37/utils.EncoderRope2D``. This module exists to mirror
``qwen_vl/modelling_qwen3_vl/rope.py``; it re-exports those entry points so
downstream code can import them from a single, Step3.7-namespaced location.
"""

from megatron.bridge.models.stepfun.modelling_step37.utils import (
    EncoderRope2D,
    apply_rotary_emb,
    rotate_half,
)


__all__ = ["EncoderRope2D", "apply_rotary_emb", "rotate_half"]
