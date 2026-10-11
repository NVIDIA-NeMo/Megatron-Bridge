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

"""Targeted corrections for released Transformers MiniMax implementations."""

from typing import Any

import torch
import transformers


if transformers.__version__ == "5.17.0":
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLVisionRotaryEmbedding as _HFVisionRotaryEmbedding,
    )

    class _MiniMaxM3Vision3DRotaryEmbedding(_HFVisionRotaryEmbedding):
        _bridge_3d_rope = True

        @staticmethod
        def compute_axial_rope_parameters(
            config: Any, device: torch.device | None = None, **kwargs: Any
        ) -> tuple[torch.Tensor, float]:
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            axis_dim = 2 * ((head_dim // 3) // 2)
            theta = config.rope_parameters["rope_theta"]
            inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim))
            return inv_freq, 1.0

        def recomposition_frequencies(self, freq: torch.Tensor) -> torch.Tensor:
            frequencies = freq.flatten(1)
            return torch.cat((frequencies, frequencies), dim=-1)

        def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Match 5.16 even after model.half()/bfloat16() rounds the inherited buffers.
            device_type = x.device.type if x.device.type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                inv_freq, scaling = self.compute_axial_rope_parameters(self.config, x.device)
                frequencies = position_ids.to(device=x.device, dtype=torch.float32)[..., None] * inv_freq
                cosine = self.recomposition_frequencies(frequencies.cos() * scaling)
                sine = self.recomposition_frequencies(frequencies.sin() * scaling)
            return cosine.to(x.dtype), sine.to(x.dtype)


def patch_minimax_m3_vision_rope() -> bool:
    """Restore 3D vision RoPE for new HF MiniMax models on Transformers 5.17.0.

    Transformers PR #48105 changed the frequency ladder to 2D and dropped the
    width coordinate from the T/H/W position IDs. Keep the released checkpoint's
    partial 3D rotation until an upstream fixed release is available. Replacing
    the class affects new MiniMax models only. Import Bridge before constructing
    HF MiniMax models; existing instances retain their old numerical behavior
    and should be reconstructed before serialization.
    """
    if transformers.__version__ != "5.17.0":
        return False

    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as modeling

    original = modeling.MiniMaxM3VLVisionRotaryEmbedding
    if getattr(original, "_bridge_3d_rope", False):
        return True

    modeling.MiniMaxM3VLVisionRotaryEmbedding = _MiniMaxM3Vision3DRotaryEmbedding
    return True
