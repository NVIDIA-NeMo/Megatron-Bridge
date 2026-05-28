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

"""LLaDA15TEDotProductAttention: TE-backed core attention for LLaDA1.5.

LLaDA1.5 is a block-diffusion masked diffusion language model. Unlike LLaDA2,
the *reference implementation* in modeling_llada.py uses **fully bidirectional
attention at both training and inference time** — the block structure exists
only in the sampling schedule (which positions get unmasked per iteration),
not in the attention pattern.

This shim therefore needs only one job: override Megatron's default
``AttnMaskType.causal`` to ``AttnMaskType.no_mask`` so attention is
bidirectional. RoPE is full and handled upstream by Megatron's
``SelfAttention`` — this module does not touch RoPE.

The ``set_block_mask`` / ``reset_inference_state`` hooks are kept as no-op
extension points for users who want to experiment with block-diagonal
attention (the LLaDA2-style design) — they are not wired into the default
inference loop.
"""

from typing import Optional

import torch
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor


class LLaDA15TEDotProductAttention(TEDotProductAttention):
    """TE-backed core attention for LLaDA1.5 masked-diffusion training/inference.

    Overrides ``attn_mask_type`` to ``AttnMaskType.no_mask`` so the model
    sees fully bidirectional attention, matching modeling_llada.py's
    ``get_bidirectional_attention_bias`` (a zero tensor) at line 1273.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        **kwargs,
    ):
        # Force bidirectional attention regardless of what the layer spec set.
        # This must happen before super().__init__ because TE caches mask
        # behavior at construction.
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.no_mask,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            **kwargs,
        )
        # Optional extension state for users experimenting with block-diagonal
        # attention. Default LLaDA1.5 inference does not use these.
        self._block_attn_mask: Optional[Tensor] = None

    def set_block_mask(self, mask: Optional[Tensor]) -> None:
        """Install a boolean block-diagonal mask (True = blocked) for experimental use.

        Not used by the default LLaDA1.5 inference loop; reserved for users
        who want to layer LLaDA2-style block-diagonal attention on top.
        """
        self._block_attn_mask = mask

    def reset_inference_state(self) -> None:
        """Clear any stored inference state. Safe to call between generations."""
        self._block_attn_mask = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        if self._block_attn_mask is not None:
            S = query.shape[0]
            block = self._block_attn_mask[:, :, :S, :S]
            B = query.shape[1]
            bool_mask = (
                torch.isinf(block).expand(B, 1, S, S) if block.is_floating_point() else block.expand(B, 1, S, S)
            )
            return super().forward(
                query,
                key,
                value,
                attention_mask=bool_mask,
                attn_mask_type=AttnMaskType.arbitrary,
                attention_bias=None,
                packed_seq_params=packed_seq_params,
            )

        return super().forward(
            query,
            key,
            value,
            attention_mask=attention_mask,
            attn_mask_type=AttnMaskType.no_mask,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
