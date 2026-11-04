# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt current Whisper attention to MiniCPM-o's offline encoder contract."""

from typing import Any

import torch
from transformers.models.whisper.modeling_whisper import WhisperAttention


class MiniCPMOAudioAttention(WhisperAttention):
    """Keep HF attention math and parameter names with a legacy return tuple.

    MiniCPM-o 4.5's encoder layer unpacks three values, whereas current HF
    Whisper returns two. Offline understanding does not consume a KV cache.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        past_key_value: Any = None,
        past_key_values: Any = None,
        layer_head_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        """Return the native attention outputs plus an unused cache slot."""
        if past_key_value is not None or past_key_values is not None:
            raise NotImplementedError("MiniCPM-o understanding does not support audio KV caching.")
        if layer_head_mask is not None:
            raise NotImplementedError("MiniCPM-o understanding does not support audio head masking.")
        states, weights = super().forward(hidden_states=hidden_states, **kwargs)
        return states, weights, None
