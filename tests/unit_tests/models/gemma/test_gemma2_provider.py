# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.bridge.models.gemma.gemma2_provider import Gemma2FlexDotProductAttention


@pytest.mark.unit
@pytest.mark.parametrize(
    ("window_left", "expected"),
    [
        (-1, [[True, True, True, True]]),
        (2, [[False, True, True, True]]),
    ],
)
def test_flex_block_mask_right_aligns_cached_decode_queries(window_left, expected):
    """Cached decode queries use absolute positions at the right edge of the KV cache."""
    captured_mask = None

    def capture_mask(mask_mod, *, Q_LEN, KV_LEN, **kwargs):
        del kwargs
        nonlocal captured_mask
        query_indices = torch.arange(Q_LEN)[:, None]
        key_value_indices = torch.arange(KV_LEN)[None, :]
        captured_mask = mask_mod(None, None, query_indices, key_value_indices)
        return object()

    attention = SimpleNamespace(_flex_window_size=(window_left, 0))
    with patch(
        "megatron.bridge.models.gemma.gemma2_provider._create_flex_block_mask",
        side_effect=capture_mask,
    ):
        Gemma2FlexDotProductAttention._build_flex_block_mask(
            attention,
            sq=1,
            sk=4,
            device=torch.device("cpu"),
        )

    assert captured_mask.tolist() == expected
