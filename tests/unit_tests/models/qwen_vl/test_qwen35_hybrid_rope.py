# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Core rotary-call compatibility and runtime CP slicing for Qwen3.5 VL."""

from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.qwen_vl.modeling_qwen35.hybrid_model import (
    _POSITION_IDS,
    _HybridMultimodalRotaryEmbedding,
)


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("packed_seq", [False, True])
def test_hybrid_rotary_uses_runtime_cp_group(packed_seq):
    construction_group = Mock()
    construction_group.size.return_value = 2
    construction_group.rank.return_value = 0
    runtime_group = Mock()
    runtime_group.size.return_value = 2
    runtime_group.rank.return_value = 1
    unsharded_group = Mock()
    unsharded_group.size.return_value = 1

    rotary = _HybridMultimodalRotaryEmbedding(kv_channels=8, cp_group=construction_group)
    rotary.mrope_section = [2, 1, 1]
    rotary.is_thd_format = packed_seq
    token = _POSITION_IDS.set(torch.arange(8).reshape(1, 8))
    try:
        full = rotary(8, packed_seq=packed_seq, cp_group=unsharded_group)
        actual = rotary(8, packed_seq=packed_seq, cp_group=runtime_group)
        default = rotary(8, packed_seq=packed_seq, cp_group=None)
    finally:
        _POSITION_IDS.reset(token)

    # CP2's mirrored chunks assign [0, 1, 6, 7] to rank 0 and [2, 3, 4, 5] to rank 1.
    # THD positions are already prepared for the local packed batch and are not sliced here.
    torch.testing.assert_close(actual, full if packed_seq else full[[2, 3, 4, 5]])
    torch.testing.assert_close(default, full if packed_seq else full[[0, 1, 6, 7]])
    assert rotary.cp_group is construction_group
