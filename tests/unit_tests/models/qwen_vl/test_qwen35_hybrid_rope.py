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
        # A forward passes this rank's sequence length, 4 of the 8 tokens under CP2.
        actual = rotary(4, packed_seq=packed_seq, cp_group=runtime_group)
        default = rotary(4, packed_seq=packed_seq, cp_group=None)
    finally:
        _POSITION_IDS.reset(token)

    # CP2's mirrored chunks assign [0, 1, 6, 7] to rank 0 and [2, 3, 4, 5] to rank 1.
    # THD positions are already prepared for the local packed batch and are not sliced here.
    torch.testing.assert_close(actual, full if packed_seq else full[[2, 3, 4, 5]])
    torch.testing.assert_close(default, full if packed_seq else full[[0, 1, 6, 7]])
    assert rotary.cp_group is construction_group


@pytest.mark.unit
@pytest.mark.parametrize(("cp_rank", "local_positions"), [(0, [0, 1, 6, 7]), (1, [2, 3, 4, 5])])
def test_hybrid_rotary_keeps_cp_sharded_position_ids(monkeypatch, cp_rank, local_positions):
    # MegatronMIMO shards the batch for CP before the language model runs, mRoPE position ids included
    # (Megatron-Core's PartitionAdapter), so these ids already line up with the rank's local tokens.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    unsharded_group = Mock()
    unsharded_group.size.return_value = 1
    cp_group = Mock()
    cp_group.size.return_value = 2
    cp_group.rank.return_value = cp_rank
    rotary = _HybridMultimodalRotaryEmbedding(kv_channels=8, cp_group=cp_group)
    rotary.mrope_section = [2, 1, 1]
    text_positions = torch.arange(8)
    position_ids = torch.stack((text_positions, text_positions + 10, text_positions + 20))[:, None, :]

    token = _POSITION_IDS.set(position_ids)
    try:
        full = rotary(8, packed_seq=False, cp_group=unsharded_group)
    finally:
        _POSITION_IDS.reset(token)
    token = _POSITION_IDS.set(position_ids[..., local_positions])
    try:
        local = rotary(4, packed_seq=False, cp_group=cp_group)
    finally:
        _POSITION_IDS.reset(token)

    torch.testing.assert_close(local, full[local_positions])


@pytest.mark.unit
def test_hybrid_rotary_keeps_packed_position_ids(monkeypatch):
    # Packed (THD) ids are partitioned per sequence with the local tokens before the language model runs, and
    # the rotary call then gets the longest packed sequence as its length, not the number of local tokens.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    unsharded_group = Mock()
    unsharded_group.size.return_value = 1
    cp_group = Mock()
    cp_group.size.return_value = 2
    cp_group.rank.return_value = 1
    rotary = _HybridMultimodalRotaryEmbedding(kv_channels=8, cp_group=cp_group)
    rotary.mrope_section = [2, 1, 1]
    text_positions = torch.tensor([1, 2, 5, 6, 1, 2])
    position_ids = torch.stack((text_positions, text_positions + 10, text_positions + 20))[:, None, :]

    token = _POSITION_IDS.set(position_ids)
    try:
        expected = rotary(6, packed_seq=False, cp_group=unsharded_group)
        local = rotary(8, packed_seq=True, cp_group=cp_group)
    finally:
        _POSITION_IDS.reset(token)

    torch.testing.assert_close(local, expected)


@pytest.mark.unit
@pytest.mark.parametrize(("local_tokens", "sequence_parallel", "tp_size"), [(4, False, 1), (2, True, 2)])
def test_hybrid_rotary_seq_len_is_rank_local(monkeypatch, local_tokens, sequence_parallel, tp_size):
    # The rotary call tells CP-sharded ids from full-sequence ids by this length, so it must stay the
    # rank's CP-local length (sequence-parallel shards gathered back), not the full sequence.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    group = Mock()
    group.size.return_value = 2
    rotary = _HybridMultimodalRotaryEmbedding(kv_channels=8, cp_group=group)
    config = Mock(sequence_parallel=sequence_parallel, tensor_model_parallel_size=tp_size, context_parallel_size=2)

    assert rotary.get_rotary_seq_len(None, None, torch.empty(local_tokens, 1, 8), config) == 4


@pytest.mark.unit
def test_hybrid_rotary_defers_to_layer_buffer_outside_forward(monkeypatch):
    # Core's TE CUDA graph helper unwraps the VLM to this Hybrid language model, sees "rope", and asks for a
    # static embedding before any forward, as rotary(seq_len, packed_seq=...). It must get None so it keeps
    # the batch-shaped rotary buffer of Qwen3VLRotaryTransformerLayer.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    group = Mock()
    group.size.return_value = 1
    rotary = _HybridMultimodalRotaryEmbedding(kv_channels=8, cp_group=group)
    rotary.mrope_section = [2, 1, 1]
    rotary.te_attn_cuda_graph = True

    assert rotary(8, packed_seq=False) is None

    # Inside a forward, missing positions are still an error.
    token = _POSITION_IDS.set(None)
    try:
        with pytest.raises(RuntimeError, match="requires the current forward's position_ids"):
            rotary(8)
    finally:
        _POSITION_IDS.reset(token)
