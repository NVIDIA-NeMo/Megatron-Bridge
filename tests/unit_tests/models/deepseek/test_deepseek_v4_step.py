# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for deepseek_v4_step.py contiguous CP partition logic."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.bridge.models.deepseek.deepseek_v4_step as dsv4_step
from megatron.bridge.models.deepseek.deepseek_v4_step import _partition_packed_batch_contiguous


pytestmark = pytest.mark.unit


def _make_batch(tokens=None, cu_seqlens=None, **extra):
    batch = {}
    if tokens is not None:
        batch["tokens"] = tokens
    if cu_seqlens is not None:
        batch["cu_seqlens"] = cu_seqlens
    batch.update(extra)
    return batch


class TestPartitionPackedBatchContiguous:
    """Tests for _partition_packed_batch_contiguous."""

    def _run(self, monkeypatch, batch, cp_rank=0, cp_size=2):
        monkeypatch.setattr(
            "megatron.bridge.models.deepseek.deepseek_v4_step.parallel_state.get_context_parallel_rank",
            lambda: cp_rank,
        )
        return _partition_packed_batch_contiguous(batch, cp_size)

    def test_rank0_receives_first_half(self, monkeypatch):
        """Rank 0 of 2 receives the first half of each data tensor."""
        tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)
        labels = torch.arange(8, dtype=torch.long).unsqueeze(0)
        cu_seqlens = torch.tensor([[0, 4, 8]], dtype=torch.int32)
        batch = _make_batch(tokens=tokens, labels=labels, cu_seqlens=cu_seqlens)
        result = self._run(monkeypatch, batch, cp_rank=0, cp_size=2)
        assert result["tokens"].shape == (1, 4)
        assert torch.equal(result["tokens"].squeeze(), torch.arange(4, dtype=torch.long))
        # cu_seqlens kept global — not partitioned (CSA needs global sequence boundaries)
        assert result["cu_seqlens"].squeeze().tolist() == [0, 4, 8]

    def test_rank1_receives_second_half(self, monkeypatch):
        """Rank 1 of 2 receives the second half."""
        tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)
        cu_seqlens = torch.tensor([[0, 4, 8]], dtype=torch.int32)
        batch = _make_batch(tokens=tokens, cu_seqlens=cu_seqlens)
        result = self._run(monkeypatch, batch, cp_rank=1, cp_size=2)
        assert result["tokens"].shape == (1, 4)
        assert torch.equal(result["tokens"].squeeze(), torch.arange(4, 8, dtype=torch.long))
        # cu_seqlens kept global — not partitioned (CSA needs global sequence boundaries)
        assert result["cu_seqlens"].squeeze().tolist() == [0, 4, 8]

    def test_rejects_non_divisible_length(self, monkeypatch):
        """Raises RuntimeError when total_tokens is not divisible by cp_size."""
        tokens = torch.arange(5, dtype=torch.long).unsqueeze(0)
        batch = _make_batch(tokens=tokens, cu_seqlens=torch.tensor([[0, 5]], dtype=torch.int32))
        with pytest.raises(RuntimeError, match="divisible by cp_size"):
            self._run(monkeypatch, batch, cp_size=2)

    def test_middle_pp_stage_returns_unchanged(self, monkeypatch):
        """Middle PP stage (all data tensors None) returns batch unchanged."""
        cu_seqlens = torch.tensor([[0, 4, 8]], dtype=torch.int32)
        batch = {"tokens": None, "labels": None, "loss_mask": None, "cu_seqlens": cu_seqlens}
        result = self._run(monkeypatch, batch, cp_size=2)
        assert result is batch


class TestPackedMetadataForForward:
    """Tests for _packed_metadata_for_forward."""

    def test_returns_none_for_empty_batch(self):
        batch = {"tokens": None, "labels": None}
        from megatron.bridge.models.deepseek.deepseek_v4_step import _packed_metadata_for_forward

        assert _packed_metadata_for_forward(batch) is None

    def test_legacy_path_extracts_boundaries_but_not_dev_cp_mode(self):
        from megatron.bridge.models.deepseek.deepseek_v4_step import _packed_metadata_for_forward

        batch = {
            "cu_seqlens": torch.tensor([[0, 4, 8]], dtype=torch.int32),
            "max_seqlen": torch.tensor([[8]]),
            "cp_partition_mode": "contiguous",
            "total_tokens": 8,
        }
        meta = _packed_metadata_for_forward(batch)
        assert meta is not None
        assert "cp_partition_mode" not in meta
        assert "cu_seqlens" in meta

    def test_current_path_with_cu_seqlens_q(self):
        from megatron.bridge.models.deepseek.deepseek_v4_step import _packed_metadata_for_forward

        batch = {
            "cu_seqlens_q": torch.tensor([[0, 4]], dtype=torch.int32),
            "max_seqlen_q": torch.tensor([[4]]),
            "cp_partition_mode": "contiguous",
        }
        meta = _packed_metadata_for_forward(batch)
        assert meta is not None
        assert "cp_partition_mode" not in meta
        assert "cu_seqlens_q" in meta


@pytest.mark.parametrize("rank", [0, 1])
def test_explicit_rank_preserves_global_metadata_and_token_fields(rank):
    tokens = torch.arange(16).view(1, 16)
    cu = torch.tensor([[0, 4, 16]], dtype=torch.int32)
    batch = {
        "tokens": tokens,
        "labels": tokens + 1,
        "position_ids": tokens,
        "loss_mask": torch.ones(1, 16),
        "cu_seqlens_q": cu,
        "cu_seqlens_kv": cu,
        "cu_seqlens_q_padded": cu,
        "cu_seqlens_kv_padded": cu,
        "max_seqlen_q": 12,
        "max_seqlen_kv": 12,
        "total_tokens": 16,
    }
    result = _partition_packed_batch_contiguous(dict(batch), 2, rank)
    for name in ("tokens", "labels", "position_ids", "loss_mask"):
        assert torch.equal(result[name], batch[name][:, rank * 8 : (rank + 1) * 8])
    for name in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
        assert result[name] is cu
    assert result["total_tokens"] == 16 and result["max_seqlen_q"] == 12


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("stage", ["first", "middle", "last"])
def test_native_layout_dispatch_preserves_packed_boundaries(monkeypatch, cp_size, stage):
    """Use the explicit CP rank and carry metadata through intermediate PP stages."""
    cp_rank = cp_size - 1
    tokens = torch.arange(16).view(1, 16)
    cu = torch.tensor([[0, 4, 16]], dtype=torch.int32)
    batch = {
        "tokens": tokens,
        "labels": tokens + 1,
        "loss_mask": torch.ones_like(tokens),
        "position_ids": tokens,
        "cu_seqlens_q": cu,
        "cu_seqlens_kv": cu,
        "cu_seqlens_q_padded": cu,
        "cu_seqlens_kv_padded": cu,
        "max_seqlen_q": 12,
        "max_seqlen_kv": 12,
        "pad_between_seqs": False,
    }
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            attention_cp_layout="contiguous", linear_cp_layout="contiguous", virtual_pipeline_model_parallel_size=None
        ),
        dataset=SimpleNamespace(skip_getting_attention_mask_from_dataset=True),
    )
    cp_group = SimpleNamespace(size=lambda: cp_size, rank=lambda: cp_rank)
    loader = Mock(return_value=batch)
    monkeypatch.setattr(dsv4_step, "get_batch_from_iterator", loader)
    monkeypatch.setattr(dsv4_step, "is_pp_first_stage", lambda group: stage == "first")
    monkeypatch.setattr(dsv4_step, "is_pp_last_stage", lambda group: stage == "last")
    monkeypatch.setattr(dsv4_step, "_middle_pp_stage_needs_batch", lambda cfg: False)
    monkeypatch.setattr(
        dsv4_step.parallel_state,
        "get_context_parallel_rank",
        Mock(side_effect=AssertionError("Use the explicit CP process group")),
    )
    result = dsv4_step.get_batch(iter(()), cfg, pg_collection=SimpleNamespace(cp=cp_group, pp=object()))
    local_len = 16 // cp_size
    expected_tokens = tokens[:, cp_rank * local_len : (cp_rank + 1) * local_len]
    assert torch.equal(result[0], expected_tokens)
    assert torch.equal(result[1], expected_tokens + 1)
    assert torch.equal(result[4], expected_tokens)
    assert result[5]["cu_seqlens_q"] is cu
    assert result[5]["cu_seqlens_q_padded"] is cu
    assert result[5]["pad_between_seqs"] is False
    assert "cp_partition_mode" not in result[5]
    assert loader.call_args.kwargs["include_full_batch_fields"] == (stage == "middle")


@pytest.mark.parametrize("layout_field", ["attention_cp_layout", "linear_cp_layout"])
def test_native_cp_rejects_zigzag_before_reading_data(monkeypatch, layout_field):
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            attention_cp_layout="contiguous", linear_cp_layout="contiguous", virtual_pipeline_model_parallel_size=None
        ),
        dataset=SimpleNamespace(),
    )
    assert hasattr(cfg.model, layout_field)
    setattr(cfg.model, layout_field, "zigzag")
    pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: 2), pp=object())
    monkeypatch.setattr(dsv4_step, "is_pp_first_stage", lambda group: True)
    monkeypatch.setattr(dsv4_step, "is_pp_last_stage", lambda group: True)
    loader = Mock(side_effect=AssertionError("Unsupported layout must fail before reading data"))
    monkeypatch.setattr(dsv4_step, "get_batch_from_iterator", loader)
    with pytest.raises(ValueError, match=f"{layout_field}='contiguous'"):
        dsv4_step.get_batch(iter(()), cfg, pg_collection=pg_collection)
    loader.assert_not_called()


def test_native_cp_requires_packed_metadata(monkeypatch):
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            attention_cp_layout="contiguous", linear_cp_layout="contiguous", virtual_pipeline_model_parallel_size=None
        ),
        dataset=SimpleNamespace(),
    )
    cp_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    monkeypatch.setattr(dsv4_step, "is_pp_first_stage", lambda group: True)
    monkeypatch.setattr(dsv4_step, "is_pp_last_stage", lambda group: True)
    monkeypatch.setattr(
        dsv4_step, "get_batch_from_iterator", lambda *args, **kwargs: {"tokens": torch.arange(16).view(1, 16)}
    )
    with pytest.raises(ValueError, match="requires packed THD metadata"):
        dsv4_step.get_batch(iter(()), cfg, pg_collection=SimpleNamespace(cp=cp_group, pp=object()))
