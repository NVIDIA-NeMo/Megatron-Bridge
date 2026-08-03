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

import pytest
import torch

from megatron.bridge.models.deepseek.deepseek_v4_step import _partition_packed_batch_contiguous


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

    def test_legacy_path_extracts_cu_seqlens_and_cp_partition_mode(self):
        from megatron.bridge.models.deepseek.deepseek_v4_step import _packed_metadata_for_forward

        batch = {
            "cu_seqlens": torch.tensor([[0, 4, 8]], dtype=torch.int32),
            "max_seqlen": torch.tensor([[8]]),
            "cp_partition_mode": "contiguous",
            "total_tokens": 8,
        }
        meta = _packed_metadata_for_forward(batch)
        assert meta is not None
        assert meta["cp_partition_mode"] == "contiguous"
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
        assert meta.get("cp_partition_mode") == "contiguous"
        assert "cu_seqlens_q" in meta
