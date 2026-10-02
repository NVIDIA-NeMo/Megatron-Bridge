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

from megatron.bridge.models.deepseek import deepseek_v4_step
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


@pytest.mark.unit
class TestFixedTHDBatches:
    @pytest.mark.parametrize(("pp_rank", "pp_size", "vp_stage"), [(0, 4, 0), (1, 4, 2), (3, 4, 3), (0, 1, None)])
    def test_contiguous_cp_retains_padding_metadata_on_every_pipeline_stage(
        self, monkeypatch, pp_rank, pp_size, vp_stage
    ):
        """First, middle, MTP and PP1 stages see the same global THD boundaries."""
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
        monkeypatch.setattr(deepseek_v4_step.parallel_state, "get_context_parallel_rank", lambda: 1)
        monkeypatch.setattr(deepseek_v4_step, "is_pp_first_stage", lambda group: pp_rank == 0)
        monkeypatch.setattr(deepseek_v4_step, "is_pp_last_stage", lambda group: pp_rank == pp_size - 1)
        monkeypatch.setattr(deepseek_v4_step, "is_vp_first_stage", lambda **kwargs: vp_stage == 0)
        monkeypatch.setattr(deepseek_v4_step, "is_vp_last_stage", lambda **kwargs: vp_stage == 3)
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                virtual_pipeline_model_parallel_size=4 if pp_size > 1 else None,
                pipeline_model_parallel_layout=None,
                cp_partition_mode="contiguous",
            ),
            dataset=SimpleNamespace(
                enable_offline_packing=True,
                offline_packing_specs=SimpleNamespace(packed_sequence_size=8),
                skip_getting_attention_mask_from_dataset=True,
            ),
        )
        batch = {
            "tokens": torch.arange(8).unsqueeze(0),
            "labels": torch.arange(8).unsqueeze(0),
            "position_ids": torch.arange(8).unsqueeze(0),
            "loss_mask": torch.tensor([[1.0] * 7 + [0.0]]),
            "cu_seqlens_q": torch.tensor([[0, 7, 7]], dtype=torch.int32),
            "cu_seqlens_kv": torch.tensor([[0, 7, 7]], dtype=torch.int32),
            "cu_seqlens_q_padded": torch.tensor([[0, 8, 8]], dtype=torch.int32),
            "cu_seqlens_kv_padded": torch.tensor([[0, 8, 8]], dtype=torch.int32),
            "max_seqlen_q": torch.tensor([8], dtype=torch.int32),
            "max_seqlen_kv": torch.tensor([8], dtype=torch.int32),
            "pad_between_seqs": torch.tensor([True]),
            "padding_mask": torch.tensor([[False] * 7 + [True]]),
        }
        result = deepseek_v4_step.get_batch(
            iter([batch]),
            cfg,
            use_mtp=True,
            pg_collection=SimpleNamespace(pp=object(), cp=SimpleNamespace(size=lambda: 2)),
            vp_stage=vp_stage,
        )
        tokens, _, _, _, _, metadata = result
        assert tokens.tolist() == [[4, 5, 6, 7]]
        assert metadata["cu_seqlens_q"].tolist() == [[0, 7, 7]]
        assert metadata["cu_seqlens_q_padded"].tolist() == [[0, 8, 8]]
        assert metadata["max_seqlen_q"].item() == 8
        assert metadata["pad_between_seqs"].item() is True
        assert metadata["padding_mask"].tolist() == [[False, False, False, True]]
        assert metadata["cp_partition_mode"] == "contiguous"

    @pytest.mark.parametrize("supports_routes", [False, True])
    def test_packed_params_prepare_routes_without_allocating_ssm_indices(self, monkeypatch, supports_routes):
        """Development MCore routes are built once before model execution."""
        boundaries = torch.tensor([0, 7, 7], dtype=torch.int32)
        metadata = {"cu_seqlens_q": boundaries, "total_tokens": 4, "cp_partition_mode": "contiguous"}
        params = SimpleNamespace(cp_partition_route=None)
        build = Mock(return_value=params)
        finalize = Mock(return_value=params)
        monkeypatch.setattr(deepseek_v4_step, "get_packed_seq_params", build)
        monkeypatch.setattr(deepseek_v4_step, "finalize_packed_seq_params", finalize if supports_routes else None)

        assert deepseek_v4_step._get_dsv4_packed_seq_params(metadata) is params
        build.assert_called_once_with({"cu_seqlens_q": boundaries, "cp_partition_mode": "contiguous"})
        assert metadata["total_tokens"] == 4
        if supports_routes:
            finalize.assert_called_once_with(params)
        else:
            finalize.assert_not_called()
