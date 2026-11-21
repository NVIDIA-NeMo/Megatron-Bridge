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

from dataclasses import make_dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import megatron.bridge.models.deepseek.deepseek_v4_step as dsv4_step
from megatron.bridge.models.deepseek.deepseek_v4_step import _partition_packed_batch_contiguous
from megatron.bridge.perf_recipes.deepseek.gb300._deepseek_v4_compat import FixedTHDMockDatasetConfig


pytestmark = pytest.mark.unit


def _patch_legacy_cp_field(monkeypatch, supported):
    declared_fields = [("cp_partition_mode", str, "zigzag")] if supported else []
    namespace = {} if supported else {"cp_partition_mode": property(lambda self: "zigzag")}
    params_type = make_dataclass("TestPackedSeqParams", declared_fields, namespace=namespace)
    monkeypatch.setattr(dsv4_step, "PackedSeqParams", params_type)


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

    def test_slices_padding_mask_and_preserves_attention_mask(self, monkeypatch):
        tokens = torch.arange(6).view(1, 6)
        padding_mask = torch.tensor([[False, False, True, False, True, True]])
        attention_mask = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
        batch = _make_batch(
            tokens=tokens,
            cu_seqlens=torch.tensor([[0, 3, 6]], dtype=torch.int32),
            padding_mask=padding_mask,
            attention_mask=attention_mask,
        )
        result = self._run(monkeypatch, batch, cp_rank=1)
        assert torch.equal(result["padding_mask"], padding_mask[:, 3:])
        assert result["attention_mask"] is attention_mask
        # Total length 6 is divisible by CP2, but not by 2*CP: no zigzag rule.
        assert torch.equal(result["tokens"], tokens[:, 3:])

    def test_rejects_malformed_token_fields(self, monkeypatch):
        for key in ("tokens", "labels", "loss_mask", "position_ids", "padding_mask"):
            batch = _make_batch(tokens=torch.arange(8).view(1, 8))
            batch[key] = torch.zeros(2, 4)
            with pytest.raises(ValueError, match=r"\[1, tokens\]"):
                self._run(monkeypatch, batch)

    def test_rejects_unknown_populated_fields(self, monkeypatch):
        batch = _make_batch(tokens=torch.arange(8).view(1, 8), unknown_scalar=123)
        with pytest.raises(ValueError, match="Unsupported populated packed batch fields"):
            self._run(monkeypatch, batch)


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


@pytest.mark.parametrize("supports_cp_mode", [False, True])
@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("stage", ["first", "middle", "last"])
def test_native_layout_dispatch_preserves_packed_boundaries(monkeypatch, cp_size, stage, supports_cp_mode):
    """Use the explicit CP rank and carry metadata through intermediate PP stages."""
    _patch_legacy_cp_field(monkeypatch, supports_cp_mode)
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
    if supports_cp_mode:
        assert result[5]["cp_partition_mode"] == "contiguous"
    else:
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


@pytest.mark.unit
class TestFixedTHDBatches:
    @pytest.mark.parametrize("supports_cp_mode", [False, True])
    @pytest.mark.parametrize(("pp_rank", "pp_size", "vp_stage"), [(0, 4, 0), (1, 4, 2), (3, 4, 3), (0, 1, None)])
    def test_contiguous_cp_retains_padding_metadata_on_every_pipeline_stage(
        self, monkeypatch, pp_rank, pp_size, vp_stage, supports_cp_mode
    ):
        """First, middle, MTP and PP1 stages see the same global THD boundaries."""
        _patch_legacy_cp_field(monkeypatch, supports_cp_mode)
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
        monkeypatch.setattr(dsv4_step.parallel_state, "get_context_parallel_rank", lambda: 1)
        monkeypatch.setattr(dsv4_step, "is_pp_first_stage", lambda group: pp_rank == 0)
        monkeypatch.setattr(dsv4_step, "is_pp_last_stage", lambda group: pp_rank == pp_size - 1)
        monkeypatch.setattr(dsv4_step, "is_vp_first_stage", lambda **kwargs: vp_stage == 0)
        monkeypatch.setattr(dsv4_step, "is_vp_last_stage", lambda **kwargs: vp_stage == 3)
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                virtual_pipeline_model_parallel_size=4 if pp_size > 1 else None,
                pipeline_model_parallel_layout=None,
                cp_partition_mode="contiguous",
                attention_cp_layout="contiguous",
                linear_cp_layout="contiguous",
            ),
            dataset=FixedTHDMockDatasetConfig(seq_length=8, vocab_size=256, seed=1234, metadata_capacity=2),
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
        result = dsv4_step.get_batch(
            iter([batch]),
            cfg,
            use_mtp=True,
            pg_collection=SimpleNamespace(pp=object(), cp=SimpleNamespace(size=lambda: 2, rank=lambda: 1)),
            vp_stage=vp_stage,
        )
        tokens, _, _, _, _, metadata = result
        assert tokens.tolist() == [[4, 5, 6, 7]]
        assert metadata["cu_seqlens_q"].tolist() == [[0, 7, 7]]
        assert metadata["cu_seqlens_q_padded"].tolist() == [[0, 8, 8]]
        assert metadata["max_seqlen_q"].item() == 8
        assert metadata["pad_between_seqs"].item() is True
        assert metadata["padding_mask"].tolist() == [[False, False, False, True]]
        if supports_cp_mode:
            assert metadata["cp_partition_mode"] == "contiguous"
        else:
            assert "cp_partition_mode" not in metadata

    @pytest.mark.parametrize("supports_routes", [False, True])
    def test_packed_params_prepare_routes_without_allocating_ssm_indices(self, monkeypatch, supports_routes):
        """Development MCore routes are built once before model execution."""
        boundaries = torch.tensor([0, 7, 7], dtype=torch.int32)
        metadata = {"cu_seqlens_q": boundaries, "total_tokens": 4, "cp_partition_mode": "contiguous"}
        params = SimpleNamespace(cp_partition_route=None)
        build = Mock(return_value=params)
        routed = SimpleNamespace(
            cp_partition_route=object(), thd_cp_host_cu_seqlens_q=(0, 7, 7), thd_cp_host_cu_seqlens_kv=(0, 7, 7)
        )
        finalize = Mock(return_value=routed)
        monkeypatch.setattr(dsv4_step, "get_packed_seq_params", build)
        monkeypatch.setattr(dsv4_step, "finalize_packed_seq_params", finalize if supports_routes else None)

        result = dsv4_step._get_dsv4_packed_seq_params(metadata)
        assert result is (routed if supports_routes else params)
        assert result.cp_partition_route is (routed.cp_partition_route if supports_routes else None)
        build.assert_called_once_with({"cu_seqlens_q": boundaries, "cp_partition_mode": "contiguous"})
        assert metadata["total_tokens"] == 4
        if supports_routes:
            assert result.thd_cp_host_cu_seqlens_q == result.thd_cp_host_cu_seqlens_kv == (0, 7, 7)
            finalize.assert_called_once_with(params)
        else:
            finalize.assert_not_called()


@pytest.mark.parametrize("boundary_key", ["cu_seqlens", "cu_seqlens_q"])
def test_padding_mask_requires_explicit_fixed_thd_opt_in(boundary_key):
    boundaries = torch.tensor([[0, 7, 8]], dtype=torch.int32)
    mask = torch.tensor([[False] * 7 + [True]])
    batch = {boundary_key: boundaries, "padding_mask": mask, "cp_partition_mode": "contiguous"}
    native = dsv4_step._packed_metadata_for_forward(batch)
    fixed_thd = dsv4_step._packed_metadata_for_forward(batch, include_padding_mask=True)
    assert native[boundary_key] is fixed_thd[boundary_key] is boundaries
    assert "padding_mask" not in native
    assert fixed_thd["padding_mask"] is mask
    assert "cp_partition_mode" not in native and "cp_partition_mode" not in fixed_thd
