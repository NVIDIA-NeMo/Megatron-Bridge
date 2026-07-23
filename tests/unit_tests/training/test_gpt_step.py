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

from functools import partial
from unittest.mock import MagicMock, Mock, patch

import modelopt.torch.distill as mtd
import pytest
import torch
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.training.gpt_step import (
    _create_loss_function_modelopt,
    _cu_seqlens_for_cp_partition,
    _forward_step_common,
    _partition_packed_batch_for_cp,
    get_batch,
    get_packed_seq_params,
)
from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)


class _Iterator:
    def __init__(self, batch):
        self.batch = batch
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return self.batch


class _MockProcessGroup:
    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _MockPGCollection:
    def __init__(self, cp_size=1, pp_rank=0, pp_size=1):
        self.pp = _MockProcessGroup(rank=pp_rank, size=pp_size)
        self._cp_size = cp_size

    @property
    def cp(self):
        return _MockProcessGroup(size=self._cp_size)


class _NoCudaTensor(torch.Tensor):
    def cuda(self, non_blocking=False):  # type: ignore[override]
        return self


def _as_nocuda(tensor):
    return tensor.as_subclass(_NoCudaTensor)


def _make_cfg(
    *,
    enable_offline_packing=False,
    offline_packing_specs=None,
    skip_getting_attention_mask_from_dataset=True,
    pipeline_model_parallel_layout=None,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    mtp_num_layers=0,
):
    cfg = type("Cfg", (), {})()
    cfg.dataset = type(
        "D",
        (),
        {
            "enable_offline_packing": enable_offline_packing,
            "offline_packing_specs": offline_packing_specs,
            "skip_getting_attention_mask_from_dataset": skip_getting_attention_mask_from_dataset,
        },
    )()
    cfg.model = type(
        "M",
        (),
        {
            "pipeline_model_parallel_layout": pipeline_model_parallel_layout,
            "pipeline_model_parallel_size": pipeline_model_parallel_size,
            "virtual_pipeline_model_parallel_size": virtual_pipeline_model_parallel_size,
            "mtp_num_layers": mtp_num_layers,
        },
    )()
    return cfg


def _set_middle_pp_stage(monkeypatch):
    monkeypatch.setattr("megatron.bridge.training.gpt_step.is_pp_first_stage", lambda pg: False)
    monkeypatch.setattr("megatron.bridge.training.gpt_step.is_pp_last_stage", lambda pg: False)


def _set_last_pp_stage(monkeypatch):
    monkeypatch.setattr("megatron.bridge.training.gpt_step.is_pp_first_stage", lambda pg: False)
    monkeypatch.setattr("megatron.bridge.training.gpt_step.is_pp_last_stage", lambda pg: True)


def _set_distributed_initialized(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)


class _NoopTimer:
    def __call__(self, *args, **kwargs):
        return self

    def start(self):
        return None

    def stop(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _RecordingModel:
    def __init__(self, *, vp_stage=None, output=None):
        self.vp_stage = vp_stage
        self.output = output if output is not None else torch.tensor(1.0)
        self.forward_kwargs = None

    def __call__(self, **kwargs):
        self.forward_kwargs = kwargs
        return self.output


class _VpStageWrapper:
    def __init__(self, module):
        self.vp_stage = None
        self.module = module

    def __call__(self, **kwargs):
        return self.module(**kwargs)


class TestGetBatch:
    """Tests for the get_batch helper."""

    @pytest.mark.parametrize("metadata_key", ["cu_seqlens_q", "cu_seqlens"])
    def test_packed_cp_partition_rejects_multiple_physical_thd_rows(self, metadata_key):
        """Packed CP slicing requires one physical THD row after collation."""
        batch = {metadata_key: torch.tensor([[0, 4, 8], [0, 3, 8]], dtype=torch.int32)}

        with pytest.raises(ValueError, match="expect micro-batch size 1"):
            _cu_seqlens_for_cp_partition(batch)

    @staticmethod
    def _patch_cp_indices(monkeypatch, seen_cu_seqlens):
        cp_group = _MockProcessGroup(size=2)

        def fake_get_indices(cu_seqlens, *, total_tokens, cp_group, device):
            seen_cu_seqlens.append(cu_seqlens.clone())
            assert total_tokens == 8
            return torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)

        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_thd_cp_partition_indices",
            fake_get_indices,
        )
        return cp_group

    def test_partition_current_packed_batch_uses_padded_cu_seqlens(self, monkeypatch):
        """Packed CP slicing should use current padded cu-seqlens when present."""
        seen_cu_seqlens = []

        cp_group = self._patch_cp_indices(monkeypatch, seen_cu_seqlens)

        batch = {
            "tokens": torch.arange(8).unsqueeze(0),
            "labels": torch.arange(100, 108).unsqueeze(0),
            "loss_mask": torch.ones(1, 8),
            "position_ids": torch.arange(8).unsqueeze(0),
            "cu_seqlens_q": torch.tensor([0, 3, 8], dtype=torch.int32),
            "cu_seqlens_kv": torch.tensor([0, 3, 8], dtype=torch.int32),
            "cu_seqlens_q_padded": torch.tensor([0, 4, 8], dtype=torch.int32),
            "cu_seqlens_kv_padded": torch.tensor([0, 4, 8], dtype=torch.int32),
            "max_seqlen_q": torch.tensor(4, dtype=torch.int32),
            "max_seqlen_kv": torch.tensor(4, dtype=torch.int32),
        }

        out = _partition_packed_batch_for_cp(batch, cp_group)

        assert len(seen_cu_seqlens) == 1
        assert torch.equal(seen_cu_seqlens[0], torch.tensor([0, 4, 8], dtype=torch.int32))
        assert torch.equal(out["tokens"], torch.tensor([[0, 1, 2, 3]]))
        assert torch.equal(out["labels"], torch.tensor([[100, 101, 102, 103]]))
        assert torch.equal(out["position_ids"], torch.tensor([[0, 1, 2, 3]]))
        assert torch.equal(out["loss_mask"], torch.ones(1, 4))

    def test_partition_packed_batch_trims_negative_sentinel_fallback(self, monkeypatch):
        """Packed CP slicing can trim CPU cu_seqlens without a precomputed argmin."""
        seen_cu_seqlens = []

        cp_group = self._patch_cp_indices(monkeypatch, seen_cu_seqlens)

        batch = {
            "tokens": torch.arange(8).unsqueeze(0),
            "labels": torch.arange(100, 108).unsqueeze(0),
            "loss_mask": torch.ones(1, 8),
            "position_ids": torch.arange(8).unsqueeze(0),
            "cu_seqlens": torch.tensor([[0, 4, 6, 8, -1, -1]], dtype=torch.int32),
            "max_seqlen": torch.tensor([[4]], dtype=torch.int32),
        }

        out = _partition_packed_batch_for_cp(batch, cp_group)

        assert len(seen_cu_seqlens) == 1
        assert torch.equal(seen_cu_seqlens[0], torch.tensor([0, 4, 6, 8], dtype=torch.int32))
        assert torch.equal(out["tokens"], torch.tensor([[0, 1, 2, 3]]))

    def test_partition_packed_batch_no_padding_passthrough(self, monkeypatch):
        """Packed CP slicing should leave unpadded cu_seqlens unchanged."""
        seen_cu_seqlens = []

        cp_group = self._patch_cp_indices(monkeypatch, seen_cu_seqlens)

        batch = {
            "tokens": torch.arange(8).unsqueeze(0),
            "labels": torch.arange(100, 108).unsqueeze(0),
            "loss_mask": torch.ones(1, 8),
            "position_ids": torch.arange(8).unsqueeze(0),
            "cu_seqlens": torch.tensor([[0, 4, 8]], dtype=torch.int32),
            "max_seqlen": torch.tensor([[4]], dtype=torch.int32),
        }

        out = _partition_packed_batch_for_cp(batch, cp_group)

        assert len(seen_cu_seqlens) == 1
        assert torch.equal(seen_cu_seqlens[0], torch.tensor([0, 4, 8], dtype=torch.int32))
        assert torch.equal(out["tokens"], torch.tensor([[0, 1, 2, 3]]))

    def test_partition_packed_batch_skips_none_attention_mask(self, monkeypatch):
        """Packed CP slicing should keep the packed attention mask as None."""
        seen_cu_seqlens = []
        cp_group = self._patch_cp_indices(monkeypatch, seen_cu_seqlens)

        batch = {
            "tokens": torch.arange(8).unsqueeze(0),
            "labels": torch.arange(100, 108).unsqueeze(0),
            "loss_mask": torch.ones(1, 8),
            "attention_mask": None,
            "position_ids": torch.arange(8).unsqueeze(0),
            "cu_seqlens": torch.tensor([[0, 4, 8]], dtype=torch.int32),
            "max_seqlen": torch.tensor([[4]], dtype=torch.int32),
        }

        out = _partition_packed_batch_for_cp(batch, cp_group)

        assert out["attention_mask"] is None
        assert len(seen_cu_seqlens) == 1

    def test_middle_pp_stage_preserves_full_packed_batch(self, monkeypatch):
        """Middle PP stages load full tensors when packed metadata is active."""
        _set_middle_pp_stage(monkeypatch)
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_batch_on_this_cp_rank",
            lambda batch, is_hybrid_cp=False, cp_group=None, hybrid_cp_group_func=None: batch,
        )

        tokens = _as_nocuda(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))
        labels = _as_nocuda(torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]]))
        loss_mask = _as_nocuda(torch.ones(1, 8))
        attention_mask = _as_nocuda(torch.ones(1, 1, 8, 8, dtype=torch.bool))
        position_ids = _as_nocuda(torch.arange(8).unsqueeze(0))
        cu_seqlens_q = _as_nocuda(torch.tensor([0, 3, 8], dtype=torch.int32))
        cu_seqlens_kv = _as_nocuda(torch.tensor([0, 3, 8], dtype=torch.int32))
        cu_seqlens_q_padded = _as_nocuda(torch.tensor([0, 4, 8], dtype=torch.int32))
        cu_seqlens_kv_padded = _as_nocuda(torch.tensor([0, 4, 8], dtype=torch.int32))
        max_seqlen_q = torch.tensor(4, dtype=torch.int32)
        max_seqlen_kv = torch.tensor(4, dtype=torch.int32)
        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_kv": cu_seqlens_kv,
            "cu_seqlens_q_padded": cu_seqlens_q_padded,
            "cu_seqlens_kv_padded": cu_seqlens_kv_padded,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_kv": max_seqlen_kv,
        }

        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            packed_seq_metadata,
        ) = get_batch(
            _Iterator(batch),
            _make_cfg(enable_offline_packing=True, offline_packing_specs=object()),
            use_mtp=False,
            pg_collection=_MockPGCollection(),
        )

        assert torch.equal(tokens, batch["tokens"])
        assert torch.equal(labels, batch["labels"])
        assert torch.equal(loss_mask, batch["loss_mask"])
        assert torch.equal(attention_mask, batch["attention_mask"])
        assert torch.equal(position_ids, batch["position_ids"])
        assert packed_seq_metadata is not None
        assert torch.equal(packed_seq_metadata["cu_seqlens_q"], cu_seqlens_q)
        assert torch.equal(packed_seq_metadata["cu_seqlens_kv"], cu_seqlens_kv)
        assert torch.equal(packed_seq_metadata["cu_seqlens_q_padded"], cu_seqlens_q_padded)
        assert torch.equal(packed_seq_metadata["cu_seqlens_kv_padded"], cu_seqlens_kv_padded)
        assert torch.equal(packed_seq_metadata["max_seqlen_q"], max_seqlen_q)
        assert torch.equal(packed_seq_metadata["max_seqlen_kv"], max_seqlen_kv)
        assert "cu_seqlens" not in packed_seq_metadata
        assert "cu_seqlens_argmin" not in packed_seq_metadata

    def test_middle_pp_stage_keeps_non_packed_fast_path(self, monkeypatch):
        """Middle PP stages without attention metadata keep the all-None fast path."""
        _set_middle_pp_stage(monkeypatch)
        data_iterator = MagicMock()

        result = get_batch(
            data_iterator,
            _make_cfg(offline_packing_specs=None),
            use_mtp=False,
            pg_collection=_MockPGCollection(),
        )

        assert result == (None, None, None, None, None, None)
        data_iterator.__next__.assert_not_called()

    def test_middle_pp_stage_without_mtp_keeps_fast_path_when_mtp_enabled(self, monkeypatch):
        """Global MTP does not force ordinary middle PP stages to load a batch."""
        _set_middle_pp_stage(monkeypatch)
        _set_distributed_initialized(monkeypatch)
        data_iterator = MagicMock()

        result = get_batch(
            data_iterator,
            _make_cfg(
                pipeline_model_parallel_layout=[["embedding", "decoder"], ["decoder"], ["mtp"], ["loss"]],
                pipeline_model_parallel_size=4,
                mtp_num_layers=1,
            ),
            use_mtp=True,
            pg_collection=_MockPGCollection(pp_rank=1, pp_size=4),
        )

        assert result == (None, None, None, None, None, None)
        data_iterator.__next__.assert_not_called()

    def test_standalone_mtp_middle_pp_stage_loads_tokens_and_position_ids(self, monkeypatch):
        """A middle PP stage that owns MTP receives input ids for MCore MTP."""
        _set_middle_pp_stage(monkeypatch)
        _set_distributed_initialized(monkeypatch)
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_batch_on_this_cp_rank",
            lambda batch, is_hybrid_cp=False, cp_group=None, hybrid_cp_group_func=None: batch,
        )
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.parallel_state.get_virtual_pipeline_model_parallel_rank",
            lambda: None,
        )

        tokens = _as_nocuda(torch.tensor([[1, 2, 3, 4]]))
        labels = _as_nocuda(torch.tensor([[2, 3, 4, 5]]))
        loss_mask = _as_nocuda(torch.ones(1, 4))
        position_ids = _as_nocuda(torch.arange(4).unsqueeze(0))
        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": None,
            "position_ids": position_ids,
        }

        (
            out_tokens,
            out_labels,
            out_loss_mask,
            out_attention_mask,
            out_position_ids,
            packed_seq_metadata,
        ) = get_batch(
            _Iterator(batch),
            _make_cfg(
                pipeline_model_parallel_layout=[["embedding", "decoder"], ["decoder"], ["mtp"], ["loss"]],
                pipeline_model_parallel_size=4,
                mtp_num_layers=1,
            ),
            use_mtp=True,
            pg_collection=_MockPGCollection(pp_rank=2, pp_size=4),
        )

        assert torch.equal(out_tokens, tokens)
        assert out_labels is None
        assert out_loss_mask is None
        assert out_attention_mask is None
        assert torch.equal(out_position_ids, position_ids)
        assert packed_seq_metadata is None

    def test_standalone_mtp_loss_stage_skips_mtp_inputs(self, monkeypatch):
        """The loss-only final PP stage does not load token ids for standalone MTP."""
        _set_last_pp_stage(monkeypatch)
        _set_distributed_initialized(monkeypatch)
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_batch_on_this_cp_rank",
            lambda batch, is_hybrid_cp=False, cp_group=None, hybrid_cp_group_func=None: batch,
        )
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.parallel_state.get_virtual_pipeline_model_parallel_rank",
            lambda: None,
        )

        tokens = _as_nocuda(torch.tensor([[1, 2, 3, 4]]))
        labels = _as_nocuda(torch.tensor([[2, 3, 4, 5]]))
        loss_mask = _as_nocuda(torch.ones(1, 4))
        position_ids = _as_nocuda(torch.arange(4).unsqueeze(0))
        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": None,
            "position_ids": position_ids,
        }

        (
            out_tokens,
            out_labels,
            out_loss_mask,
            out_attention_mask,
            out_position_ids,
            *_,
        ) = get_batch(
            _Iterator(batch),
            _make_cfg(
                pipeline_model_parallel_layout=[["embedding", "decoder"], ["decoder"], ["mtp"], ["loss"]],
                pipeline_model_parallel_size=4,
                mtp_num_layers=1,
            ),
            use_mtp=True,
            pg_collection=_MockPGCollection(pp_rank=3, pp_size=4),
        )

        assert out_tokens is None
        assert torch.equal(out_labels, labels)
        assert torch.equal(out_loss_mask, loss_mask)
        assert out_attention_mask is None
        assert out_position_ids is None

    def test_forward_common_uses_model_chunk_vp_stage_for_vpp_stage(self, monkeypatch):
        """Interleaved MTP chunks load tokens using the model chunk VP stage."""
        _set_last_pp_stage(monkeypatch)
        _set_distributed_initialized(monkeypatch)
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_batch_on_this_cp_rank",
            lambda batch, is_hybrid_cp=False, cp_group=None, hybrid_cp_group_func=None: batch,
        )
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.parallel_state.get_virtual_pipeline_model_parallel_rank",
            lambda: None,
        )

        tokens = _as_nocuda(torch.tensor([[1, 2, 3, 4]]))
        labels = _as_nocuda(torch.tensor([[2, 3, 4, 5]]))
        loss_mask = _as_nocuda(torch.ones(1, 4))
        position_ids = _as_nocuda(torch.arange(4).unsqueeze(0))
        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": None,
            "position_ids": position_ids,
        }
        layout = [[] for _ in range(16)]
        layout[15] = ["mtp"]
        inner_model = _RecordingModel(vp_stage=1)
        model = _VpStageWrapper(inner_model)
        state = Mock()
        state.cfg = _make_cfg(
            pipeline_model_parallel_layout=layout,
            pipeline_model_parallel_size=8,
            virtual_pipeline_model_parallel_size=2,
            mtp_num_layers=1,
        )
        state.timers = _NoopTimer()
        state.straggler_timer = _NoopTimer()
        state._flops_seqlen_sum = 0
        config = type(
            "Config",
            (),
            {
                "is_hybrid_model": False,
                "mtp_num_layers": 1,
                "overlap_moe_expert_parallel_comm": False,
            },
        )()

        monkeypatch.setattr("megatron.bridge.training.gpt_step.get_model_config", lambda model: config)
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_pg_collection",
            lambda model: _MockPGCollection(pp_rank=7, pp_size=8),
        )

        output, returned_loss_mask = _forward_step_common(state, _Iterator(batch), model)

        assert torch.equal(output, torch.tensor(1.0))
        assert torch.equal(returned_loss_mask, loss_mask)
        assert inner_model.forward_kwargs is not None
        assert torch.equal(inner_model.forward_kwargs["input_ids"], tokens)
        assert torch.equal(inner_model.forward_kwargs["position_ids"], position_ids)
        assert torch.equal(inner_model.forward_kwargs["labels"], labels)
        assert state._flops_seqlen_sum == 0

    def test_forward_common_uses_model_chunk_vp_stage_instead_of_global_vpp_rank(self, monkeypatch):
        """The model chunk VP stage must override stale global VPP rank state."""
        _set_last_pp_stage(monkeypatch)
        _set_distributed_initialized(monkeypatch)
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_batch_on_this_cp_rank",
            lambda batch, is_hybrid_cp=False, cp_group=None, hybrid_cp_group_func=None: batch,
        )
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.parallel_state.get_virtual_pipeline_model_parallel_rank",
            lambda: 1,
        )

        tokens = _as_nocuda(torch.tensor([[1, 2, 3, 4]]))
        labels = _as_nocuda(torch.tensor([[2, 3, 4, 5]]))
        loss_mask = _as_nocuda(torch.ones(1, 4))
        position_ids = _as_nocuda(torch.arange(4).unsqueeze(0))
        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": None,
            "position_ids": position_ids,
        }
        layout = [[] for _ in range(16)]
        layout[15] = ["mtp"]
        model = _RecordingModel(vp_stage=0)
        state = Mock()
        state.cfg = _make_cfg(
            pipeline_model_parallel_layout=layout,
            pipeline_model_parallel_size=8,
            virtual_pipeline_model_parallel_size=2,
            mtp_num_layers=1,
        )
        state.timers = _NoopTimer()
        state.straggler_timer = _NoopTimer()
        config = type(
            "Config",
            (),
            {
                "is_hybrid_model": False,
                "mtp_num_layers": 1,
                "overlap_moe_expert_parallel_comm": False,
            },
        )()

        monkeypatch.setattr("megatron.bridge.training.gpt_step.get_model_config", lambda model: config)
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_pg_collection",
            lambda model: _MockPGCollection(pp_rank=7, pp_size=8),
        )

        output, returned_loss_mask = _forward_step_common(state, _Iterator(batch), model)

        assert torch.equal(output, torch.tensor(1.0))
        assert returned_loss_mask is None
        assert model.forward_kwargs is not None
        assert model.forward_kwargs["input_ids"] is None
        assert model.forward_kwargs["position_ids"] is None
        assert model.forward_kwargs["labels"] is None

    def test_forward_common_passes_packed_seq_params_on_middle_pp_stage(self, monkeypatch):
        """Forward path must pass packed metadata on middle PP stages."""
        sentinel_packed_seq_params = object()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        labels = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        loss_mask = torch.ones(1, 8)
        position_ids = torch.arange(8).unsqueeze(0)
        packed_seq_metadata = {
            "cu_seqlens_q": torch.tensor([0, 3, 8], dtype=torch.int32),
            "cu_seqlens_kv": torch.tensor([0, 3, 8], dtype=torch.int32),
            "max_seqlen_q": torch.tensor(5, dtype=torch.int32),
            "max_seqlen_kv": torch.tensor(5, dtype=torch.int32),
        }
        model = Mock(return_value=torch.tensor(1.0))
        state = Mock()
        state.cfg = _make_cfg(enable_offline_packing=True, offline_packing_specs=object())
        state.timers = _NoopTimer()
        state.straggler_timer = _NoopTimer()
        config = type(
            "Config",
            (),
            {
                "is_hybrid_model": False,
                "mtp_num_layers": 0,
                "overlap_moe_expert_parallel_comm": False,
            },
        )()

        monkeypatch.setattr("megatron.bridge.training.gpt_step.get_model_config", lambda model: config)
        monkeypatch.setattr("megatron.bridge.training.gpt_step.get_pg_collection", lambda model: _MockPGCollection())
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_batch",
            lambda data_iterator, cfg, use_mtp, *, pg_collection, vp_stage=None: (
                tokens,
                labels,
                loss_mask,
                None,
                position_ids,
                packed_seq_metadata,
            ),
        )
        get_packed_seq_params_mock = Mock(return_value=sentinel_packed_seq_params)
        monkeypatch.setattr("megatron.bridge.training.gpt_step.get_packed_seq_params", get_packed_seq_params_mock)

        output, returned_loss_mask = _forward_step_common(state, _Iterator({}), model)

        assert torch.equal(output, torch.tensor(1.0))
        assert torch.equal(returned_loss_mask, loss_mask)
        model.assert_called_once_with(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            packed_seq_params=sentinel_packed_seq_params,
        )
        get_packed_seq_params_mock.assert_called_once_with(packed_seq_metadata)
        assert "cu_seqlens" not in get_packed_seq_params_mock.call_args.args[0]
        assert "cu_seqlens_argmin" not in get_packed_seq_params_mock.call_args.args[0]


class _FakePackedPartitioner:
    """Record MCore-backed THD partition requests and return a prefix shard."""

    def __init__(self):
        self.seq_lens_seen = []

    def __call__(self, cu_seqlens, *, total_tokens, cp_group, device):
        self.seq_lens_seen.append(total_tokens)
        return torch.arange(total_tokens // cp_group.size(), dtype=torch.long, device=device)


class TestPartitionPackedBatchForCp:
    """Tests for _partition_packed_batch_for_cp (THD/packed context-parallel slicing)."""

    def _run(self, monkeypatch, batch, cp_size=2):
        fake_partitioner = _FakePackedPartitioner()
        monkeypatch.setattr(
            "megatron.bridge.training.gpt_step.get_thd_cp_partition_indices",
            fake_partitioner,
        )
        result = _partition_packed_batch_for_cp(batch, _MockProcessGroup(size=cp_size))
        return result, fake_partitioner

    def test_skips_attention_mask_and_does_not_crash(self, monkeypatch):
        """A degenerate attention_mask must be skipped, not fed to val.size(1) (#4228)."""
        tokens = torch.arange(8, dtype=torch.long).unsqueeze(0)
        labels = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
        loss_mask = torch.ones(1, 8)
        position_ids = torch.arange(8).unsqueeze(0)
        # 1-D placeholder mask that finetuning paths can emit; it has no seq dim at index 1,
        # so the pre-fix code raised IndexError on attention_mask.size(1).
        attention_mask = torch.tensor([1])
        cu_seqlens = torch.tensor([[0, 3, 8, -1]], dtype=torch.int32)
        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
        }

        result, fake_partitioner = self._run(monkeypatch, batch, cp_size=2)

        # attention_mask is passed through untouched (never partitioned).
        assert torch.equal(result["attention_mask"], attention_mask)
        assert fake_partitioner.seq_lens_seen == [8]
        for key in ("tokens", "labels", "loss_mask", "position_ids"):
            assert result[key].size(1) == 4

    def test_none_attention_mask_still_skipped(self, monkeypatch):
        """A None attention_mask is also left as-is and never partitioned."""
        tokens = torch.arange(4, dtype=torch.long).unsqueeze(0)
        batch = {
            "tokens": tokens,
            "attention_mask": None,
            "cu_seqlens": torch.tensor([[0, 4, -1]], dtype=torch.int32),
        }

        result, fake_partitioner = self._run(monkeypatch, batch, cp_size=2)

        assert result["attention_mask"] is None
        assert fake_partitioner.seq_lens_seen == [4]
        assert result["tokens"].size(1) == 2


class TestGetPackedSeqParams:
    """Tests for the get_packed_seq_params function."""

    def test_basic_packed_seq_params_with_max_seqlen(self):
        """Test basic functionality with cu_seqlens and max_seqlen."""
        # Create test batch with packed sequence data
        batch = {
            "cu_seqlens": torch.tensor([[0, 5, 12, 20, -1, -1]], dtype=torch.int32),  # batch size 1
            "max_seqlen": torch.tensor([[15]], dtype=torch.int32),  # batch size 1
        }

        result = get_packed_seq_params(batch)

        # Verify the result is a PackedSeqParams object
        assert isinstance(result, PackedSeqParams)

        # Verify cu_seqlens was squeezed and padding removed (stops at first -1)
        expected_cu_seqlens = torch.tensor([0, 5, 12, 20], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

        # Verify max_seqlen was squeezed
        expected_max_seqlen = torch.tensor(15, dtype=torch.int32)
        assert torch.equal(result.max_seqlen_q, expected_max_seqlen)
        assert torch.equal(result.max_seqlen_kv, expected_max_seqlen)

        # Verify qkv_format is correct
        assert result.qkv_format == "thd"

    def test_packed_seq_params_without_max_seqlen(self):
        """Test functionality when max_seqlen is not provided."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 3, 8, 15, -1]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        # Verify the result is a PackedSeqParams object
        assert isinstance(result, PackedSeqParams)

        # Verify cu_seqlens was processed correctly
        expected_cu_seqlens = torch.tensor([0, 3, 8, 15], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

        # Verify max_seqlen is None when not provided
        assert result.max_seqlen_q is None
        assert result.max_seqlen_kv is None

        # Verify qkv_format is correct
        assert result.qkv_format == "thd"

    def test_packed_seq_params_with_cu_seqlens_argmin(self):
        """Test functionality when cu_seqlens_argmin is provided for performance."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 4, 9, 16, 22, -1, -1, -1]], dtype=torch.int32),
            "cu_seqlens_argmin": torch.tensor(5),  # Index where -1 starts
            "max_seqlen": torch.tensor([[18]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        # Verify the result is a PackedSeqParams object
        assert isinstance(result, PackedSeqParams)

        # Verify cu_seqlens was truncated using cu_seqlens_argmin
        expected_cu_seqlens = torch.tensor([0, 4, 9, 16, 22], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

        # Verify max_seqlen was processed correctly
        expected_max_seqlen = torch.tensor(18, dtype=torch.int32)
        assert torch.equal(result.max_seqlen_q, expected_max_seqlen)
        assert torch.equal(result.max_seqlen_kv, expected_max_seqlen)

    def test_packed_seq_params_with_cu_seqlens_argmin_zero(self):
        """Test edge case when cu_seqlens_argmin is 0."""
        batch = {
            "cu_seqlens": torch.tensor([[-1, -1, -1]], dtype=torch.int32),
            "cu_seqlens_argmin": torch.tensor(0),  # All are padding
        }

        result = get_packed_seq_params(batch)

        # Verify empty cu_seqlens when argmin is 0
        expected_cu_seqlens = torch.empty(0, dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

    def test_packed_seq_params_batch_dimension_removal(self):
        """Test that batch dimensions are properly squeezed."""
        # Test with different batch size dimensions
        batch = {
            "cu_seqlens": torch.tensor([[[0, 6, 12, -1]]], dtype=torch.int32),  # Shape [1, 1, 4]
            "max_seqlen": torch.tensor([[[20]]], dtype=torch.int32),  # Shape [1, 1, 1]
        }

        result = get_packed_seq_params(batch)

        # Verify dimensions were squeezed properly
        expected_cu_seqlens = torch.tensor([0, 6, 12], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)

        expected_max_seqlen = torch.tensor(20, dtype=torch.int32)
        assert torch.equal(result.max_seqlen_q, expected_max_seqlen)

    def test_packed_seq_params_with_different_dtypes(self):
        """Test functionality with different tensor dtypes."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 10, 20, -1]], dtype=torch.int64),  # int64 instead of int32
            "max_seqlen": torch.tensor([[25]], dtype=torch.int64),
        }

        result = get_packed_seq_params(batch)

        # Function should handle different dtypes
        expected_cu_seqlens = torch.tensor([0, 10, 20], dtype=torch.int64)
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)

        expected_max_seqlen = torch.tensor(25, dtype=torch.int64)
        assert torch.equal(result.max_seqlen_q, expected_max_seqlen)

    def test_packed_seq_params_all_fields_match(self):
        """Test that cu_seqlens_q/kv and max_seqlen_q/kv are identical."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 5, 11, 18, -1]], dtype=torch.int32),
            "max_seqlen": torch.tensor([[12]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        # Verify that q and kv parameters are identical (as expected for this function)
        assert torch.equal(result.cu_seqlens_q, result.cu_seqlens_kv)
        assert torch.equal(result.max_seqlen_q, result.max_seqlen_kv)

    def test_packed_seq_params_with_cu_seqlens_unpadded(self):
        """Test functionality with cu_seqlens_unpadded for THD CP support."""
        # Padded cu_seqlens (includes padding for CP divisibility)
        cu_seqlens_padded = torch.tensor([[0, 8, 16, -1, -1]], dtype=torch.int32)
        # Unpadded cu_seqlens (actual sequence boundaries)
        cu_seqlens_unpadded = torch.tensor([[0, 6, 14, -1, -1]], dtype=torch.int32)

        batch = {
            "cu_seqlens": cu_seqlens_padded,
            "cu_seqlens_unpadded": cu_seqlens_unpadded,
            "max_seqlen": torch.tensor([[10]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        # cu_seqlens_q and cu_seqlens_kv should use unpadded values
        expected_unpadded = torch.tensor([0, 6, 14], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_unpadded)
        assert torch.equal(result.cu_seqlens_kv, expected_unpadded)

        # cu_seqlens_q_padded and cu_seqlens_kv_padded should use padded values
        expected_padded = torch.tensor([0, 8, 16], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q_padded, expected_padded)
        assert torch.equal(result.cu_seqlens_kv_padded, expected_padded)

    def test_packed_seq_params_cu_seqlens_unpadded_with_argmin(self):
        """Test cu_seqlens_unpadded processing with argmin hint."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 4, 8, 12, -1, -1]], dtype=torch.int32),
            "cu_seqlens_argmin": torch.tensor(4),  # Index where -1 starts
            "cu_seqlens_unpadded": torch.tensor([[0, 3, 7, 10, -1, -1]], dtype=torch.int32),
            "cu_seqlens_unpadded_argmin": torch.tensor(4),  # Index where -1 starts
        }

        result = get_packed_seq_params(batch)

        # Verify unpadded values are used for q/kv
        expected_unpadded = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q, expected_unpadded)
        assert torch.equal(result.cu_seqlens_kv, expected_unpadded)

        # Verify padded values are set for _padded fields
        expected_padded = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
        assert torch.equal(result.cu_seqlens_q_padded, expected_padded)
        assert torch.equal(result.cu_seqlens_kv_padded, expected_padded)

    def test_packed_seq_params_without_unpadded_fallback(self):
        """Test fallback to cu_seqlens when cu_seqlens_unpadded is not provided."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 5, 10, 15, -1]], dtype=torch.int32),
            "max_seqlen": torch.tensor([[8]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        expected_cu_seqlens = torch.tensor([0, 5, 10, 15], dtype=torch.int32)

        # Without unpadded, q/kv should use padded values
        assert torch.equal(result.cu_seqlens_q, expected_cu_seqlens)
        assert torch.equal(result.cu_seqlens_kv, expected_cu_seqlens)

        # Padded fields should be None when cu_seqlens_unpadded is not provided
        # (to avoid slower TE kernel paths)
        assert result.cu_seqlens_q_padded is None
        assert result.cu_seqlens_kv_padded is None

    def test_packed_seq_params_qkv_format_is_thd(self):
        """Test that qkv_format is always set to 'thd'."""
        batch = {
            "cu_seqlens": torch.tensor([[0, 10, -1]], dtype=torch.int32),
        }

        result = get_packed_seq_params(batch)

        assert result.qkv_format == "thd"


class TestCreateLossFunction:
    """Tests for the _create_loss_function helper function."""

    def test_create_loss_function_both_true(self):
        """Test create_loss_function with both flags as True."""
        loss_mask = torch.tensor([[1.0, 1.0, 0.0]])

        loss_func = _create_loss_function(loss_mask=loss_mask, check_for_nan_in_loss=True, check_for_spiky_loss=True)

        # Verify it returns a partial function
        assert isinstance(loss_func, partial)
        assert loss_func.func.__name__ == "masked_next_token_loss"

        # Verify the partial has correct arguments
        assert torch.equal(loss_func.args[0], loss_mask)
        assert loss_func.keywords["check_for_nan_in_loss"] == True
        assert loss_func.keywords["check_for_spiky_loss"] == True

    def test_create_loss_function_both_false(self):
        """Test _create_loss_function with both flags as False."""
        loss_mask = torch.tensor([[1.0, 0.0, 1.0]])

        loss_func = _create_loss_function(loss_mask=loss_mask, check_for_nan_in_loss=False, check_for_spiky_loss=False)

        # Verify the partial has correct arguments
        assert torch.equal(loss_func.args[0], loss_mask)
        assert loss_func.keywords["check_for_nan_in_loss"] == False
        assert loss_func.keywords["check_for_spiky_loss"] == False

    def test_create_loss_function_mixed_values(self):
        """Test create_loss_function with mixed flag values."""
        loss_mask = torch.tensor([[0.0, 1.0, 1.0]])

        loss_func = _create_loss_function(loss_mask=loss_mask, check_for_nan_in_loss=True, check_for_spiky_loss=False)

        # Verify the partial has correct mixed values
        assert torch.equal(loss_func.args[0], loss_mask)
        assert loss_func.keywords["check_for_nan_in_loss"] == True
        assert loss_func.keywords["check_for_spiky_loss"] == False

    @patch("megatron.bridge.training.losses.masked_next_token_loss")
    def test_create_loss_function_callable(self, mock_loss_func):
        """Test that the created loss function can be called correctly."""
        loss_mask = torch.tensor([[1.0, 1.0, 1.0]])
        output_tensor = torch.tensor([2.5])

        # Mock return value
        expected_result = (torch.tensor(3.0), torch.tensor(2), {"lm loss": torch.tensor([3.0, 2.0])})
        mock_loss_func.return_value = expected_result

        # Create the loss function
        loss_func = _create_loss_function(loss_mask=loss_mask, check_for_nan_in_loss=True, check_for_spiky_loss=False)

        # Call the partial function
        result = loss_func(output_tensor)

        # Verify the underlying function was called correctly
        mock_loss_func.assert_called_once_with(
            loss_mask, output_tensor, check_for_nan_in_loss=True, check_for_spiky_loss=False
        )

        # Verify the result
        assert result == expected_result


class TestCreateLossFunctionModelopt:
    """Tests for the _create_loss_function_modelopt helper function."""

    def test_create_loss_function_modelopt_regular_model(self):
        """Test _create_loss_function_modelopt with a regular (non-DistillationModel) model."""
        loss_mask = torch.tensor([[1.0, 1.0, 0.0]])
        mock_model = Mock()
        mock_unwrapped_model = Mock()

        with patch("megatron.bridge.training.gpt_step.unwrap_model", return_value=mock_unwrapped_model):
            loss_func = _create_loss_function_modelopt(
                loss_mask=loss_mask,
                model=mock_model,
                check_for_nan_in_loss=True,
                check_for_spiky_loss=True,
            )

            # Verify it returns a partial function for masked_next_token_loss (regular loss)
            assert isinstance(loss_func, partial)
            assert loss_func.func.__name__ == "masked_next_token_loss"

            # Verify the partial has correct arguments
            assert torch.equal(loss_func.args[0], loss_mask)
            assert loss_func.keywords["check_for_nan_in_loss"] == True
            assert loss_func.keywords["check_for_spiky_loss"] == True

    def test_create_loss_function_modelopt_distillation_model(self):
        """Test _create_loss_function_modelopt with a DistillationModel."""
        loss_mask = torch.tensor([[1.0, 0.0, 1.0]])
        mock_model = Mock()
        mock_distillation_model = Mock(spec=mtd.DistillationModel)

        with patch("megatron.bridge.training.gpt_step.unwrap_model", return_value=mock_distillation_model):
            loss_func = _create_loss_function_modelopt(
                loss_mask=loss_mask,
                model=mock_model,
                check_for_nan_in_loss=False,
                check_for_spiky_loss=True,
            )

            # Verify it returns a partial function for loss_func_kd (distillation loss)
            assert isinstance(loss_func, partial)
            assert loss_func.func.__name__ == "loss_func_kd"

            # Verify the partial has correct keyword arguments
            assert torch.equal(loss_func.keywords["loss_mask"], loss_mask)
            assert loss_func.keywords["model"] == mock_distillation_model
            assert isinstance(loss_func.keywords["original_loss_fn"], partial)
            # Verify original_loss_fn is correctly configured
            assert loss_func.keywords["original_loss_fn"].func.__name__ == "masked_next_token_loss"
            assert loss_func.keywords["original_loss_fn"].keywords["check_for_nan_in_loss"] == False
            assert loss_func.keywords["original_loss_fn"].keywords["check_for_spiky_loss"] == True

    def test_create_loss_function_modelopt_both_flags_false(self):
        """Test _create_loss_function_modelopt with both flags as False."""
        loss_mask = torch.tensor([[0.0, 1.0, 1.0]])
        mock_model = Mock()
        mock_unwrapped_model = Mock()

        with patch("megatron.bridge.training.gpt_step.unwrap_model", return_value=mock_unwrapped_model):
            loss_func = _create_loss_function_modelopt(
                loss_mask=loss_mask,
                model=mock_model,
                check_for_nan_in_loss=False,
                check_for_spiky_loss=False,
            )

            # Verify the partial has correct arguments
            assert torch.equal(loss_func.args[0], loss_mask)
            assert loss_func.keywords["check_for_nan_in_loss"] == False
            assert loss_func.keywords["check_for_spiky_loss"] == False
