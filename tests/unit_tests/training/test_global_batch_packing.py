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

"""Unit tests for the global-batch packing training-loop glue."""

import enum
import sys
import types
from types import SimpleNamespace

import pytest

from megatron.bridge.training import global_batch_packing


pytestmark = pytest.mark.unit


class _Group:
    def __init__(self, size: int, rank: int = 0) -> None:
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


def _pg(*, tp: int = 1, tp_rank: int = 0, cp: int = 1) -> SimpleNamespace:
    return SimpleNamespace(dp=_Group(1), cp=_Group(cp), tp=_Group(tp, tp_rank), pp=_Group(1))


def _model(**overrides) -> SimpleNamespace:
    fields = {
        "sequence_packing_scheduler": "default_dynamic_cp",
        "dynamic_context_parallel": True,
        "virtual_pipeline_model_parallel_size": None,
        "pipeline_model_parallel_layout": None,
        "mtp_num_layers": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_enabled_flags():
    assert global_batch_packing.global_batch_packing_enabled(_model())
    assert not global_batch_packing.global_batch_packing_enabled(_model(sequence_packing_scheduler=None))
    assert global_batch_packing.dynamic_context_parallel_enabled(_model())
    assert not global_batch_packing.dynamic_context_parallel_enabled(_model(dynamic_context_parallel=False))


def test_wrap_passes_iterator_only_on_tp_rank_zero(monkeypatch):
    calls = []

    def fake_wrap(data_iterator, config, num_microbatches, pg_collection=None):
        calls.append(data_iterator)
        return None if data_iterator is None else iter([]), 3, 100.0, 1000.0

    monkeypatch.setattr(global_batch_packing, "_scheduler_api", lambda: (fake_wrap, None))
    sentinel = object()
    packed, count, seqlen_sum, seqlen_sq = global_batch_packing.wrap_data_iterator_for_global_batch_packing(
        sentinel, _model(), 4, _pg(tp=2, tp_rank=0)
    )
    assert (count, seqlen_sum, seqlen_sq) == (3, 100.0, 1000.0) and packed is not None
    packed_other, *_ = global_batch_packing.wrap_data_iterator_for_global_batch_packing(
        sentinel, _model(), 4, _pg(tp=2, tp_rank=1)
    )
    # The scheduler returns None on TP ranks > 0: callers must not use `is None` as "not wrapped yet".
    assert packed_other is None
    assert calls == [sentinel, None]


def test_get_batch_forwards_dynamic_cp_flag_and_finalizes(monkeypatch):
    seen = {}
    params = SimpleNamespace(cp_group=None)

    def fake_get_batch(data_iterator, **kwargs):
        seen.update(kwargs)
        seen["iterator"] = data_iterator
        return ("tokens", "labels", "loss_mask", None, "position_ids", params, "padding_mask")

    monkeypatch.setattr(global_batch_packing, "_scheduler_api", lambda: (None, fake_get_batch))
    monkeypatch.setattr(global_batch_packing, "finalize_packed_seq_params", lambda psp, pg: psp)
    monkeypatch.setattr(
        "megatron.core.transformer.multi_token_prediction.mtp_on_this_rank", lambda *a, **k: False, raising=False
    )
    out = global_batch_packing.get_batch_for_global_batch_packing(
        "iterator", _model(), pg_collection=_pg(cp=4), vp_stage=None
    )
    assert out == ("tokens", "labels", "loss_mask", None, "position_ids", params, "padding_mask")
    assert seen["dynamic_cp"] is True and seen["iterator"] == "iterator" and seen["vp_stage"] is None


def _install_fake_data_schedule(monkeypatch, *, schedulers, dynamic_kwargs: bool):
    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.PackingSchedulerEnum = enum.Enum("PackingSchedulerEnum", {name.upper(): name for name in schedulers})
    module.scheduler_map = {member: object for member in module.PackingSchedulerEnum}
    if dynamic_kwargs:

        def get_batch(
            data_iterator,
            vpp_size=None,
            mtp_on_this_rank=False,
            vp_stage=None,
            dynamic_cp=False,
            pg_collection=None,
            config=None,
        ):
            return None

    else:

        def get_batch(data_iterator, vpp_size=None, mtp_on_this_rank=False, vp_stage=None, pg_collection=None):
            return None

    def wrap_data_iterator(data_iterator, config, num_microbatches, pg_collection=None):
        return None

    module.get_batch_on_this_rank_for_sequence_packing = get_batch
    module.wrap_data_iterator = wrap_data_iterator
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)


def test_probe_reports_missing_scheduler_or_dynamic_cp_support(monkeypatch):
    _install_fake_data_schedule(monkeypatch, schedulers=["dp_balanced"], dynamic_kwargs=False)
    assert global_batch_packing.probe_global_batch_packing_support("dp_balanced", dynamic_cp=False) is None
    message = global_batch_packing.probe_global_batch_packing_support("default_dynamic_cp", dynamic_cp=True)
    assert message is not None and "default_dynamic_cp" in message and "switch_mcore.sh dev" in message

    _install_fake_data_schedule(monkeypatch, schedulers=["dp_balanced", "default_dynamic_cp"], dynamic_kwargs=False)
    message = global_batch_packing.probe_global_batch_packing_support("default_dynamic_cp", dynamic_cp=True)
    assert message is not None and "dynamic context parallel" in message

    _install_fake_data_schedule(monkeypatch, schedulers=["dp_balanced", "default_dynamic_cp"], dynamic_kwargs=True)
    assert global_batch_packing.probe_global_batch_packing_support("default_dynamic_cp", dynamic_cp=True) is None
