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
from unittest.mock import Mock

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
        "sequence_packing_scheduler": "dp_balanced",
        "virtual_pipeline_model_parallel_size": None,
        "pipeline_model_parallel_layout": None,
        "mtp_num_layers": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


# Packed batch fetch signatures of the two Megatron-Core generations Bridge pins: the
# main pin takes no model config, newer releases read CP partitioning options from it.
def _get_batch_without_config(data_iterator, vpp_size=None, mtp_on_this_rank=False, vp_stage=None, pg_collection=None):
    return ("tokens", "labels", "loss_mask", None, "position_ids", SimpleNamespace(cp_group=None), "padding_mask")


def _get_batch_with_config(
    data_iterator, vpp_size=None, mtp_on_this_rank=False, vp_stage=None, pg_collection=None, config=None
):
    return ("tokens", "labels", "loss_mask", None, "position_ids", SimpleNamespace(cp_group=config), "padding_mask")


def _wrap_data_iterator(data_iterator, config, num_microbatches, pg_collection=None):
    return None


def _install_fake_data_schedule(monkeypatch, *, schedulers, get_batch=_get_batch_without_config, wrap=None):
    module = types.ModuleType("megatron.core.datasets.data_schedule")
    module.PackingSchedulerEnum = enum.Enum("PackingSchedulerEnum", {name.upper(): name for name in schedulers})
    module.scheduler_map = {member: object for member in module.PackingSchedulerEnum}
    module.get_batch_on_this_rank_for_sequence_packing = get_batch
    module.wrap_data_iterator = wrap or _wrap_data_iterator
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", module)


@pytest.fixture
def no_mtp(monkeypatch):
    monkeypatch.setattr(
        "megatron.core.transformer.multi_token_prediction.mtp_on_this_rank", lambda *a, **k: False, raising=False
    )


def test_enabled_flag():
    assert global_batch_packing.global_batch_packing_enabled(_model())
    assert not global_batch_packing.global_batch_packing_enabled(_model(sequence_packing_scheduler=None))
    # A mock config answers every attribute; only a scheduler name enables packing.
    assert not global_batch_packing.global_batch_packing_enabled(Mock())


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


@pytest.mark.parametrize("get_batch", [_get_batch_without_config, _get_batch_with_config])
def test_get_batch_passes_only_arguments_the_pinned_fetch_accepts(monkeypatch, no_mtp, get_batch):
    model = _model()
    monkeypatch.setattr(global_batch_packing, "_scheduler_api", lambda: (None, get_batch))
    monkeypatch.setattr(global_batch_packing, "finalize_packed_seq_params", lambda psp, pg: psp)
    tokens, labels, loss_mask, attention_mask, position_ids, params, padding_mask = (
        global_batch_packing.get_batch_for_global_batch_packing(
            "iterator", model, pg_collection=_pg(cp=2), vp_stage=None
        )
    )
    assert (tokens, labels, loss_mask, attention_mask, position_ids, padding_mask) == (
        "tokens",
        "labels",
        "loss_mask",
        None,
        "position_ids",
        "padding_mask",
    )
    # The fake echoes the config it received through cp_group.
    assert params.cp_group is (model if get_batch is _get_batch_with_config else None)


def test_get_batch_passes_the_iterator_only_on_tp_rank_zero(monkeypatch, no_mtp):
    seen = []

    def fake_get_batch(data_iterator, vpp_size=None, mtp_on_this_rank=False, vp_stage=None, pg_collection=None):
        seen.append(data_iterator)
        return (None, None, None, None, None, None, None)

    monkeypatch.setattr(global_batch_packing, "_scheduler_api", lambda: (None, fake_get_batch))
    for tp_rank in (0, 1):
        global_batch_packing.get_batch_for_global_batch_packing(
            "iterator", _model(), pg_collection=_pg(tp=2, tp_rank=tp_rank), vp_stage=None
        )
    assert seen == ["iterator", None]


def test_supported_main_like_pin_also_runs_the_first_fetch(monkeypatch, no_mtp):
    # Regression: validation accepted dp_balanced on a pin whose batch fetch rejects extra
    # arguments, and the first training step then failed with a TypeError.
    _install_fake_data_schedule(monkeypatch, schedulers=["dp_balanced"], get_batch=_get_batch_without_config)
    monkeypatch.setattr(global_batch_packing, "finalize_packed_seq_params", lambda psp, pg: psp)
    assert global_batch_packing.probe_global_batch_packing_support("dp_balanced") is None
    out = global_batch_packing.get_batch_for_global_batch_packing(
        "iterator", _model(), pg_collection=_pg(cp=2), vp_stage=None
    )
    assert out[0] == "tokens"


@pytest.mark.parametrize("has_routes", [True, False])
def test_finalize_binds_the_cp_group_and_prebuilds_routes_when_available(monkeypatch, has_routes):
    import megatron.core.packed_seq_params as packed_seq_params_module

    cp_group = _Group(2)
    params = SimpleNamespace(cp_group=None, local_cp_size=None)
    prebuilt = []
    monkeypatch.setattr(packed_seq_params_module, "resolve_cp_group", lambda static, psp: static)
    monkeypatch.setattr(
        global_batch_packing,
        "_thd_cp_route_prebuilder",
        lambda: (lambda psp, group: prebuilt.append((psp, group))) if has_routes else None,
    )
    finalized = global_batch_packing.finalize_packed_seq_params(params, SimpleNamespace(cp=cp_group))
    assert finalized is params and params.cp_group is cp_group
    assert prebuilt == ([(params, cp_group)] if has_routes else [])
    assert global_batch_packing.finalize_packed_seq_params(None, SimpleNamespace(cp=cp_group)) is None


def test_probe_reports_what_the_pinned_scheduler_is_missing(monkeypatch):
    _install_fake_data_schedule(monkeypatch, schedulers=["dp_balanced"])
    assert global_batch_packing.probe_global_batch_packing_support("dp_balanced") is None
    message = global_batch_packing.probe_global_batch_packing_support("custom_scheduler")
    assert message is not None and "custom_scheduler" in message and "dp_balanced" in message

    def wrap_without_pg_collection(data_iterator, config, num_microbatches):
        return None

    _install_fake_data_schedule(monkeypatch, schedulers=["dp_balanced"], wrap=wrap_without_pg_collection)
    message = global_batch_packing.probe_global_batch_packing_support("dp_balanced")
    assert message is not None and "ProcessGroupCollection" in message


def test_probe_reports_a_release_without_the_scheduler(monkeypatch):
    monkeypatch.setitem(sys.modules, "megatron.core.datasets.data_schedule", None)
    message = global_batch_packing.probe_global_batch_packing_support("dp_balanced")
    assert message is not None and "data_schedule" in message
