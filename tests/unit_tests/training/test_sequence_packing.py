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

"""Unit tests for the online sequence packing / dynamic CP glue."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.training import sequence_packing


pytestmark = pytest.mark.unit


class _Group:
    def __init__(self, size: int, rank: int = 0) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


def _pg(*, dp: int = 1, cp: int = 1, tp: int = 1, tp_rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(dp=_Group(dp), cp=_Group(cp), tp=_Group(tp, tp_rank), pp=_Group(1))


def _model(**overrides) -> SimpleNamespace:
    fields = {
        "sequence_packing_scheduler": "default_dynamic_cp",
        "dynamic_context_parallel": True,
        "min_dynamic_context_parallel_size": 1,
        "max_seqlen_per_dp_cp_rank": 8192,
        "calculate_per_token_loss": True,
        "cuda_graph_impl": "none",
        "enable_cuda_graph": False,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 4,
        "seq_length": 32768,
        "is_hybrid_model": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _cfg(*, dp_size: int | None = 2, **model_overrides) -> SimpleNamespace:
    return SimpleNamespace(
        model=_model(**model_overrides),
        train=SimpleNamespace(micro_batch_size=1),
        dataset=SimpleNamespace(dataloader_type="single", sequence_padding_multiple=None, seq_length=32768),
        ddp=SimpleNamespace(average_in_collective=False, use_megatron_fsdp=False),
        dist=SimpleNamespace(use_decentralized_pg=False),
        data_parallel_size=dp_size,
    )


def test_identity_collate_keeps_samples_separate():
    samples = [{"tokens": torch.zeros(3)}, {"tokens": torch.zeros(5)}]
    collated = sequence_packing.identity_collate(samples)
    assert collated == samples and collated is not samples


@pytest.mark.parametrize(
    ("dynamic_cp", "dp", "cp", "tp", "sp", "expected"),
    [
        (False, 2, 1, 1, False, 1),  # static, no CP: nothing to align
        (False, 2, 4, 1, False, 8),  # static CP4: 2 * cp
        (True, 2, 4, 1, False, 16),  # dynamic, pool 8: 2 * 8 covers 1/2/4/8
        (True, 3, 2, 1, False, 24),  # dynamic, pool 6: groups 1/2/4 and the full pool -> lcm(8, 12)
        (True, 2, 4, 2, True, 32),  # sequence parallel multiplies by TP
    ],
)
def test_padding_multiple(dynamic_cp, dp, cp, tp, sp, expected):
    assert (
        sequence_packing.padding_multiple(
            dynamic_cp=dynamic_cp, dp_size=dp, cp_size=cp, tp_size=tp, sequence_parallel=sp
        )
        == expected
    )
    model = _model(dynamic_context_parallel=dynamic_cp, sequence_parallel=sp)
    assert sequence_packing.sequence_padding_multiple(model, _pg(dp=dp, cp=cp, tp=tp)) == expected


def test_scheduled_num_microbatches_falls_back_to_default():
    sequence_packing.set_scheduled_num_microbatches(None)
    assert sequence_packing.scheduled_num_microbatches(lambda: 16) == 16
    sequence_packing.set_scheduled_num_microbatches(83)
    try:
        assert sequence_packing.scheduled_num_microbatches(lambda: 16) == 83
    finally:
        sequence_packing.set_scheduled_num_microbatches(None)


def test_wrap_passes_iterator_only_on_tp_rank_zero(monkeypatch):
    calls = []

    def fake_wrap(data_iterator, config, num_microbatches, pg_collection=None):
        calls.append(data_iterator)
        return iter([]), 3, 100.0, 1000.0

    monkeypatch.setattr(sequence_packing, "_scheduler_api", lambda: (fake_wrap, None))
    sentinel = object()
    result = sequence_packing.wrap_data_iterator_for_sequence_packing(sentinel, _model(), 4, _pg(tp=2, tp_rank=0))
    assert result[1:] == (3, 100.0, 1000.0)
    sequence_packing.wrap_data_iterator_for_sequence_packing(sentinel, _model(), 4, _pg(tp=2, tp_rank=1))
    assert calls == [sentinel, None]


def test_get_batch_forwards_dynamic_cp_flag_and_finalizes(monkeypatch):
    seen = {}
    params = SimpleNamespace(cp_group=None)

    def fake_get_batch(data_iterator, **kwargs):
        seen.update(kwargs)
        seen["iterator"] = data_iterator
        return ("tokens", "labels", "loss_mask", None, "position_ids", params, "padding_mask")

    monkeypatch.setattr(sequence_packing, "_scheduler_api", lambda: (None, fake_get_batch))
    monkeypatch.setattr(sequence_packing, "finalize_packed_seq_params", lambda psp, pg: psp)
    monkeypatch.setattr(
        "megatron.core.transformer.multi_token_prediction.mtp_on_this_rank", lambda *a, **k: False, raising=False
    )
    out = sequence_packing.get_batch_for_sequence_packing("iterator", _model(), pg_collection=_pg(cp=4), vp_stage=None)
    assert out == ("tokens", "labels", "loss_mask", None, "position_ids", params, "padding_mask")
    assert seen["dynamic_cp"] is True and seen["iterator"] == "iterator" and seen["vp_stage"] is None


def test_validate_noop_without_scheduler():
    cfg = _cfg(sequence_packing_scheduler=None, dynamic_context_parallel=False)
    cfg.train.micro_batch_size = 4  # would be rejected if packing were on
    sequence_packing.validate_sequence_packing_config(cfg)


def test_validate_accepts_recipe_shaped_config_and_derives_padding_multiple():
    cfg = _cfg()
    sequence_packing.validate_sequence_packing_config(cfg)
    # dp 2 x cp 4 = pool 8 -> 2 * 8
    assert cfg.dataset.sequence_padding_multiple == 16


def test_validate_static_scheduler_uses_static_multiple():
    cfg = _cfg(sequence_packing_scheduler="dp_balanced", dynamic_context_parallel=False)
    sequence_packing.validate_sequence_packing_config(cfg)
    assert cfg.dataset.sequence_padding_multiple == 8


def test_validate_skips_layout_checks_without_data_parallel_size():
    cfg = _cfg(dp_size=None)
    sequence_packing.validate_sequence_packing_config(cfg)
    assert cfg.dataset.sequence_padding_multiple is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: setattr(c.train, "micro_batch_size", 2), "micro_batch_size"),
        (lambda c: setattr(c.dataset, "dataloader_type", "batch"), "dataloader_type"),
        (lambda c: setattr(c.model, "calculate_per_token_loss", False), "calculate_per_token_loss"),
        (lambda c: setattr(c.ddp, "average_in_collective", True), "average_in_collective"),
        (lambda c: setattr(c.model, "max_seqlen_per_dp_cp_rank", None), "max_seqlen_per_dp_cp_rank"),
        (lambda c: setattr(c.model, "sequence_packing_scheduler", "dp_balanced"), "dynamic-CP scheduler"),
        (lambda c: setattr(c.model, "sequence_packing_scheduler", None), "dynamic-CP scheduler"),
        (lambda c: setattr(c.model, "min_dynamic_context_parallel_size", 3), "power of two"),
        (lambda c: setattr(c.model, "min_dynamic_context_parallel_size", 16), "power of two"),
        (lambda c: setattr(c.model, "cuda_graph_impl", "transformer_engine"), "CUDA graphs"),
        (lambda c: setattr(c.ddp, "use_megatron_fsdp", True), "FSDP"),
        (lambda c: setattr(c.dist, "use_decentralized_pg", True), "use_decentralized_pg"),
        (lambda c: setattr(c.model, "virtual_pipeline_model_parallel_size", 2), "virtual pipeline"),
        (lambda c: setattr(c.model, "is_hybrid_model", True), "hybrid"),
        (lambda c: setattr(c.model, "max_seqlen_per_dp_cp_rank", 2048), "Bin capacity"),
        (lambda c: setattr(c.dataset, "sequence_padding_multiple", 12), "not a multiple"),
        (lambda c: setattr(c.dataset, "seq_length", 32760), "multiple of the padding multiple"),
        (lambda c: setattr(c.dataset, "seq_length", 65536), "exceeds model.seq_length"),
        (lambda c: delattr(c.dataset, "sequence_padding_multiple"), "unpacked per-sample dicts"),
    ],
)
def test_validate_rejects_incompatible_settings(mutate, message):
    cfg = _cfg()
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        sequence_packing.validate_sequence_packing_config(cfg)


def test_validate_static_scheduler_allows_cuda_graphs_with_static_thd_shapes():
    cfg = _cfg(
        sequence_packing_scheduler="dp_balanced",
        dynamic_context_parallel=False,
        cuda_graph_impl="transformer_engine",
        thd_max_packed_sequences=64,
        pad_packed_seq_alignment="max",
    )
    sequence_packing.validate_sequence_packing_config(cfg)


def test_validate_rejects_main_pin_without_dynamic_cp_fields():
    cfg = _cfg()
    del cfg.model.dynamic_context_parallel
    cfg.model.sequence_packing_scheduler = "dp_balanced"
    with pytest.raises(ValueError, match="switch_mcore.sh dev"):
        sequence_packing.validate_sequence_packing_config(cfg)
