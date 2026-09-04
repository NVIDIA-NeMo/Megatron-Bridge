# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from types import SimpleNamespace

import torch

from megatron.bridge.data.dynamic_context_parallel import (
    DynamicCPBatchPolicy,
    _materialize_microbatch,
    _runtime_cp_group_histogram,
    _unpack_thd_global_batch,
    prepare_dynamic_cp_batch,
)


def _packed_global_batch() -> dict[str, torch.Tensor | int | None]:
    tokens = torch.zeros((1, 16), dtype=torch.long)
    labels = torch.zeros_like(tokens)
    loss_mask = torch.zeros_like(tokens)
    position_ids = torch.zeros_like(tokens)
    tokens[0, :3] = torch.tensor([10, 11, 12])
    tokens[0, 4:9] = torch.tensor([20, 21, 22, 23, 24])
    labels.copy_(tokens + 100)
    loss_mask[0, :3] = 1
    loss_mask[0, 4:9] = 1
    position_ids[0, :3] = torch.arange(3)
    position_ids[0, 4:9] = torch.arange(5)
    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": None,
        "cu_seqlens_q": torch.tensor([[0, 3, 8, 8]], dtype=torch.int32),
        "cu_seqlens_kv": torch.tensor([[0, 3, 8, 8]], dtype=torch.int32),
        "cu_seqlens_q_padded": torch.tensor([[0, 4, 12, 16]], dtype=torch.int32),
        "cu_seqlens_kv_padded": torch.tensor([[0, 4, 12, 16]], dtype=torch.int32),
        "max_seqlen_q": 8,
        "max_seqlen_kv": 8,
    }


def test_unpack_thd_global_batch_uses_physical_starts_and_drops_fill_segment():
    unpacked = _unpack_thd_global_batch(_packed_global_batch(), DynamicCPBatchPolicy())

    assert unpacked.logical_lengths == [3, 5]
    assert unpacked.padded_lengths == [4, 8]
    assert torch.equal(unpacked.samples[0]["tokens"], torch.tensor([10, 11, 12]))
    assert torch.equal(unpacked.samples[1]["tokens"], torch.tensor([20, 21, 22, 23, 24]))
    assert torch.equal(unpacked.samples[1]["position_ids"], torch.arange(5))


def test_materialize_microbatch_applies_framework_field_policy():
    policy = DynamicCPBatchPolicy(
        sequence_field_pad_values={
            "tokens": 0,
            "labels": 0,
            "loss_mask": 0,
            "position_ids": 0,
            "advantages": -1.0,
        }
    )
    samples = {
        0: {
            "tokens": torch.tensor([10, 11, 12]),
            "labels": torch.tensor([110, 111, 112]),
            "loss_mask": torch.ones(3),
            "position_ids": torch.arange(3),
            "advantages": torch.tensor([0.1, 0.2, 0.3]),
        },
        1: {
            "tokens": torch.tensor([20, 21, 22, 23, 24]),
            "labels": torch.tensor([120, 121, 122, 123, 124]),
            "loss_mask": torch.ones(5),
            "position_ids": torch.arange(5),
            "advantages": torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8]),
        },
    }

    batch = _materialize_microbatch(
        samples,
        [[0, 1], [0, 1]],
        rank=0,
        logical_lengths=[3, 5],
        padded_lengths=[4, 8],
        policy=policy,
    )

    assert batch["local_cp_size"] == 2
    assert batch["cu_seqlens_q"].tolist() == [0, 3, 8]
    assert batch["cu_seqlens_q_padded"].tolist() == [0, 4, 12]
    assert batch["padding_mask"].tolist() == [
        [False, False, False, True, False, False, False, False, False, True, True, True]
    ]
    assert batch["advantages"][0, 3].item() == -1.0
    assert torch.equal(batch["tokens"][0, 4:9], torch.tensor([20, 21, 22, 23, 24]))


def test_runtime_cp_group_histogram_counts_distinct_groups_per_microbatch():
    sample_id_groups = [
        [[0], [0], [1], [2]],
        [[3, 4], [3, 4], [3, 4], [3, 4]],
    ]

    assert _runtime_cp_group_histogram(sample_id_groups) == {1: 2, 2: 1, 4: 1}


def test_prepare_dynamic_cp_batch_delegates_placement_and_transport(monkeypatch):
    import megatron.bridge.data.dynamic_context_parallel as dcp

    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    pg_collection = SimpleNamespace(dp=group, dp_cp=group, pp=group)
    model_config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=16,
        min_dynamic_context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
    )

    def _gather(lengths, _group):
        values = [int(value) for value in lengths.tolist()]
        ids = torch.arange(len(values), dtype=torch.int32)
        return list(enumerate(values)), ids, torch.tensor([0, len(values)]), values

    monkeypatch.setattr(dcp, "gather_global_sequence_lengths", _gather)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    result = prepare_dynamic_cp_batch(
        iter([_packed_global_batch()]),
        num_microbatches=1,
        model_config=model_config,
        pg_collection=pg_collection,
    )

    assert result.num_microbatches == 1
    assert result.padded_token_sum == 12
    assert result.logical_seqlen_squared_sum == 34
    batch = next(result.data_iterator)
    assert batch["local_cp_size"] == 1
    # The MCore placement policy sorts by descending padded length, so the
    # five-token sample is materialized before the three-token sample.
    assert batch["cu_seqlens_q"].tolist() == [0, 5, 8]
    assert batch["cu_seqlens_q_padded"].tolist() == [0, 8, 12]
    assert torch.equal(batch["tokens"][0, :5], torch.tensor([20, 21, 22, 23, 24]))
