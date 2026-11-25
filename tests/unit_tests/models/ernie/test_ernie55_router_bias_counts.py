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

"""Reference the original contiguous CP owner using unique token IDs and native BF16 accumulation."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.bridge.models.ernie.ernie55_context_parallel import build_ernie55_thd_cp_batch
from megatron.bridge.models.ernie.modeling_ernie55.moe import Ernie55SplitFeatureRouter
from megatron.bridge.models.ernie.modeling_ernie55.router_bias_counts import (
    contiguous_owner_histogram,
    finalize_with_contiguous_cp_counts,
)


def _group(rank, width=4):
    return SimpleNamespace(rank=lambda: rank, size=lambda: width)


def _sample(lengths, rank, layout, cp_size=4):
    samples = []
    starts = []
    end = 0
    for length in lengths:
        ids = torch.arange(end, end + length)
        samples.append((ids, ids.clone()))
        starts.append(end)
        multiple = 2 * cp_size
        end += (length + multiple - 1) // multiple * multiple
    group = _group(rank, cp_size)
    batch = build_ernie55_thd_cp_batch(samples, pad_token_id=-1, cp_group=group, cp_partition_mode=layout)
    return batch, group, starts, end


@pytest.mark.parametrize("lengths", [(33,), (16,) * 8, (2, 31, 7, 43, 5, 11, 25, 1, 17, 9, 6, 14)])
@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
@pytest.mark.parametrize("cp_size", [4, 16])
def test_masked_physical_counts_for_packs(lengths, layout, cp_size):
    experts = 13
    shards = []
    for rank in range(cp_size):
        batch, group, starts, end = _sample(lengths, rank, layout, cp_size)
        ids = batch.tokens.flatten()
        valid = (ids >= 0) & (ids % 7 != 0)
        selected = torch.stack([ids % experts, (ids + 3) % experts], 1)
        shards.append(contiguous_owner_histogram(selected, ~valid, batch.packed_seq_params, group, experts))
    expected = torch.zeros((cp_size, experts), dtype=torch.int32)
    for start, length in zip(starts, lengths):
        for token in range(start, start + length):
            if token % 7:
                owner = token // (end // cp_size)
                expected[owner, token % experts] += 1
                expected[owner, (token + 3) % experts] += 1
    assert torch.equal(sum(shards), expected)


def test_refwd_counts_only_while_training_and_grad_enabled():
    batch, group, _, _ = _sample((16, 24), 0, "zigzag")
    indices = batch.tokens.flatten()
    router = SimpleNamespace(
        config=SimpleNamespace(
            ernie_router_bias_contiguous_cp_counts=True,
            context_parallel_size=4,
            num_moe_experts=7,
            ernie_norm_topk_prob=True,
        ),
        cp_group=group,
        topk=2,
        frozen_expert_bias=False,
        training=True,
        force_balance=False,
        expert_bias=torch.zeros(7),
        routed_scaling_factor_param=torch.ones(7),
        _ernie_cp_bias_histograms=[],
        apply_z_loss=lambda t, padding_mask=None: t,
    )
    logits = torch.randn(len(indices), 7).sigmoid()
    with torch.no_grad():
        Ernie55SplitFeatureRouter.routing(router, logits, packed_seq_params=batch.packed_seq_params)
    assert not router._ernie_cp_bias_histograms
    router.training = False
    Ernie55SplitFeatureRouter.routing(router, logits, packed_seq_params=batch.packed_seq_params)
    assert not router._ernie_cp_bias_histograms
    router.training = True
    Ernie55SplitFeatureRouter.routing(router, logits, packed_seq_params=batch.packed_seq_params)
    assert len(router._ernie_cp_bias_histograms) == 1
    assert router._ernie_cp_bias_histograms[0].sum() == len(indices) * 2
    router.frozen_expert_bias = True
    Ernie55SplitFeatureRouter.routing(router, logits, packed_seq_params=batch.packed_seq_params)
    assert len(router._ernie_cp_bias_histograms) == 1


def test_native_local_bf16_order_and_one_collective_for_multiple_routers(monkeypatch):
    # Test rank 1 using distinct layer, microbatch, expert and source contributions.
    group = _group(1, 2)
    rows = torch.arange(2 * 2 * 2 * 8 * 13).reshape(2, 2, 2, 8, 13).int() % 117
    root = nn.Module()
    root.trunk, root.mtp = nn.Module(), nn.Module()
    calls = []
    for index, router in enumerate((root.trunk, root.mtp)):
        router.cp_group = group
        router.local_tokens_per_expert = torch.zeros(13, dtype=torch.bfloat16)
        router._ernie_cp_bias_histograms = list(rows[1, :, index].unbind(1))

    def reduce_scatter(result, input_, *, group):
        assert group.rank() == 1
        assert torch.equal(input_, rows[1].flatten(0, 1))
        result.copy_(rows[:, 1].sum(0))
        calls.append("exchange")

    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", reduce_scatter)

    def finalize(chunks, count, *, extra):
        assert chunks[0] is root and count == 42 and extra
        for index, router in enumerate((root.trunk, root.mtp)):
            expected = torch.zeros(13, dtype=torch.bfloat16)
            for microbatch in range(8):
                expected += rows[:, 1, index, microbatch].sum(0)
            assert torch.equal(router.local_tokens_per_expert, expected)
            assert not router._ernie_cp_bias_histograms
        calls.append("finalize")
        return 7

    assert finalize_with_contiguous_cp_counts(finalize, [root], 42, extra=True) == 7
    assert calls == ["exchange", "finalize"]
