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

"""Pipeline payload contract: the stream state rides the stage payload."""

import pytest
import torch

from megatron.bridge.models.shensi.modeling_shensi import (
    ShensiStreamState,
    pack_shensi_payload,
    shensi_payload_rows,
    shensi_topk_payload_rows,
    unpack_shensi_payload,
)


SEQ, BATCH, HIDDEN, BLOCKS, RANK, STREAMS = 4, 2, 8, 3, 8, 2


def _state():
    return ShensiStreamState(
        inputs_embeds=torch.randn(SEQ, BATCH, HIDDEN),
        streams=torch.randn(SEQ, BATCH, STREAMS, HIDDEN),
        prefix_sum=torch.randn(SEQ, BATCH, HIDDEN),
        blocks=torch.randn(SEQ, BATCH, BLOCKS, HIDDEN),
        rosa_ids=torch.arange(SEQ * BATCH).view(BATCH, SEQ),
    )


def test_payload_rows_counts_every_state_row():
    assert shensi_payload_rows(STREAMS, BLOCKS) == 1 + STREAMS + 1 + BLOCKS + 1


def test_payload_roundtrip():
    state = _state()
    activations = torch.randn(SEQ, BATCH, HIDDEN)
    payload = pack_shensi_payload(activations, state)
    restored_activations, restored = unpack_shensi_payload(payload, num_streams=STREAMS, num_blocks=BLOCKS)
    assert torch.equal(restored_activations, activations)
    assert torch.allclose(restored.inputs_embeds, state.inputs_embeds)
    assert torch.allclose(restored.streams, state.streams)
    assert torch.allclose(restored.blocks, state.blocks)
    assert torch.equal(restored.rosa_ids, state.rosa_ids)


def test_payload_rejects_a_short_payload():
    with pytest.raises(ValueError, match="does not carry"):
        unpack_shensi_payload(torch.zeros(SEQ + 1, BATCH, HIDDEN), num_streams=STREAMS, num_blocks=BLOCKS)


def test_payload_carries_topk_indices_byte_exactly():
    """A shared indexer can sit on the stage after its full indexer."""
    state = _state()
    index_topk = 5
    state.topk_indices = torch.randint(-1, 50, (BATCH, SEQ, index_topk), dtype=torch.int32)
    topk_rows = shensi_topk_payload_rows(index_topk, HIDDEN)
    payload = pack_shensi_payload(torch.randn(SEQ, BATCH, HIDDEN), state, topk_rows=topk_rows, index_topk=index_topk)
    assert payload.shape[0] == SEQ * (1 + shensi_payload_rows(STREAMS, BLOCKS, topk_rows))
    _, restored = unpack_shensi_payload(
        payload, num_streams=STREAMS, num_blocks=BLOCKS, topk_rows=topk_rows, index_topk=index_topk
    )
    assert torch.equal(restored.topk_indices, state.topk_indices)


def test_payload_topk_none_round_trips_as_none():
    state = _state()
    assert state.topk_indices is None
    topk_rows = shensi_topk_payload_rows(5, HIDDEN)
    payload = pack_shensi_payload(torch.randn(SEQ, BATCH, HIDDEN), state, topk_rows=topk_rows, index_topk=5)
    _, restored = unpack_shensi_payload(
        payload, num_streams=STREAMS, num_blocks=BLOCKS, topk_rows=topk_rows, index_topk=5
    )
    assert restored.topk_indices is None
