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

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.shensi.modeling_shensi import (
    ShensiAttnResBank,
    ShensiStreamCompression,
    ShensiStreamState,
    pack_shensi_payload,
    shensi_payload_rows,
    shensi_topk_payload_rows,
    unpack_shensi_payload,
)


SEQ, BATCH, HIDDEN, BLOCKS, RANK, STREAMS = 4, 2, 8, 3, 8, 2


def _modules():
    cfg = SimpleNamespace(
        hidden_size=HIDDEN,
        hc_stream_rank=RANK,
        attn_res_block_summary_rank=RANK,
        hc_mult=STREAMS,
        hc_conv_kernels=[2],
        latent_size=4,
        layernorm_epsilon=1e-6,
        num_layers=2,
    )
    compression = ShensiStreamCompression(cfg)
    bank = ShensiAttnResBank(cfg)
    with torch.no_grad():
        compression.basis.copy_(torch.eye(RANK, HIDDEN))
        bank.basis.copy_(torch.eye(RANK, HIDDEN))
    return compression, bank


def _state(compression, bank):
    embeds = torch.randn(SEQ, BATCH, HIDDEN)
    streams = torch.randn(SEQ, BATCH, STREAMS, HIDDEN)
    prefix = torch.randn(SEQ, BATCH, HIDDEN)
    residual = torch.randn(SEQ, BATCH, BLOCKS, HIDDEN)
    return ShensiStreamState(
        inputs_embeds=embeds,
        streams=compression.encode(streams),
        prefix_sum=prefix,
        residual_encoded=bank.encode(residual),
        rosa_ids=torch.arange(SEQ * BATCH).view(BATCH, SEQ),
    )


def test_payload_rows_counts_every_state_row():
    assert shensi_payload_rows(STREAMS, BLOCKS) == 1 + STREAMS + 1 + BLOCKS + 1


def test_payload_roundtrip():
    compression, bank = _modules()
    state = _state(compression, bank)
    activations = torch.randn(SEQ, BATCH, HIDDEN)
    payload = pack_shensi_payload(activations, state, compression=compression, bank=bank)
    restored_activations, restored = unpack_shensi_payload(
        payload,
        compression=compression,
        bank=bank,
        num_streams=STREAMS,
        num_blocks=BLOCKS,
    )
    assert torch.equal(restored_activations, activations)
    assert torch.allclose(restored.inputs_embeds, state.inputs_embeds)
    assert torch.allclose(compression.decode(restored.streams), compression.decode(state.streams))
    assert torch.allclose(bank.decode(restored.residual_encoded), bank.decode(state.residual_encoded))
    assert torch.equal(restored.rosa_ids, state.rosa_ids)


def test_payload_rejects_a_short_payload():
    compression, bank = _modules()
    with pytest.raises(ValueError, match="does not carry"):
        unpack_shensi_payload(
            torch.zeros(SEQ + 1, BATCH, HIDDEN),
            compression=compression,
            bank=bank,
            num_streams=STREAMS,
            num_blocks=BLOCKS,
        )


def test_payload_carries_topk_indices_byte_exactly():
    """A shared indexer can sit on the stage after its full indexer."""
    compression, bank = _modules()
    state = _state(compression, bank)
    index_topk = 5
    state.topk_indices = torch.randint(-1, 50, (BATCH, SEQ, index_topk), dtype=torch.int32)
    topk_rows = shensi_topk_payload_rows(index_topk, HIDDEN)
    payload = pack_shensi_payload(
        torch.randn(SEQ, BATCH, HIDDEN),
        state,
        compression=compression,
        bank=bank,
        topk_rows=topk_rows,
        index_topk=index_topk,
    )
    assert payload.shape[0] == SEQ * (1 + shensi_payload_rows(STREAMS, BLOCKS, topk_rows))
    _, restored = unpack_shensi_payload(
        payload,
        compression=compression,
        bank=bank,
        num_streams=STREAMS,
        num_blocks=BLOCKS,
        topk_rows=topk_rows,
        index_topk=index_topk,
    )
    assert torch.equal(restored.topk_indices, state.topk_indices)


def test_payload_topk_none_round_trips_as_none():
    compression, bank = _modules()
    state = _state(compression, bank)
    assert state.topk_indices is None
    topk_rows = shensi_topk_payload_rows(5, HIDDEN)
    payload = pack_shensi_payload(
        torch.randn(SEQ, BATCH, HIDDEN),
        state,
        compression=compression,
        bank=bank,
        topk_rows=topk_rows,
        index_topk=5,
    )
    _, restored = unpack_shensi_payload(
        payload,
        compression=compression,
        bank=bank,
        num_streams=STREAMS,
        num_blocks=BLOCKS,
        topk_rows=topk_rows,
        index_topk=5,
    )
    assert restored.topk_indices is None
