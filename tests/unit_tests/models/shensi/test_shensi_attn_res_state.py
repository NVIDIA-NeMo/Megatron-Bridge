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

"""Attention-residual bank and stream payload layout."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.shensi.modeling_shensi import (
    ShensiAttentionResidual,
    ShensiAttnResBank,
    ShensiStreamState,
    pack_shensi_payload,
    unpack_shensi_payload,
)


SEQ, BATCH, HIDDEN, BLOCKS, RANK, STREAMS = 3, 2, 8, 3, 8, 2


def _config(channels: int = STREAMS) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=HIDDEN,
        latent_size=4,
        hc_mult=channels,
        layernorm_epsilon=1e-6,
        hc_conv_kernels=[2],
        hc_stream_rank=RANK,
        attn_res_rank=RANK,
        attn_res_block_summary_rank=RANK,
        num_layers=2,
    )


def _identity_bank() -> ShensiAttnResBank:
    bank = ShensiAttnResBank(_config())
    with torch.no_grad():
        bank.basis.copy_(torch.eye(RANK, HIDDEN))
        bank.stream_basis.copy_(torch.eye(RANK, HIDDEN))
        bank.token_up.weight.zero_()
    return bank


def test_bank_initialize_finalize_roundtrip():
    bank = _identity_bank()
    embeds = torch.randn(SEQ, BATCH, HIDDEN)
    rows = bank.initialize(embeds, BLOCKS)
    assert rows.shape == (SEQ, BATCH, 1 + BLOCKS, HIDDEN)
    assert torch.equal(rows[..., 0, :], embeds)

    summary = torch.randn(SEQ, BATCH, HIDDEN)
    rows = rows.index_copy(-2, torch.tensor([1]), summary.unsqueeze(-2))
    prefix = torch.randn(SEQ, BATCH, HIDDEN)
    final = bank.finalize(rows, prefix, BLOCKS)
    assert torch.equal(final[..., 0, :], embeds)
    assert torch.allclose(final[..., 1, :], summary)
    assert torch.allclose(final[..., BLOCKS, :], prefix)


def test_stream_bank_roundtrip():
    bank = _identity_bank()
    streams = torch.randn(SEQ, BATCH, STREAMS, HIDDEN)
    latents = bank.encode_streams(streams)
    assert latents.shape == (SEQ, BATCH, STREAMS, RANK)
    assert torch.allclose(bank.decode_streams(latents), streams)


def test_residual_updates_the_stream_and_reads_blocks():
    residual = ShensiAttentionResidual(_config())
    prefix = torch.randn(SEQ, BATCH, HIDDEN, requires_grad=True)
    delta = torch.randn(SEQ, BATCH, HIDDEN)
    blocks = torch.randn(SEQ, BATCH, BLOCKS, HIDDEN)
    output, updated = residual(prefix, delta, blocks, BLOCKS)
    assert output.shape == prefix.shape
    assert updated.shape == prefix.shape
    output.sum().backward()
    assert prefix.grad is not None and torch.isfinite(prefix.grad).all()


def test_payload_roundtrip_preserves_state_and_gradients():
    embeds = torch.randn(SEQ, BATCH, HIDDEN, requires_grad=True)
    streams = torch.randn(SEQ, BATCH, STREAMS, HIDDEN, requires_grad=True)
    prefix = torch.randn(SEQ, BATCH, HIDDEN, requires_grad=True)
    blocks = torch.randn(SEQ, BATCH, BLOCKS, HIDDEN, requires_grad=True)
    rosa_ids = torch.arange(SEQ * BATCH, dtype=torch.long).view(BATCH, SEQ)
    state = ShensiStreamState(
        inputs_embeds=embeds,
        streams=streams,
        prefix_sum=prefix,
        blocks=blocks,
        rosa_ids=rosa_ids,
    )
    activations = torch.randn(SEQ, BATCH, HIDDEN)
    payload = pack_shensi_payload(activations, state)
    assert payload.shape[0] == SEQ * (1 + STREAMS + 1 + BLOCKS + 1) + SEQ

    restored_activations, restored = unpack_shensi_payload(payload, num_streams=STREAMS, num_blocks=BLOCKS)
    assert torch.equal(restored_activations, activations)
    assert torch.allclose(restored.inputs_embeds, embeds)
    assert torch.allclose(restored.prefix_sum, prefix)
    assert torch.allclose(restored.streams, streams)
    assert torch.allclose(restored.blocks, blocks)
    assert torch.equal(restored.rosa_ids, rosa_ids)

    payload.sum().backward()
    assert embeds.grad is not None and streams.grad is not None
    assert prefix.grad is not None and blocks.grad is not None


def test_payload_rejects_a_mismatched_row_count():
    with pytest.raises(ValueError, match="does not carry"):
        unpack_shensi_payload(torch.zeros(SEQ + 1, BATCH, HIDDEN), num_streams=STREAMS, num_blocks=BLOCKS)
