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

"""Attention-residual state layout: one tensor per block, no buffer copy, reference values.

The reference implementation keeps the block history in one ``[s, b, hc, nb, H]`` buffer and
copies the whole buffer at every block boundary. These tests pin the state to the same values
while storing one tensor per slot (``state.blocks``), and pin ``state.residual`` -- the property
that re-assembles the reference layout for the pipeline payload and tooling -- to that layout.
"""

import pytest
import torch

from megatron.bridge.models.shensi.modeling_shensi import (
    ShensiAttnResState,
    _attn_res_block_slots,
)


SEQ, BATCH, HC, HIDDEN, BLOCKS = 3, 2, 4, 8, 3


def _streams(requires_grad: bool = False) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(SEQ, BATCH, HC, HIDDEN, requires_grad=requires_grad)


def _begun_state() -> tuple[ShensiAttnResState, torch.Tensor]:
    state = ShensiAttnResState(num_blocks=BLOCKS, hc_mult=HC, hidden_size=HIDDEN)
    streams = _streams()
    state.begin(streams)
    return state, streams


def test_write_block_keeps_the_tensor_instead_of_copying_a_buffer():
    state, streams = _begun_state()
    assert state.blocks == [None] * BLOCKS
    assert state.num_written == 0
    # Before any write the state reads as the reference's freshly allocated (zero) buffer.
    assert torch.equal(state.residual, torch.zeros(SEQ, BATCH, HC, BLOCKS, HIDDEN))

    state.write_block(1, streams)

    assert state.blocks[1] is streams, "the slot must hold the tensor itself, not a copy"
    assert state.num_written == 2
    assert [b is None for b in state.blocks] == [True, False, True]


def test_residual_reassembles_the_reference_layout():
    state, streams = _begun_state()
    second = torch.full_like(streams, 0.25)
    state.write_block(0, streams)
    state.write_block(2, second)

    expected = torch.zeros(SEQ, BATCH, HC, BLOCKS, HIDDEN)
    expected[..., 0, :] = streams
    expected[..., 2, :] = second
    assert torch.equal(state.residual, expected)


def test_residual_stays_differentiable():
    state = ShensiAttnResState(num_blocks=BLOCKS, hc_mult=HC, hidden_size=HIDDEN)
    streams = _streams(requires_grad=True)
    state.begin(streams)
    state.write_block(0, streams)

    state.residual.float().sum().backward()

    assert streams.grad is not None
    assert torch.equal(streams.grad, torch.ones_like(streams))


def test_slot_writes_are_bounded_and_typed():
    state = ShensiAttnResState(num_blocks=BLOCKS, hc_mult=HC, hidden_size=HIDDEN)
    streams = _streams().to(torch.bfloat16)
    state.begin(streams)
    state.write_block(0, _streams().float())

    assert state.blocks[0].dtype == torch.bfloat16, "the slot takes the dtype of the block state"
    with pytest.raises(ValueError, match="outside the state's"):
        state.write_block(BLOCKS, streams)


def test_read_slots_reject_unwritten_and_missing_blocks():
    state, streams = _begun_state()
    state.write_block(0, streams)

    written = _attn_res_block_slots(state.blocks, 1)
    assert written[0] is streams
    with pytest.raises(RuntimeError, match="has not been written"):
        _attn_res_block_slots(state.blocks, 2)
    with pytest.raises(ValueError, match="block list holds"):
        _attn_res_block_slots(state.blocks[:1], 2)
    # The reference layout is accepted slotwise and yields the same values.
    stacked = state.residual
    assert torch.equal(_attn_res_block_slots(stacked, 2)[1], stacked[..., 1, :])


def test_state_before_begin_has_no_layout():
    state = ShensiAttnResState(num_blocks=BLOCKS)
    assert state.residual is None
    with pytest.raises(RuntimeError, match="did not call begin"):
        state.write_block(0, _streams())
