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

"""Attention-residual state carriage inside a pipeline stage payload.

A block that spans a stage boundary hands its state to the next stage by riding the activation
payload itself (``pack_attn_res_payload`` / ``unpack_attn_res_payload``): the schedule ships
whatever the model returns, so no side communicator and no separate process group are involved, and
the state's gradients travel back through the same tensor. These tests pin the layout, the round
trip, the gradient path and the per-stage conditions.
"""

import pytest
import torch

from megatron.bridge.models.shensi.modeling_shensi import (
    AttnResStagePlan,
    ShensiAttnResState,
    attn_res_payload_rows,
    pack_attn_res_payload,
    unpack_attn_res_payload,
)


SEQ, BATCH, HC, HIDDEN, BLOCKS = 3, 2, 4, 8, 3
NUM_LAYERS, N_HASH_LAYERS, BLOCK_SIZE = 8, 2, 2
STAGE_LAYERS = {(0, 0): (0, 1, 2, 3), (1, 0): (4, 5, 6, 7)}


def _streams(dtype=torch.float32, requires_grad=False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(SEQ, BATCH, HC, HIDDEN, generator=generator, dtype=dtype, requires_grad=requires_grad)


def _state_with_blocks(written: int = 2, *, dtype=torch.float32, requires_grad: bool = False):
    state = ShensiAttnResState(num_blocks=BLOCKS, hc_mult=HC, hidden_size=HIDDEN)
    streams = _streams(dtype, requires_grad)
    state.begin(streams)
    for slot in range(written):
        state.write_block(slot, state.prefix_sum + slot)
    return state, streams


def _plan(local_stage=(0, 0)) -> AttnResStagePlan:
    return AttnResStagePlan(
        num_layers=NUM_LAYERS,
        n_hash_layers=N_HASH_LAYERS,
        block_size=BLOCK_SIZE,
        pp_size=2,
        vpp_size=1,
        stage_layers=STAGE_LAYERS,
        local_stage=local_stage,
    )


def test_payload_rows_scale_with_tokens():
    assert attn_res_payload_rows(SEQ, HC, BLOCKS) == SEQ * HC * (1 + BLOCKS)
    assert attn_res_payload_rows(1, HC, BLOCKS) == HC * (1 + BLOCKS)


def test_pack_appends_the_state_below_the_activations():
    state, _ = _state_with_blocks()
    activations = torch.randn(SEQ, BATCH, HIDDEN)

    payload = pack_attn_res_payload(activations, state)

    assert payload.shape == (SEQ * (1 + HC * (1 + BLOCKS)), BATCH, HIDDEN)
    # The activations stay untouched at the top of the payload, and the payload keeps their dtype.
    assert torch.equal(payload[:SEQ], activations)
    assert payload.dtype == activations.dtype


def test_pack_before_any_write_sends_the_empty_slots_as_zeros():
    state = ShensiAttnResState(num_blocks=BLOCKS, hc_mult=HC, hidden_size=HIDDEN)
    streams = _streams()
    state.begin(streams)  # a block may cross the boundary before its write layer runs

    payload = pack_attn_res_payload(torch.zeros(SEQ, BATCH, HIDDEN), state)
    receiver = ShensiAttnResState(num_blocks=BLOCKS, hc_mult=HC, hidden_size=HIDDEN)
    unpack_attn_res_payload(payload, receiver, hc_mult=HC, num_blocks=BLOCKS)

    # The running prefix sum travels, the not-yet-written slots travel as zeros.
    assert torch.equal(receiver.prefix_sum, streams)
    assert all(int(slot.abs().sum()) == 0 for slot in receiver.blocks)


def test_pack_unpack_round_trip():
    state, _ = _state_with_blocks()
    activations = torch.randn(SEQ, BATCH, HIDDEN)

    payload = pack_attn_res_payload(activations, state)
    receiver = ShensiAttnResState(num_blocks=BLOCKS, hc_mult=HC, hidden_size=HIDDEN)
    plain = unpack_attn_res_payload(payload, receiver, hc_mult=HC, num_blocks=BLOCKS)

    assert torch.equal(plain, activations)
    assert receiver.prefix_sum is not None and receiver.blocks[0] is not None
    assert torch.equal(receiver.prefix_sum, state.prefix_sum)
    for idx, want in enumerate(state.blocks):
        if want is None:  # unwritten slots travel as zeros
            assert int(receiver.blocks[idx].abs().sum()) == 0
        else:
            assert torch.equal(receiver.blocks[idx], want)
    assert receiver.num_blocks == BLOCKS
    assert receiver.hidden_size == HIDDEN


def test_payload_gradients_flow_into_the_state():
    """The state's gradients come back through the payload, so no side channel is needed."""
    state, _ = _state_with_blocks(requires_grad=True)
    for slot in state.blocks[:2]:
        slot.retain_grad()  # non-leaf tensors: ask for their gradient explicitly
    activations = torch.randn(SEQ, BATCH, HIDDEN, requires_grad=True)

    payload = pack_attn_res_payload(activations, state)
    payload.sum().backward()

    assert state.prefix_sum is not None and state.prefix_sum.grad is not None
    assert float(state.prefix_sum.grad.abs().sum()) > 0
    assert all(slot is not None and slot.grad is not None for slot in state.blocks[:2])
    assert activations.grad is not None and float(activations.grad.abs().sum()) > 0


def test_unpack_rejects_a_shape_that_is_not_a_payload():
    state = ShensiAttnResState(num_blocks=BLOCKS, hc_mult=HC, hidden_size=HIDDEN)
    bad = torch.randn(SEQ, BATCH, HIDDEN)  # plain activations, no state appended

    with pytest.raises(ValueError, match="does not carry an attention-residual state"):
        unpack_attn_res_payload(bad, state, hc_mult=HC, num_blocks=BLOCKS)


def test_plan_reports_which_stages_carry_the_state():
    """Stage 0 exports (its block continues into stage 1), stage 1 imports, and neither is both."""
    first, second = _plan((0, 0)), _plan((1, 0))

    assert first.carries_state and not first.imports_state
    assert second.imports_state and not second.carries_state
    assert first.local_export_layers() == (3,)
    assert second.local_import_layers() == (4,)
