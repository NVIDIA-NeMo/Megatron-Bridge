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

"""Cross-stage attention-residual state transfer.

The attention-residual state of a block that spans a pipeline boundary travels on a dedicated
group: the layer that closes the block publishes header and state tensors, the first reader on the
next stage imports them, and the gradient of the imported state returns on the reverse channel.
Two ranks on a gloo group exercise the whole exchange, so it runs without accelerators; a
single-process case covers what the per-transfer record keeps and what it leaves out.
"""

import hashlib
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.bridge.models.shensi.modeling_shensi import (
    HEADER_LEN,
    AttnResStagePlan,
    ShensiAttnResPPHandoff,
    ShensiAttnResState,
    _assert_replica,
    _background_send,
    num_attn_res_blocks,
)


NUM_LAYERS = 4
N_HASH_LAYERS = 1
BLOCK_SIZE = 3
SEQ, BATCH, HC, HIDDEN = 3, 2, 4, 8
STATE_DTYPE = torch.bfloat16
# Layer 1 closes a block whose readers sit on the second stage, so the boundary is crossed there.
STAGE_LAYERS = {(0, 0): (0, 1), (1, 0): (2, 3)}
EXPORT_LAYER = 1
IMPORT_LAYER = 2
STATE_NUMEL = SEQ * BATCH * HC * HIDDEN * (1 + 2)  # prefix_sum plus two written block slots
HEADER_BYTES = HEADER_LEN * 8
# The second stage weights the two halves of the state differently so the returning gradient
# identifies which half it came from. Slot 0 of the residual holds a copy of the prefix sum, so
# the gradient of that slot reaches the prefix sum as well.
PREFIX_GRAD = 2.0
RESIDUAL_GRAD = 5.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _streams() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(SEQ, BATCH, HC, HIDDEN, generator=generator).to(STATE_DTYPE)


def _worker(rank: int, port: int, out: dict) -> None:
    torch.distributed.init_process_group("gloo", rank=rank, world_size=2, init_method=f"tcp://127.0.0.1:{port}")
    try:
        group = dist.new_group([0, 1])
        num_blocks = num_attn_res_blocks(NUM_LAYERS, N_HASH_LAYERS, BLOCK_SIZE)
        plan = AttnResStagePlan(
            num_layers=NUM_LAYERS,
            n_hash_layers=N_HASH_LAYERS,
            block_size=BLOCK_SIZE,
            pp_size=2,
            vpp_size=1,
            stage_layers=STAGE_LAYERS,
            local_stage=(rank, 0),
        )
        if rank == 0:
            assert plan.needs_export(EXPORT_LAYER)
            assert not plan.needs_import(EXPORT_LAYER)
        else:
            assert plan.needs_import(IMPORT_LAYER)
            assert not plan.needs_export(IMPORT_LAYER)

        handoff = ShensiAttnResPPHandoff(plan, group=group, local_stage=(rank, 0), auto_group=False)
        state = ShensiAttnResState(num_blocks=num_blocks, hc_mult=HC, hidden_size=HIDDEN)
        streams = _streams()

        if rank == 0:
            streams.requires_grad_(True)
            state.begin(streams)
            state.write_block(0, state.prefix_sum)
            # An output that does not depend on the state leaves the returning gradient alone.
            stage_output = torch.full((SEQ, BATCH, HC, HIDDEN), 0.5, requires_grad=True)
            stage_output = handoff.export_layer(state, EXPORT_LAYER, stage_output=stage_output)
            dist.barrier()
            stage_output.sum().backward()
            returned = PREFIX_GRAD + RESIDUAL_GRAD
            assert state.prefix_sum.grad is not None
            assert torch.equal(state.prefix_sum.grad.float(), torch.full_like(state.prefix_sum.grad.float(), returned))
            handoff.flush()
            out["export_bytes"] = int(handoff.payload_diagnostics()["released_bytes"])
        else:
            hidden_flat = torch.empty(SEQ * BATCH * HC * HIDDEN, dtype=STATE_DTYPE)
            info = handoff.import_layer(state, IMPORT_LAYER, hidden_flat=hidden_flat)
            assert info["numel"] == STATE_NUMEL
            assert info["num_blocks"] == num_blocks
            assert info["num_written"] == 1
            assert info["shape"] == (SEQ, BATCH, HC, HIDDEN)
            assert torch.equal(state.prefix_sum, streams)
            expected_residual = torch.zeros(SEQ, BATCH, HC, num_blocks, HIDDEN, dtype=STATE_DTYPE)
            expected_residual[..., 0, :] = streams
            assert torch.equal(state.residual, expected_residual)
            dist.barrier()
            loss = state.prefix_sum.float().sum() * PREFIX_GRAD + state.residual.float().sum() * RESIDUAL_GRAD
            loss.backward()
            handoff.flush()
            out["import_bytes"] = int(handoff.payload_diagnostics()["released_bytes"])
    finally:
        torch.distributed.destroy_process_group()


def test_block_state_crosses_a_pipeline_boundary() -> None:
    """The imported block state matches the published one, in its own dtype, gradient included."""
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_worker, args=(_free_port(), out), nprocs=2, join=True)

    state_bytes = STATE_NUMEL * torch.tensor([], dtype=STATE_DTYPE).element_size()
    assert out["export_bytes"] == HEADER_BYTES + state_bytes
    assert out["import_bytes"] == state_bytes


def test_transfer_record_holds_a_digest_only_in_the_tracking_mode(monkeypatch) -> None:
    """A transfer always records its shape, and reads the payload back only when asked to."""
    num_blocks = num_attn_res_blocks(NUM_LAYERS, N_HASH_LAYERS, BLOCK_SIZE)
    plan = AttnResStagePlan(
        num_layers=NUM_LAYERS,
        n_hash_layers=N_HASH_LAYERS,
        block_size=BLOCK_SIZE,
        pp_size=2,
        vpp_size=1,
        stage_layers=STAGE_LAYERS,
        local_stage=(0, 0),
    )
    handoff = ShensiAttnResPPHandoff(plan, group=None, local_stage=(0, 0), auto_group=False)
    state = ShensiAttnResState(num_blocks=num_blocks, hc_mult=HC, hidden_size=HIDDEN)
    streams = _streams()
    state.begin(streams)
    state.write_block(0, state.prefix_sum)
    _, payload = state.state_for_pp()
    expected = hashlib.sha256(payload.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]

    monkeypatch.delenv("SHENSI_ATTN_RES_PP_TRACK_PAYLOAD", raising=False)
    handoff.export_layer(state, EXPORT_LAYER, send=False)
    assert handoff.records[-1]["payload_numel"] == STATE_NUMEL
    assert "payload_sha256" not in handoff.records[-1]

    monkeypatch.setenv("SHENSI_ATTN_RES_PP_TRACK_PAYLOAD", "1")
    handoff.export_layer(state, EXPORT_LAYER, send=False)
    assert handoff.records[-1]["payload_sha256"] == expected


def _replica_worker(rank: int, port: int, out: dict) -> None:
    torch.distributed.init_process_group("gloo", rank=rank, world_size=2, init_method=f"tcp://127.0.0.1:{port}")
    try:
        group = dist.new_group([0, 1])
        _assert_replica(torch.full((4, 3), 0.25, dtype=STATE_DTYPE), group, "same")
        out["same_ok"] = True
        different = torch.full((4, 3), 0.25 if rank == 0 else 0.5, dtype=STATE_DTYPE)
        try:
            _assert_replica(different, group, "different")
        except RuntimeError as exc:
            out[f"raised_{rank}"] = "differs" in str(exc)
    finally:
        torch.distributed.destroy_process_group()


def test_replica_check_compares_every_rank() -> None:
    """The expert-parallel replica check accepts equal shards and reports the rank that differs."""
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_replica_worker, args=(_free_port(), out), nprocs=2, join=True)
    assert out["same_ok"]
    assert out["raised_0"] and out["raised_1"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the rule is about device tensors")
def test_send_rule_follows_the_backend(monkeypatch) -> None:
    """A device tensor travels in the background on NCCL, and blocks everywhere else."""
    device = torch.empty(4, device="cuda")
    host = torch.empty(4)

    monkeypatch.setattr(dist, "get_backend", lambda group=None: "nccl")
    assert _background_send(object(), device) is True
    assert _background_send(object(), host) is False

    monkeypatch.setattr(dist, "get_backend", lambda group=None: "gloo")
    assert _background_send(object(), device) is False

    def _raise(group=None):
        raise RuntimeError("no backend for this group")

    monkeypatch.setattr(dist, "get_backend", _raise)
    assert _background_send(object(), device) is False
