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

"""The ERC global view gathers the routed-expert weights of every expert-parallel rank.

Two ranks on a gloo group hold different local experts; the gather has to return them in rank
order, hand each rank its own shard back in the backward pass, and keep the local-view mode
(``local_view=True``) at the rank's own experts. The gradient reduction modes are the
reason this path carries its own autograd function, so both of them are asserted here.
"""

import socket

import pytest
import torch
import torch.multiprocessing as mp

from megatron.bridge.models.shensi.modeling_shensi import erc_gather_expert_weights


E_LOCAL = 2  # routed experts held by each rank
HIDDEN = 8
RANK_DIM = 4  # 2 * moe_latent_size of the stub
WORLD = 2


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _StubMoE:
    """The surface ``erc_gather_expert_weights`` reads from a MoE layer."""

    def __init__(self, rank: int, *, requires_grad: bool = False) -> None:
        self.num_local_experts = E_LOCAL
        base = torch.arange(E_LOCAL * 2 * RANK_DIM * HIDDEN, dtype=torch.float32) + rank * 1000.0
        self._gate_up = base.reshape(E_LOCAL, 2 * RANK_DIM, HIDDEN).to(torch.bfloat16)
        if requires_grad:
            self._gate_up.requires_grad_(True)
        self.fc1_latent_proj = torch.nn.Linear(1, 1, bias=False)

    def erc_weights(self):
        router = torch.full((4 * E_LOCAL, HIDDEN), 0.5, dtype=torch.bfloat16)
        return router, torch.ones(1, 1, dtype=torch.bfloat16), self._gate_up


def _worker(rank: int, port: int, out: dict) -> None:
    torch.distributed.init_process_group("gloo", rank=rank, world_size=WORLD, init_method=f"tcp://127.0.0.1:{port}")
    try:
        from megatron.core import parallel_state as ps

        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=WORLD,
        )
        out[f"ep_size_{rank}"] = int(ps.get_expert_model_parallel_world_size())
        out[f"ep_rank_{rank}"] = int(ps.get_expert_model_parallel_rank())

        mlp = _StubMoE(rank)
        router, down, gate_up = erc_gather_expert_weights(mlp)
        local = mlp.erc_weights()[2]
        out["gathered_shape"] = tuple(gate_up.shape)
        out[f"own_block_{rank}"] = bool(torch.equal(gate_up[rank * E_LOCAL : (rank + 1) * E_LOCAL], local))
        out[f"router_{rank}"] = tuple(router.shape)
        if rank == 0:
            out["blocks_differ"] = not torch.equal(gate_up[E_LOCAL : 2 * E_LOCAL], local)

        # Both reduction modes, told apart by scoring on rank 0 only: a slice reduction keeps the
        # gradient a rank's own loss produced, the mean averages that gradient over the group.
        scale = 1.0 if rank == 0 else 0.0
        for mode in ("slice", "reduce_scatter_mean"):
            stub = _StubMoE(rank, requires_grad=True)
            _, _, gathered = erc_gather_expert_weights(stub, grad_reduce=mode)
            (gathered.float().sum() * scale).backward()
            out[f"grad_{mode}_{rank}"] = float(stub._gate_up.grad.float().sum())

        # Local-view mode keeps the rank's own experts and skips the gather.
        _, _, local_only = erc_gather_expert_weights(_StubMoE(rank), local_view=True)
        out["local_view_shape"] = tuple(local_only.shape)
        ps.destroy_model_parallel()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the expert-parallel init needs a CUDA device")
def test_global_view_covers_every_rank() -> None:
    """The gathered weights follow rank order, and each reduction mode returns its own layout."""
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_worker, args=(_free_port(), out), nprocs=WORLD, join=True)

    for rank in range(WORLD):
        assert out[f"ep_size_{rank}"] == WORLD
        assert out[f"ep_rank_{rank}"] == rank
        assert out[f"own_block_{rank}"], f"rank {rank} did not get its own experts back"
        assert out[f"router_{rank}"] == (4 * E_LOCAL, HIDDEN)
    assert out["gathered_shape"] == (WORLD * E_LOCAL, 2 * RANK_DIM, HIDDEN)
    assert out["blocks_differ"], "the gathers carry the same experts on both ranks"

    # A slice reduction keeps the gradient a rank's own loss produced, so only rank 0 sees its
    # shard. The mean averages over the group instead: rank 0's loss covers every shard, so the
    # averaged gradient reaches both ranks at 1 / group size of the element count.
    elements = E_LOCAL * 2 * RANK_DIM * HIDDEN
    assert out["grad_slice_0"] == elements
    assert out["grad_slice_1"] == 0.0
    assert out["grad_reduce_scatter_mean_0"] == elements / WORLD
    assert out["grad_reduce_scatter_mean_1"] == elements / WORLD

    assert out["local_view_shape"] == (E_LOCAL, 2 * RANK_DIM, HIDDEN)
