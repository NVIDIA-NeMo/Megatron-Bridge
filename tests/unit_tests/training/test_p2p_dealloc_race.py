# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""
Test for the async P2P send + deallocate race fixed in schedules.py.

Uses the same P2PCommunicator.send_forward_recv_forward() + deallocate_output_tensor()
calls as pp_post_forward. A sleep after deallocating guarantees the race fires,
making both the buggy and fixed cases deterministic.

Run: pytest tests/unit_tests/training/test_p2p_dealloc_race.py -v
"""

import time, types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import deallocate_output_tensor

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least 2 GPUs"
)

SHAPE = (2048, 16384)  # ~64 MB BF16


def _make_config():
    return types.SimpleNamespace(
        variable_seq_lengths=False,
        mtp_standalone=False,
        pipeline_dtype=torch.bfloat16,
        use_ring_exchange_p2p=False,
        batch_p2p_comm=False,
        batch_p2p_sync=False,
        timers=None,
        # Mirror the VPP production path: activations are pseudo-deallocated
        # after each send to reclaim GPU memory across pipeline stages.
        deallocate_pipeline_outputs=True,
    )


def _worker(rank: int, fix: bool, result_queue) -> None:
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:29503',
                            rank=rank, world_size=2)
    torch.cuda.set_device(rank)

    config = _make_config()
    comm = P2PCommunicator(dist.group.WORLD, config)

    if rank == 0:  # sender — mirrors pp_post_forward
        output_tensor = torch.full(SHAPE, 1.0, dtype=torch.bfloat16, device='cuda')

        _, wait_handles = comm.send_forward_recv_forward(
            output_tensor, recv_prev=False, tensor_shape=SHAPE, overlap_p2p_comm=True
        )
        send_handle = wait_handles['send_next']

        if fix:
            send_handle.wait()                      # FIX: wait before free

        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        # Allocate a same-size tensor with a different value to force the allocator
        # to reuse the freed block — mirroring what the next microbatch's matmul
        # would do in the real training loop. In production the overwriting values
        # aren't NaN specifically; they're arbitrary compute results that corrupt
        # the transmitted data and eventually cause NaN in loss/grad norm downstream.
        overwrite = torch.full(SHAPE, 2.0, dtype=torch.bfloat16, device='cuda')
        torch.cuda.synchronize()
        time.sleep(0.05)                            # guarantee NCCL reads overwritten buffer

        if not fix:
            send_handle.wait()
        del overwrite

    else:  # receiver — mirrors pp_pre_forward
        recv_tensor, wait_handles = comm.send_forward_recv_forward(
            None, recv_prev=True, tensor_shape=SHAPE, overlap_p2p_comm=True
        )
        wait_handles['recv_prev'].wait()
        # With fix: all values should be 1.0 (what sender put in)
        # Without fix: values are corrupted by the overwriting kernel
        result_queue.put((recv_tensor == 1.0).all().item())

    dist.destroy_process_group()


def _run(fix: bool) -> bool:
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, fix, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    return q.get_nowait()  # has_nan


def test_buggy_dealloc_before_wait_corrupts_data():
    """Deallocating before send completes lets the next kernel overwrite the buffer,
    corrupting what the receiver gets — mirroring what happens in the training loop
    when the next microbatch's matmul reuses the freed activation memory."""
    all_correct = _run(fix=False)
    assert not all_correct, "Race did not fire — increase SHAPE or sleep duration"


def test_fixed_wait_before_dealloc_preserves_data():
    """Waiting for send before deallocating guarantees the receiver always gets
    the original tensor values regardless of what runs next on the GPU."""
    all_correct = _run(fix=True)
    assert all_correct, "Received wrong values even with fix applied"
