"""
Minimal reproduction of the async P2P send + deallocate race using the same
P2PCommunicator.send_forward_recv_forward() call as pp_post_forward in schedules.py.
See DEBUG_DETERMINISM_NAN.md for full analysis.

Run: torchrun --nproc_per_node=2 reproduce_p2p_dealloc_race.py
"""

import os, time, types
import torch
import torch.distributed as dist
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import deallocate_output_tensor

SHAPE = (2048, 16384)  # ~64 MB BF16


def make_config():
    """Minimal config required by P2PCommunicator._communicate."""
    return types.SimpleNamespace(
        variable_seq_lengths=False,
        mtp_standalone=False,
        pipeline_dtype=torch.bfloat16,
        use_ring_exchange_p2p=False,
        batch_p2p_comm=False,
        batch_p2p_sync=False,
        timers=None,
    )


def run(rank):
    dist.init_process_group('nccl', rank=rank, world_size=2)
    torch.cuda.set_device(rank)

    pp_group = dist.group.WORLD
    comm = P2PCommunicator(pp_group, make_config())

    for fix in [False, True]:
        dist.barrier()

        if rank == 0:  # sender — mirrors pp_post_forward
            output_tensor = torch.full(SHAPE, 1.0, dtype=torch.bfloat16, device='cuda')

            _, wait_handles = comm.send_forward_recv_forward(
                output_tensor, recv_prev=False, tensor_shape=SHAPE, overlap_p2p_comm=True
            )
            send_handle = wait_handles['send_next']

            if fix:
                send_handle.wait()              # FIX: wait before freeing

            deallocate_output_tensor(output_tensor, True)   # free GPU buffer

            # Simulate the next microbatch's matmul reusing the freed memory.
            # In production this isn't NaN — it's whatever the next GPU kernel
            # computes. The corruption is that NCCL reads a mix of old and new data.
            overwrite = torch.full(SHAPE, 2.0, dtype=torch.bfloat16, device='cuda')
            torch.cuda.synchronize()
            time.sleep(0.05)                    # guarantee NCCL reads overwritten memory

            if not fix:
                send_handle.wait()
            del overwrite

        else:  # receiver — mirrors pp_pre_forward
            recv_tensor, wait_handles = comm.send_forward_recv_forward(
                None, recv_prev=True, tensor_shape=SHAPE, overlap_p2p_comm=True
            )
            wait_handles['recv_prev'].wait()

            all_correct = (recv_tensor == 1.0).all().item()
            label = 'FIXED' if fix else 'BUGGY'
            ok = '✓' if all_correct else '✗ race triggered (got corrupted values)'
            print(f'[{label}] all_correct={all_correct} {ok}', flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    run(int(os.environ['RANK']))
