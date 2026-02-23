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

"""Benchmark: direct tensor broadcast vs broadcast_object_list for TP data loading.

Measures the two broadcast mechanisms used by ``get_batch_on_this_tp_rank``
when ``broadcast_data_across_tp=True``:

1. **Direct broadcast** -- fixed-shape tensors (tokens, labels, loss_mask,
   position_ids) are broadcast via individual ``torch.distributed.broadcast``
   calls.  This is the fast path for the standard batch keys.

2. **broadcast_object_list** -- variable-shape or extra keys (e.g. packed-
   sequence metadata) are broadcast via ``torch.distributed.broadcast_object_list``.
   In the common pretraining case there are no extra keys and this path is
   skipped entirely (only the boolean presence flag is broadcast).

Usage::

    torchrun --nproc_per_node=8 benchmarks/bench_broadcast_tp.py
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist


def make_batch(
    mbs: int,
    seq_len: int,
    extra_keys: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Create a synthetic dataloader batch (CPU tensors)."""
    batch = {
        "tokens": torch.randint(0, 32000, (mbs, seq_len), dtype=torch.int64),
        "labels": torch.randint(0, 32000, (mbs, seq_len), dtype=torch.int64),
        "loss_mask": torch.rand(mbs, seq_len, dtype=torch.float32),
        "position_ids": torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).expand(mbs, -1).contiguous(),
    }
    if extra_keys:
        batch.update(extra_keys)
    return batch


def bench_direct_broadcast(
    batch_cpu: dict[str, torch.Tensor],
    group: dist.ProcessGroup,
    src: int,
    rank: int,
    device: torch.device,
    warmup: int = 10,
    iters: int = 50,
) -> list[float]:
    """Benchmark direct per-tensor NCCL broadcast (the fast path)."""
    times: list[float] = []
    for i in range(warmup + iters):
        dist.barrier(group)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if rank == src:
            cuda_batch = {k: v.to(device, non_blocking=True) for k, v in batch_cpu.items()}
            torch.cuda.synchronize()
            for t in cuda_batch.values():
                dist.broadcast(t, src=src, group=group)
        else:
            cuda_batch = {k: torch.empty_like(v, device=device) for k, v in batch_cpu.items()}
            for t in cuda_batch.values():
                dist.broadcast(t, src=src, group=group)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000)

    return times


def bench_broadcast_object_list(
    batch_cpu: dict[str, torch.Tensor],
    group: dist.ProcessGroup,
    src: int,
    rank: int,
    device: torch.device,
    warmup: int = 10,
    iters: int = 50,
) -> list[float]:
    """Benchmark broadcast_object_list path (pickle + NCCL + .cuda())."""
    times: list[float] = []
    for i in range(warmup + iters):
        dist.barrier(group)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        obj = [batch_cpu if rank == src else None]
        dist.broadcast_object_list(obj, src=src, group=group)
        data = obj[0]

        cuda_batch = {}
        for k, v in data.items():
            cuda_batch[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000)

    return times


def _median(values: list[float]) -> float:
    s = sorted(values)
    return s[len(s) // 2]


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    group = dist.group.WORLD
    src = 0

    configs = [
        (1, 8192, None, "mbs=1 seq=8K"),
        (1, 8192, {"cu_seqlens": torch.tensor([0, 100, 200], dtype=torch.int32)}, "mbs=1 seq=8K +extra"),
        (1, 32768, None, "mbs=1 seq=32K"),
        (1, 32768, {"cu_seqlens": torch.tensor([0, 100, 200], dtype=torch.int32)}, "mbs=1 seq=32K +extra"),
        (1, 131072, None, "mbs=1 seq=128K"),
        (1, 262144, None, "mbs=1 seq=256K"),
    ]

    if rank == 0:
        print(f"\nBroadcast TP benchmark -- {world_size} GPUs ({torch.cuda.get_device_name(0)})\n")
        print(f"| {'Config':<24} | {'Data':>8} | {'Direct (ms)':>12} | {'ObjList (ms)':>13} | {'Overhead':>10} |")
        print(f"|{'-'*26}|{'-'*10}|{'-'*14}|{'-'*15}|{'-'*12}|")

    for mbs, seq_len, extra, label in configs:
        batch_cpu = make_batch(mbs, seq_len, extra_keys=extra)
        data_mb = sum(v.nelement() * v.element_size() for v in batch_cpu.values() if isinstance(v, torch.Tensor)) / 1e6

        t_direct = bench_direct_broadcast(batch_cpu, group, src, rank, device)
        t_objlist = bench_broadcast_object_list(batch_cpu, group, src, rank, device)

        if rank == 0:
            d = _median(t_direct)
            o = _median(t_objlist)
            print(f"| {label:<24} | {data_mb:>6.1f}MB | {d:>10.2f}ms | {o:>11.2f}ms | {o - d:>+8.2f}ms |")

    if rank == 0:
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
