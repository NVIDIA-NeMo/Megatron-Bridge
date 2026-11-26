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
"""Disaggregated refit: push a trainer's weights to rollout ranks on other GPUs over NCCL M2N.

Every HF tensor takes one of two routes, fixed when the two sides connect:

* **reshard**: the trainer's local Megatron shard already is a canonical HF view (Bridge's
  ``LocalHFParamSpec``: dense and expert linear weights, the bulk of a MoE model's bytes). It moves
  shard-to-shard from the trainer's TP/ETP layout to the rollout's, with no gather and no copy.
* **collective**: everything else (fused QKV, offset norms, vocab-padded embeddings, GDN projections,
  LoRA-merged weights). Bridge's export gathers it on the trainer (TP/EP/PP), and the full tensor goes
  from the trainer ranks to every rollout rank.

Both routes are ``nccl.m2n`` reshard calls on one NCCL communicator spanning trainer ranks ``[0, M)`` and
rollout ranks ``[M, M + N)``. Calls are recorded in groups of about ``bucket_bytes``, which the library
packs into few transfers, on a side stream: ``send`` and ``receive`` return once everything is enqueued.
The trainer builds the plan (entry order, buckets, meshes) once and hands it to the rollout through the
rendezvous store, so both sides issue identical atoms (``RefitPlan.atoms``) with no per-refit metadata exchange.

This module imports only torch (and nccl4py when connecting), so a rollout process without Megatron can
load this file by path (``importlib.util.spec_from_file_location``); the plan crosses as plain JSON.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from typing import Any

import torch


logger = logging.getLogger(__name__)

_EXPERT_INDEX = re.compile(r"\.experts\.(\d+)\.")
# Vocab-parallel tensors carry Megatron's vocab padding, so their local shards are not HF views.
_VOCAB_PARAMS = ("word_embeddings.weight", "output_layer.weight")
_RESHARD_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

Mesh = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RolloutLayout:
    """How the rollout ranks hold the weights; the trainer and every rollout rank pass the same value.

    Rollout rank ``r`` is communicator rank ``M + r``. A tensor that tensor-parallel engines shard is split
    along the dim the trainer's Megatron layer shards (0 for column-, 1 for row-parallel) over each run of
    ``tensor_parallel_size`` consecutive ranks. With ``expert_parallel_size > 1``, expert ``e`` of ``E``
    lives whole on the ranks with ``r % expert_parallel_size == e // (E // expert_parallel_size)`` (vLLM's
    linear placement when expert parallelism spans its data- and tensor-parallel ranks).
    """

    world_size: int
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1

    def __post_init__(self) -> None:
        if self.world_size % self.tensor_parallel_size or self.world_size % self.expert_parallel_size:
            raise ValueError(f"{self} does not tile its world size.")


@dataclass(frozen=True)
class RefitEntry:
    """One HF tensor of the plan.

    Attributes:
        name: HF name. Experts are per-expert tensors (``...experts.<e>.gate_proj.weight``).
        shape: Global shape.
        dtype: Wire dtype; the receiver gets exactly this.
        sources: Trainer ranks holding it. Each row is a run of consecutive ranks that together hold the
            whole tensor, split along ``split_dim`` when the row has several ranks; rows are replicas.
        split_dim: Dim a source row splits; ``None`` when each source rank holds the tensor whole.
        tp_dim: Dim a tensor-parallel rollout splits; ``None`` keeps it whole on every rollout rank.
        expert: Global expert index, or ``None`` for a non-expert tensor.
        reshard: Route; ``False`` is the collective route (full tensors from Bridge's export).
        bucket: Submission group; the entries of a bucket go in one ``nccl.m2n`` group.
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    sources: Mesh
    split_dim: int | None
    tp_dim: int | None
    expert: int | None
    reshard: bool
    bucket: int = 0


@dataclass(frozen=True)
class RefitAtom:
    """Representing one ``nccl.m2n`` reshard: ``entry`` from one source row to one block of rollout ranks.
    One RefitEntry can be split into one or multiple RefitAtoms.

    Attributes:
        entry: The tensor.
        src: The source row, as a one-row rank grid, split along ``src_dim`` (``None``: held whole).
        src_dim: Split dim of the source row.
        dst: The destination block; several rows when a tensor-parallel rollout splits along ``dst_dim``.
        dst_dim: Split dim within each destination row (``None``: every rank receives it whole).
    """

    entry: RefitEntry
    src: Mesh
    src_dim: int | None
    dst: Mesh
    dst_dim: int | None


@dataclass(frozen=True)
class RefitPlan:
    """Every tensor a refit moves, in issue order, and the sizes of the two sides."""

    trainer_world_size: int
    layout: RolloutLayout
    num_experts: int
    entries: tuple[RefitEntry, ...]

    @property
    def num_buckets(self) -> int:
        """Number of submission groups."""
        return self.entries[-1].bucket + 1 if self.entries else 0

    def atoms(self) -> list[list[RefitAtom]]:
        """Per bucket, the atoms every rank issues, in order; both sides derive the same list.

        Each destination block reads from one source row; blocks rotate over the replicas to spread the load.
        """
        buckets: list[list[RefitAtom]] = [[] for _ in range(self.num_buckets)]
        for entry in self.entries:
            for i, (dst, dst_dim) in enumerate(self._targets(entry)):
                src = entry.sources[i % len(entry.sources)]
                buckets[entry.bucket].append(RefitAtom(entry, (src,), entry.split_dim, dst, dst_dim))
        return buckets

    def _targets(self, entry: RefitEntry) -> list[tuple[Mesh, int | None]]:
        """Destination blocks of ``entry``: contiguous blocks of rollout ranks, each with its split dim."""
        offset, layout = self.trainer_world_size, self.layout
        ranks = tuple(range(offset, offset + layout.world_size))
        if entry.expert is not None and layout.expert_parallel_size > 1:
            owner = entry.expert // (self.num_experts // layout.expert_parallel_size)
            return [
                (((offset + r,),), None) for r in range(layout.world_size) if r % layout.expert_parallel_size == owner
            ]
        tp = layout.tensor_parallel_size
        if entry.reshard and entry.tp_dim is not None and tp > 1:
            return [(tuple(ranks[i : i + tp] for i in range(0, len(ranks), tp)), entry.tp_dim)]
        return [((ranks,), None)]

    def encode(self) -> bytes:
        """Serialize for the rollout; plain JSON rather than pickle, so no Megatron class is needed to read it."""
        entries = [
            [
                e.name,
                e.shape,
                str(e.dtype).removeprefix("torch."),
                e.sources,
                e.split_dim,
                e.tp_dim,
                e.expert,
                e.reshard,
                e.bucket,
            ]
            for e in self.entries
        ]
        return json.dumps(
            {
                "trainers": self.trainer_world_size,
                "layout": asdict(self.layout),
                "experts": self.num_experts,
                "entries": entries,
            }
        ).encode()

    @classmethod
    def decode(cls, data: bytes) -> RefitPlan:
        """Inverse of ``encode``."""
        raw = json.loads(data)
        entries = tuple(
            RefitEntry(
                name,
                tuple(shape),
                getattr(torch, dtype),
                tuple(map(tuple, sources)),
                split,
                tp_dim,
                expert,
                reshard,
                bucket,
            )
            for name, shape, dtype, sources, split, tp_dim, expert, reshard, bucket in raw["entries"]
        )
        return cls(raw["trainers"], RolloutLayout(**raw["layout"]), raw["experts"], entries)


class _Watchdog:
    """Aborts the channel unless the refit it guards completes within ``timeout_s`` of starting.

    It covers the host side too: when a peer is gone, recording an M2N group can block before any
    event exists, and only an abort from another thread unblocks it. Some blocks are CUDA driver calls
    an abort cannot interrupt (``cuMemMap`` of a dead peer's memory while the first group registers
    its staging window), so if the refit is still stuck ``abort_grace_s`` after the abort, the process
    exits, as torch's NCCL watchdog does.
    """

    abort_grace_s = 30.0

    def __init__(self, channel: _Channel, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self.fired = False
        self.event: torch.cuda.Event | None = None
        self._done = threading.Event()
        deadline = time.monotonic() + timeout_s
        threading.Thread(target=self._run, args=(channel, deadline), daemon=True).start()

    def _run(self, channel: _Channel, deadline: float) -> None:
        while time.monotonic() < deadline:
            if self._done.wait(0.01) or (self.event is not None and self.event.query()):
                return
        self.fired = True
        threading.Thread(target=channel.abort, daemon=True).start()
        if not self._done.wait(self.abort_grace_s):
            logger.critical(f"{self.error()} The refit stayed blocked after the abort; exiting the process.")
            os._exit(1)

    def stop(self) -> None:
        self._done.set()

    def error(self) -> TimeoutError:
        return TimeoutError(f"refit did not complete within {self.timeout_s} s; the channel was aborted.")


@dataclass
class RefitHandle:
    """Completion of one ``send`` or ``receive``, recorded on the refit stream.

    When the refit had a timeout and missed it (a peer died or never joined), the communicator is aborted
    and ``synchronize`` raises ``TimeoutError``; both sides must then reconnect, since M2N is fail-stop.
    If nothing raises within ``_Watchdog.abort_grace_s`` of the abort (the host is stuck in a driver
    call, or nobody synchronizes), the process exits instead.
    """

    event: torch.cuda.Event
    watchdog: _Watchdog | None = None

    def wait(self) -> None:
        """Order the current stream after the transfer, without blocking the host."""
        torch.cuda.current_stream().wait_event(self.event)

    def synchronize(self) -> None:
        """Block the host until the transfer is complete."""
        if self.watchdog is None:
            self.event.synchronize()
            return
        while not self.event.query() and not self.watchdog.fired:
            time.sleep(1e-4)
        if self.watchdog.fired:
            self.watchdog.stop()
            raise self.watchdog.error()


# ---- geometry: shapes and M2N descriptors of one rank -------------------------------


def _local_shape(shape: tuple[int, ...], mesh: Mesh, dim: int | None) -> tuple[int, ...]:
    if dim is None:
        return shape
    parts = len(mesh[0])
    if shape[dim] % parts:
        raise ValueError(f"dim {dim} of {shape} does not split evenly over {parts} ranks.")
    return shape[:dim] + (shape[dim] // parts,) + shape[dim + 1 :]


@dataclass(frozen=True)
class _Side:
    mesh: Any  # nccl.m2n.Mesh
    placements: tuple[Any, ...]
    wire_shape: tuple[int, ...]  # what M2N moves; collective tensors travel flat
    shape: tuple[int, ...]  # the local logical shape the caller sees
    member: bool


class _Channel:
    """The communicator, the M2N handle, the refit stream and the compiled atoms of one rank."""

    def __init__(self, plan: RefitPlan, rank: int, comm, store: Any) -> None:
        from nccl import m2n

        self.plan = plan
        self.rank = rank
        self.comm = comm
        self.store = store  # the master side must outlive the rendezvous
        self.handle = m2n.init()
        self.aborted = False
        self.stream = torch.cuda.Stream()
        self.buckets = [[self._compile(atom) for atom in atoms] for atoms in plan.atoms()]

    @classmethod
    def connect(cls, address: str, port: int, rank: int, *, plan: RefitPlan | None, timeout_s: float) -> _Channel:
        """Rendezvous through the trainer rank 0's TCP store and build the M+N communicator.

        The trainer's rank 0 (``plan`` given and ``rank == 0``) hosts the store and publishes the plan and
        the NCCL id; every other rank reads them. A rollout rank passes ``plan=None`` and its rollout index.

        Both ends then run one full warm-up refit (``RefitSender`` / ``RefitReceiver`` constructors). The
        first M2N group registers its NCCL windows, mapping every peer's memory with driver calls
        (cuMemMap) an abort cannot interrupt; after the warm-up, a peer that dies during ``refit()`` leaves
        only abortable work, so the caller gets ``TimeoutError``.
        """
        from nccl.core.communicator import Communicator
        from nccl.core.utils import UniqueId, get_unique_id

        is_master = plan is not None and rank == 0
        store = torch.distributed.TCPStore(
            address, port, is_master=is_master, timeout=timedelta(seconds=timeout_s), wait_for_workers=False
        )
        if is_master:
            store.set("refit/plan", plan.encode())
            store.set("refit/nccl_id", get_unique_id().as_bytes)
        if plan is None:
            plan = RefitPlan.decode(store.get("refit/plan"))
            rank += plan.trainer_world_size
        unique_id = UniqueId.from_bytes(store.get("refit/nccl_id"))
        world_size = plan.trainer_world_size + plan.layout.world_size
        comm = Communicator.init(nranks=world_size, rank=rank, unique_id=unique_id)
        return cls(plan, rank, comm, store)

    def _compile(self, atom: RefitAtom) -> tuple[RefitEntry, _Side, _Side]:
        from nccl import m2n

        entry = atom.entry

        def side(mesh: Mesh, dim: int | None) -> _Side:
            inner = m2n.Shard(dim) if dim is not None else m2n.Replicate()
            two_d = len(mesh) > 1
            shape = _local_shape(entry.shape, mesh, dim)
            # M2N handles at most 3 dims; whole tensors can travel flat.
            wire = shape if entry.reshard else (int(torch.Size(shape).numel()),)
            return _Side(
                mesh=m2n.Mesh.from_ranks(mesh if two_d else mesh[0]),
                placements=(m2n.Replicate(), inner) if two_d else (inner,),
                wire_shape=wire,
                shape=shape,
                member=any(self.rank in row for row in mesh),
            )

        return entry, side(atom.src, atom.src_dim), side(atom.dst, atom.dst_dim)

    def issue(self, bucket: int, sources: dict[str, torch.Tensor], destinations: dict[str, torch.Tensor]) -> None:
        """Record one bucket's atoms as one group; the dicts hold this rank's buffers by HF name."""
        from nccl import m2n

        with m2n.group():
            for entry, src, dst in self.buckets[bucket]:
                descriptors = []
                for side, buffers in ((src, sources), (dst, destinations)):
                    buffer = buffers[entry.name] if side.member else None
                    if buffer is not None:
                        if buffer.dtype != entry.dtype or tuple(buffer.shape) != side.shape:
                            raise ValueError(
                                f"{entry.name}: buffer {tuple(buffer.shape)} {buffer.dtype}, "
                                f"expected {side.shape} {entry.dtype}."
                            )
                        buffer = buffer.view(side.wire_shape)
                    descriptors.append(
                        m2n.DistTensor(
                            buffer, side.wire_shape, entry.dtype, mesh=side.mesh, placements=side.placements
                        )
                    )
                self.handle.reshard(self.comm, *descriptors, stream=self.stream)

    def fence(self, watchdog: _Watchdog | None) -> RefitHandle:
        handle = RefitHandle(self.stream.record_event(), watchdog)
        if watchdog is not None:
            watchdog.event = handle.event
        return handle

    @contextlib.contextmanager
    def guarded(self, timeout_s: float | None) -> Iterator[_Watchdog | None]:
        """Arm a watchdog for one refit and turn a failure it caused into ``TimeoutError``."""
        if self.aborted:
            raise RuntimeError("The refit channel was aborted; reconnect both sides.")
        watchdog = _Watchdog(self, timeout_s) if timeout_s is not None else None
        try:
            yield watchdog
        except Exception as error:
            if watchdog is not None:
                watchdog.stop()
                if watchdog.fired:
                    raise watchdog.error() from error
            raise

    def abort(self) -> None:
        self.aborted = True
        self.comm.abort()

    def close(self) -> None:
        self.handle.destroy()
        if not self.aborted:
            self.comm.destroy()


# ---- trainer side ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _View:
    param: torch.Tensor
    spec: Any  # LocalHFParamSpec

    def tensor(self) -> torch.Tensor:
        return self.spec.select(self.param.detach())


def _reshard_views(tasks: list[Any], pg_collection: Any) -> tuple[dict[str, _View], list[tuple]]:
    """This rank's HF views that can go shard-to-shard, and the facts the plan needs about each.

    A view qualifies when Bridge declares the parameter a canonical local HF view and the parameter is a
    plain 2-D weight split, if at all, evenly along one dim (``partition_stride`` 1) by TP or expert TP.
    """
    rank = torch.distributed.get_rank()
    views: dict[str, _View] = {}
    reports: list[tuple] = []
    for task in tasks:
        param = task.param_weight
        if (
            param is None
            or param.ndim != 2
            or param.dtype not in _RESHARD_DTYPES
            or type(param.data) is not torch.Tensor  # quantized storage (FP8 tensor subclasses)
            or task.global_param_name.endswith(_VOCAB_PARAMS)
            or getattr(param, "partition_stride", 1) != 1
        ):
            continue
        specs = task.local_hf_param_specs()
        expert = not getattr(param, "allreduce", True)
        index, count, tp_dim = 0, 1, None
        if getattr(param, "tensor_model_parallel", False):
            group = pg_collection.expt_tp if expert else pg_collection.tp
            index, count, tp_dim = group.rank(), group.size(), param.partition_dim
        for spec in specs:
            match = _EXPERT_INDEX.search(spec.name)
            expert_index = int(match.group(1)) if expert and match else None
            views[spec.name] = _View(param, spec)
            shape = tuple(spec.selected_shape(param.shape))
            reports.append(
                (spec.name, shape, param.dtype, tp_dim, count, expert_index, index, rank, task.global_param_name)
            )
    return views, reports


def _reshard_entries(reports_by_rank: list[list[tuple]]) -> list[RefitEntry]:
    """Merge every trainer rank's reports into entries, in a rank-independent order."""
    merged: dict[str, tuple[tuple, dict[int, list[int]]]] = {}
    for reports in reports_by_rank:
        for name, *meta, index, rank, _ in reports:
            known, owners = merged.setdefault(name, (tuple(meta), {}))
            if known != tuple(meta):
                raise RuntimeError(f"Trainer ranks disagree on {name}: {known} vs {tuple(meta)}.")
            owners.setdefault(index, []).append(rank)

    entries = []
    for name, ((shape, dtype, tp_dim, count, expert), owners) in merged.items():
        replicas = {len(ranks) for ranks in owners.values()}
        if sorted(owners) != list(range(count)) or len(replicas) != 1:
            raise RuntimeError(f"{name}: shards {sorted(owners)} of {count} with replica counts {replicas}.")
        columns = [sorted(owners[s]) for s in range(count)]
        rows = [tuple(column[i] for column in columns) for i in range(replicas.pop())]
        # M2N meshes are contiguous rank runs; with TP innermost (Megatron's order) every row is one.
        rows = [row for row in rows if row == tuple(range(row[0], row[0] + len(row)))]
        if not rows:
            raise RuntimeError(f"{name}: no replica's shards sit on consecutive ranks ({columns}).")
        split = tp_dim if count > 1 else None
        if split is not None:
            shape = shape[:split] + (shape[split] * count,) + shape[split + 1 :]
        entries.append(RefitEntry(name, shape, dtype, tuple(rows), split, tp_dim, expert, reshard=True))
    return entries


def _assign_buckets(entries: list[RefitEntry], bucket_bytes: int) -> tuple[RefitEntry, ...]:
    """Cut the entries into consecutive buckets of about ``bucket_bytes``; routes never share one."""
    out, bucket, size = [], 0, 0
    for entry in entries:
        nbytes = torch.Size(entry.shape).numel() * entry.dtype.itemsize
        if out and (size + nbytes > bucket_bytes or entry.reshard != out[-1].reshard):
            bucket, size = bucket + 1, 0
        size += nbytes
        out.append(replace(entry, bucket=bucket))
    return tuple(out)


class RefitSender:
    """The trainer's end of the refit channel; built by ``TrainingEngine.connect_rollout`` on every rank.

    Construction is collective over the trainer ranks (it gathers the reshard views and runs one export of
    the collective tensors to learn their names and shapes) and then waits for every rollout rank.
    """

    def __init__(
        self,
        model: list[Any],
        export_state: Any,
        pg_collection: Any,
        *,
        address: str,
        port: int,
        layout: RolloutLayout,
        bucket_bytes: int,
        merged_adapters: bool,
        timeout_s: float,
    ) -> None:
        self._model = model
        self._bridge = export_state.bridge
        if export_state.tasks is None:
            export_state.tasks = self._bridge.get_conversion_tasks(model)
        tasks = export_state.tasks

        # LoRA weights are only correct after Bridge merges the adapters, so they all take the collective route.
        self._views, reports = ({}, []) if merged_adapters else _reshard_views(tasks, pg_collection)
        world_size = torch.distributed.get_world_size()
        reports_by_rank: list[list[tuple]] = [[] for _ in range(world_size)]
        torch.distributed.all_gather_object(reports_by_rank, reports)
        resharded = {report[-1] for reports in reports_by_rank for report in reports}
        self._collective_tasks = [task for task in tasks if task.global_param_name not in resharded]

        trainers = (tuple(range(world_size)),)
        collective = [
            RefitEntry(name, tuple(tensor.shape), tensor.dtype, trainers, None, None, None, reshard=False)
            for name, tensor in self._export()
        ]
        entries = _assign_buckets(_reshard_entries(reports_by_rank) + collective, bucket_bytes)
        num_experts = 1 + max((entry.expert for entry in entries if entry.expert is not None), default=-1)
        if layout.expert_parallel_size > 1 and num_experts % layout.expert_parallel_size:
            raise ValueError(f"{num_experts} experts do not split over {layout.expert_parallel_size} ranks.")
        plan = RefitPlan(world_size, layout, num_experts, entries)
        self._channel = _Channel.connect(address, port, torch.distributed.get_rank(), plan=plan, timeout_s=timeout_s)
        self.send(timeout_s).synchronize()  # warm-up: see _Channel.connect

    @property
    def plan(self) -> RefitPlan:
        """The plan both sides issue."""
        return self._channel.plan

    def _export(self) -> Iterator[tuple[str, torch.Tensor]]:
        if not self._collective_tasks:  # an empty task list would mean "all tasks" to Bridge
            return iter(())
        return (
            (item.param_name, item.weight)
            for item in self._bridge.export_hf_weights(
                self._model,
                cpu=False,
                show_progress=False,
                conversion_tasks=self._collective_tasks,
                merge_adapter_weights=True,
            )
        )

    def send(self, timeout_s: float | None = None) -> RefitHandle:
        """Enqueue one refit of the current weights; collective over the trainer and the rollout.

        The reshard buckets read the parameters in place, so the weights must not change until the
        returned handle completes (``TrainingEngine`` orders its updates after it). ``timeout_s`` bounds
        the whole refit, issue and transfer; see ``RefitHandle``.
        """
        channel = self._channel
        with channel.guarded(timeout_s) as watchdog:
            self._issue()
        return channel.fence(watchdog)

    def _issue(self) -> None:
        channel, current = self._channel, torch.cuda.current_stream()
        exported = self._export()
        for bucket, atoms in enumerate(channel.buckets):
            entries = list(dict.fromkeys(entry for entry, _, _ in atoms))
            buffers: dict[str, torch.Tensor] = {}
            for entry in entries:
                if entry.reshard:
                    if entry.name in self._views:
                        buffers[entry.name] = self._views[entry.name].tensor()
                    continue
                name, tensor = next(exported)
                if name != entry.name or tuple(tensor.shape) != entry.shape or tensor.dtype != entry.dtype:
                    raise RuntimeError(f"Export produced {name} {tuple(tensor.shape)}, the plan expects {entry}.")
                buffers[name] = tensor.contiguous()
            channel.stream.wait_stream(current)
            channel.issue(bucket, buffers, {})
            for tensor in buffers.values():
                tensor.record_stream(channel.stream)
        if next(exported, None) is not None:
            raise RuntimeError("Export produced more tensors than the plan holds.")

    def close(self) -> None:
        """Destroy the communicator."""
        self._channel.close()


# ---- rollout side ------------------------------------------------------------------------------------

Destination = Callable[[RefitEntry, tuple[int, ...]], "torch.Tensor | None"]


class RefitReceiver:
    """One rollout rank's end of the refit channel; construct it on every rollout rank.

    Construction blocks until the trainer has connected (``TrainingEngine.connect_rollout``) and every
    rollout rank has joined the communicator, then drains one warm-up refit (current trainer weights,
    discarded).

    Args:
        trainer_addr: Rendezvous host, the trainer's rank 0, as given to ``connect_rollout``.
        trainer_port: Rendezvous port, as given to ``connect_rollout``.
        rank: This rank's index among the rollout ranks, ``0 <= rank < layout.world_size``.
        layout: The rollout layout; must equal the trainer's.
        timeout_s: Rendezvous timeout.
    """

    def __init__(
        self, trainer_addr: str, trainer_port: int, rank: int, layout: RolloutLayout, *, timeout_s: float = 1800
    ) -> None:
        self._channel = _Channel.connect(trainer_addr, trainer_port, rank, plan=None, timeout_s=timeout_s)
        if self.plan.layout != layout:
            self.close()
            raise ValueError(f"The trainer planned for {self.plan.layout}; this rollout is {layout}.")
        # Warm-up (see _Channel.connect). Holding every staged tensor until the end also grows the
        # caching allocator to a full refit, so the first real receive does not pay for fresh cudaMalloc.
        warm_up = list(self.receive(timeout_s=timeout_s))
        self.synchronize()
        del warm_up

    @property
    def plan(self) -> RefitPlan:
        """The plan both sides issue."""
        return self._channel.plan

    def local_entries(self) -> list[tuple[RefitEntry, tuple[int, ...]]]:
        """Entries this rank receives, with the local shape each arrives in (its TP shard, or whole)."""
        return [(entry, dst.shape) for atoms in self._channel.buckets for entry, _, dst in atoms if dst.member]

    def receive(
        self, destination: Destination | None = None, *, timeout_s: float | None = None
    ) -> Iterator[tuple[RefitEntry, torch.Tensor]]:
        """Receive one refit; every rollout rank calls it once per trainer ``refit()`` and drains it.

        Bucket by bucket, it enqueues the transfer on the refit stream, orders the current stream after it,
        and yields the bucket's staged tensors, so the iterator can go straight to the engine's weight
        loader (vLLM: ``model_runner.reload_weights(weights_iterator=...)``). Once it is exhausted the
        current stream is ordered after the whole refit.

        Args:
            destination: ``destination(entry, local_shape)`` returns a contiguous tensor of that shape and
                ``entry.dtype`` to receive into in place (a view of the engine's own weight, never
                yielded), or ``None`` to receive into a staging buffer.
            timeout_s: Bound on the whole receive, including the caller's work between yields; when it is
                missed the channel is aborted and ``synchronize`` (or the next issue) raises ``TimeoutError``.

        Yields:
            ``(entry, tensor)`` for each staged tensor, in its local shape (its TP shard, or whole). Staging
            buffers come from the caching allocator, so a caller that keeps them across refits makes every
            receive grow the pool with fresh ``cudaMalloc`` calls; load and drop them, or pass ``destination``.
        """
        channel, current = self._channel, torch.cuda.current_stream()
        with channel.guarded(timeout_s) as watchdog:
            for bucket, atoms in enumerate(channel.buckets):
                buffers: dict[str, torch.Tensor] = {}
                staged: list[tuple[RefitEntry, torch.Tensor]] = []
                for entry, _, dst in atoms:
                    if not dst.member:
                        continue
                    buffer = destination(entry, dst.shape) if destination is not None else None
                    if buffer is None:
                        buffer = torch.empty(dst.shape, dtype=entry.dtype, device="cuda")
                        staged.append((entry, buffer))
                    buffers[entry.name] = buffer
                channel.stream.wait_stream(current)
                channel.issue(bucket, {}, buffers)
                current.wait_stream(channel.stream)
                yield from staged
        self._last = channel.fence(watchdog)

    def synchronize(self) -> None:
        """Block the host until the last drained ``receive`` has landed; see ``RefitHandle``."""
        self._last.synchronize()

    def close(self) -> None:
        """Destroy the communicator."""
        self._channel.close()
