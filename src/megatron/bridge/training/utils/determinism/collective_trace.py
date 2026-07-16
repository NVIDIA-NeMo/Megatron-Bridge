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

"""Collective-call tracer for cross-process determinism debugging (prototype).

This is **layer 3** of the design in
``scripts/performance/perf_leaderboard/design_determinism_debug_tool.md``: it
monkeypatches the ``torch.distributed`` reduction/exchange collectives to record a
stable signature of each call's **input and output**, writing an ordered fingerprint
stream to a per-*logical-rank* file.

Why this is the highest-value layer for the 48-node bug: the 48n divergence shows up
as runs that are each internally reproducible yet disagree with one another — the
signature of topology-dependent FP reduction order. A collective whose *input
signatures match* across two jobs but whose *output signature differs* is the smoking
gun. ``nn.Module`` hooks cannot see these (the MoE all-to-all and DP reduce-scatter are
not modules).

Usage (in the training script, around the iters you want to capture)::

    from megatron.bridge.training.utils.determinism import collective_trace as ct
    ct.enable(out_dir="/lustre/.../det_streams/job_2132936")
    ...
    ct.set_active(True)    # start of the iteration window
    train_step(...)
    ct.set_active(False)   # end of the window
    ct.disable()           # flush + restore original collectives

Then offline-diff two jobs' streams with ``diff_streams.py``.
"""

import contextlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist

from megatron.bridge.training.utils.determinism.signature import finalize_staged, stage_tensor


logger = logging.getLogger(__name__)


@dataclass
class _TraceState:
    """Module-level tracer state."""

    enabled: bool = False
    active: bool = False
    out_dir: Optional[str] = None
    fh: Optional[object] = None  # open file handle for this rank's stream
    window: Optional[int] = None  # current window id (e.g. training step) for alignment
    # When True, op_trace must NOT record: we are inside the tracer's own signature
    # computation (clone/.cpu/.to/.sum). Without this guard those internal ops would be
    # captured as 'aten' records, and because their count is dtype/device-dependent the
    # two jobs' streams desync (align_idx shift) → spurious "root cause".
    suspend: bool = False
    seq_id: int = 0
    # module-scope stack (layer attribution): pushed/popped by module_scope hooks so each
    # record carries the enclosing layer (e.g. "decoder.layers.7.self_attention[bwd]").
    scope_stack: list = field(default_factory=list)
    # alignment counter per (group_name, op_name) — survives physical-rank reshuffle
    op_counter: dict = field(default_factory=dict)
    # id(process_group) -> logical group name, resolved lazily after init
    group_names: dict = field(default_factory=dict)
    # cross-process-stable fallback names for groups parallel_state can't name:
    # id(group) -> "grp_sz{size}_{n}" where n is the deterministic first-use index among
    # groups of that size. Program order is identical across two jobs of the same config,
    # so the SAME logical group gets the SAME label in both -> the diff can align it
    # (the old id-based fallback used a per-process address and silently desynced).
    fallback_names: dict = field(default_factory=dict)
    fallback_size_counter: dict = field(default_factory=dict)
    originals: dict = field(default_factory=dict)
    # extra (module, attr, original) patches for import-time-captured collective refs
    # (e.g. Megatron mappings.dist_reduce_scatter_func) that bypass the dist.* monkeypatch.
    extra_patches: list = field(default_factory=list)
    # Deferred (HybridEP-safe) records: each carries GPU scalar reductions instead of host
    # values. Finalized (.item()'d) and written at the step boundary by flush_pending().
    pending: list = field(default_factory=list)


_S = _TraceState()


# ---------------------------------------------------------------------------
# Logical coordinates + group naming (align by *logical* coords, not phys rank)
# ---------------------------------------------------------------------------
def _safe(fn, default=0):
    """Call a parallel_state getter, returning ``default`` on any failure."""
    try:
        return fn()
    except Exception:
        return default


def _logical_coords() -> dict:
    """Return logical parallel coordinates of this GPU.

    ``(pp, tp, cp, dp)`` is the *minimal complete unique* coordinate of a physical
    GPU; the expert coords ``(ep, etp, edp)`` are derived re-groupings of the same
    GPU (redundant for identity, kept for readability and group attribution).
    Physical ``global_rank`` is recorded for reference only — alignment must NOT
    use it, since the same logical GPU lands on different physical nodes per job.
    """
    coords = {"global_rank": dist.get_rank() if dist.is_initialized() else 0}
    try:
        from megatron.core import parallel_state as ps

        coords["tp"] = _safe(ps.get_tensor_model_parallel_rank)
        coords["pp"] = _safe(ps.get_pipeline_model_parallel_rank)
        coords["cp"] = _safe(ps.get_context_parallel_rank)
        coords["dp"] = _safe(ps.get_data_parallel_rank)
        coords["ep"] = _safe(ps.get_expert_model_parallel_rank)
        coords["etp"] = _safe(ps.get_expert_tensor_parallel_rank)
        # No direct edp rank getter — derive from the expert-data-parallel group.
        coords["edp"] = _safe(lambda: dist.get_rank(group=ps.get_expert_data_parallel_group()))
    except Exception:
        coords.update(tp=0, pp=0, cp=0, dp=0, ep=0, etp=0, edp=0)
    return coords


def _parallel_config() -> dict:
    """Capture the parallel-config *sizes* so the diff can refuse cross-config compares.

    Comparing two jobs is only valid when these match (same rank decomposition and,
    with a seeded loader, same data shard per ``dp`` rank). Differing sizes (e.g.
    24n vs 48n, where ``dp`` differs) make same-coord comparison meaningless.
    """
    cfg = {"world_size": dist.get_world_size() if dist.is_initialized() else 1}
    try:
        from megatron.core import parallel_state as ps

        cfg["tp"] = _safe(ps.get_tensor_model_parallel_world_size, None)
        cfg["pp"] = _safe(ps.get_pipeline_model_parallel_world_size, None)
        cfg["cp"] = _safe(ps.get_context_parallel_world_size, None)
        cfg["dp"] = _safe(ps.get_data_parallel_world_size, None)
        cfg["ep"] = _safe(ps.get_expert_model_parallel_world_size, None)
        cfg["etp"] = _safe(ps.get_expert_tensor_parallel_world_size, None)
    except Exception:
        pass
    return cfg


def _build_group_names() -> dict:
    """Map id(process_group) -> a stable logical name (dp/tp/pp/ep/...)."""
    names: dict = {}
    try:
        from megatron.core import parallel_state as ps

        candidates = {
            "tp": ps.get_tensor_model_parallel_group,
            "pp": ps.get_pipeline_model_parallel_group,
            "dp": lambda: ps.get_data_parallel_group(with_context_parallel=False),
            "dp_cp": lambda: ps.get_data_parallel_group(with_context_parallel=True),
            "ep": ps.get_expert_model_parallel_group,
            "etp": ps.get_expert_tensor_parallel_group,
            "edp": ps.get_expert_data_parallel_group,
            "cp": ps.get_context_parallel_group,
            # broader coverage so DSV3 dist-optimizer / embedding groups get real names
            # instead of the fallback (these are the unnamed all_gathers seen at 48n):
            "mp": getattr(ps, "get_model_parallel_group", None),
            "tp_cp": getattr(ps, "get_tensor_and_context_parallel_group", None),
            "tp_dp": getattr(ps, "get_tensor_and_data_parallel_group", None),
            "embd": getattr(ps, "get_embedding_group", None),
            "pos_embd": getattr(ps, "get_position_embedding_group", None),
            "intra_dpopt": getattr(ps, "get_intra_distributed_optimizer_instance_group", None),
            "inter_dpopt": getattr(ps, "get_inter_distributed_optimizer_instance_group", None),
            "etp_ep": getattr(ps, "get_expert_tensor_and_model_parallel_group", None),
        }
        candidates = {k: v for k, v in candidates.items() if v is not None}
        for name, getter in candidates.items():
            try:
                g = getter()
                if g is not None:
                    names[id(g)] = name
            except Exception:
                continue
    except Exception:
        pass
    return names


def _group_name(group) -> str:
    """Resolve a process group to a cross-process-stable logical name.

    Named parallel_state groups (tp/dp/ep/...) resolve directly. Anything else gets a
    stable fallback ``grp_sz{size}_{n}`` (size + first-use index) rather than the old
    per-process id hash — so the same logical group aligns across two jobs in the diff.
    """
    if group is None:
        return "world"
    if not _S.group_names:
        _S.group_names = _build_group_names()
    named = _S.group_names.get(id(group))
    if named is not None:
        return named
    gid = id(group)
    name = _S.fallback_names.get(gid)
    if name is None:
        try:
            size = dist.get_world_size(group=group)
        except Exception:
            size = -1
        n = _S.fallback_size_counter.get(size, 0)
        _S.fallback_size_counter[size] = n + 1
        name = f"grp_sz{size}_{n}"
        _S.fallback_names[gid] = name
    return name


def _caller_tag() -> str:
    """Cheap call-site tag (first frame outside torch/this file) to aid attribution.

    Walks frames directly rather than ``traceback.extract_stack()``, which materializes
    every frame's source line via linecache — wasted work multiplied by ~500K ops/iter at
    op level, when only ``filename:lineno`` is used.
    """
    f = sys._getframe(1)
    while f is not None:
        fn = f.f_code.co_filename
        if "/torch/" not in fn and not fn.endswith(("collective_trace.py", "op_trace.py")):
            # keep last two path components for brevity
            return f"{'/'.join(fn.split('/')[-2:])}:{f.f_lineno}"
        f = f.f_back
    return "?"


# ---------------------------------------------------------------------------
# Deferred, HybridEP-safe records
#
# The old path did tensor.cpu()/.item() + work.wait() INSIDE the iteration to fingerprint
# each collective. Those host<->device syncs stall a rank while HybridEP's persistent
# all-gather kernels poll peers -> peers time out ("expecting N got N-2"). Instead we:
#   1. stage an async GPU reduction — torch.hash_tensor's uint64 result, no .item()
#      (signature.stage_tensor) — keeping only a tiny GPU scalar, no host copy.
#   2. stash the record (with those GPU scalars), reserving its alignment key in order.
#   3. at the STEP boundary (set_active(False)/disable, after the collectives + HybridEP
#      kernels retire) .item() the scalars and write the JSONL (flush_pending) — safe to sync.
# Correctness needs the traced collectives to be synchronous, so run with comm overlap OFF
# (an overlapped async collective's output is not complete at stage time).
# ---------------------------------------------------------------------------
def _staged_sig_list(x):
    """Staged signature(s) for a tensor or list of tensors (async GPU reductions, no sync).

    Runs under ``_suspended`` so that, with op_trace active, the reduction's own tensor ops
    are NOT recorded as 'aten' events (their dtype/device-dependent count would desync the
    two jobs' streams — see ``_S.suspend``).
    """
    with _suspended():
        if isinstance(x, (list, tuple)):
            return [stage_tensor(t) for t in x]
        return [stage_tensor(x)]


def _stash_named(op_name: str, group_name: str, input_staged, output_staged) -> None:
    """Reserve the alignment key now (execution order) and stash a pending record.

    ``group_name`` is an already-resolved logical label — ``"tp"``/``"dp"``/... for
    collectives, ``"aten"`` for op records. Both layers stash into the same ordered
    ``_S.pending`` (shared ``seq_id``), so ops and collectives interleave in one per-rank
    stream and flush together at the step boundary.
    """
    ckey = (group_name, op_name)
    align_idx = _S.op_counter.get(ckey, 0)
    _S.op_counter[ckey] = align_idx + 1
    _S.pending.append(
        {
            "seq_id": _S.seq_id,
            "window": _S.window,
            "op": op_name,
            "group": group_name,
            # (window, group, op, align_idx) is the cross-process alignment key
            "align_idx": align_idx,
            # enclosing layer/module (per-layer attribution); None outside any module scope
            "scope": _S.scope_stack[-1] if _S.scope_stack else None,
            "caller": _caller_tag(),
            "input": input_staged,
            "output": output_staged,
        }
    )
    _S.seq_id += 1


def _stash(op_name: str, group, input_staged, output_staged) -> None:
    """Stash a collective record, resolving the process group to a logical name."""
    _stash_named(op_name, _group_name(group), input_staged, output_staged)


def flush_pending() -> None:
    """Finalize (``.item()`` the staged GPU scalars) and write all stashed records.

    Call at the step boundary (``set_active(False)``/``disable``), where a device->host
    sync is safe — by then the iteration's collectives and HybridEP kernels have retired.
    """
    if not _S.pending:
        return
    for rec in _S.pending:
        rec["input"] = [finalize_staged(s) for s in rec["input"]]
        rec["output"] = [finalize_staged(s) for s in rec["output"]]
        if _S.fh is not None:
            _S.fh.write(json.dumps(rec) + "\n")
    if _S.fh is not None:
        # Flush once per window: a debug run may end in a hard crash (NCCL abort); a
        # buffered tail would lose exactly the window the operator launched for.
        _S.fh.flush()
    _S.pending.clear()


@contextlib.contextmanager
def _suspended():
    """Suspend op-level recording for the body (nesting-safe).

    Used around the collective wrappers' internal work — the input clone, the actual
    ``orig()`` collective (which dispatches as a C10d ATen op and, being async, would be
    fingerprinted mid-flight → garbage), and the signature computation. None of these
    are model ops; the collective is represented by its own record from ``_stash``.
    """
    prev = _S.suspend
    _S.suspend = True
    try:
        yield
    finally:
        _S.suspend = prev


# ---------------------------------------------------------------------------
# Wrappers — stage async GPU reductions (no mid-iteration sync), stash for drain
# ---------------------------------------------------------------------------
def _extract_group(args, kwargs):
    """Find the process group whether passed by keyword or positionally.

    The positional index of ``group`` differs per collective (all_reduce, all_to_all_single,
    reduce_scatter_tensor, ...), so scan args for a ProcessGroup rather than assuming an
    index — otherwise a positional group is missed and the record is mislabeled 'world'.
    """
    g = kwargs.get("group", None)
    if g is not None:
        return g
    for a in args:
        if isinstance(a, dist.ProcessGroup):
            return a
    return None


def _wrap_all_reduce(orig):
    def wrapper(tensor, *args, **kwargs):
        # Skip when suspended: a deprecated alias (e.g. _reduce_scatter_base) delegates to a
        # public name we also patched; running orig() under _suspended (below) makes that
        # nested wrapper hit this guard and NOT re-record — one physical collective, one record.
        if not (_S.enabled and _S.active) or _S.suspend:
            return orig(tensor, *args, **kwargs)
        group = _extract_group(args, kwargs)
        # all_reduce is IN-PLACE: the collective overwrites ``tensor`` (possibly on the NCCL
        # stream), which would race a reduction reading the same buffer. Clone the input
        # first (GPU-async, no host sync) and stage the clone. Guard the clone: if it OOMs,
        # skip the input sig but STILL issue the collective — never hang peers. Clone under
        # _suspended so op_trace (if active) doesn't fingerprint this tracer-internal clone.
        with _suspended():
            try:
                cloned = tensor.clone()
            except torch.cuda.OutOfMemoryError:
                cloned = None
        in_sig = _staged_sig_list(cloned) if cloned is not None else [None]
        with _suspended():
            work = orig(tensor, *args, **kwargs)
        _stash("all_reduce", group, in_sig, _staged_sig_list(tensor))
        return work

    return wrapper


def _wrap_out_in(op_name):
    """Factory for collectives with signature ``(output, input, ...)`` (out separate)."""

    def factory(orig):
        def wrapper(output, input, *args, **kwargs):
            # Skip when suspended so a deprecated alias delegating to a public name we also
            # patched (e.g. _reduce_scatter_base -> reduce_scatter_tensor) does not double-record.
            if not (_S.enabled and _S.active) or _S.suspend:
                return orig(output, input, *args, **kwargs)
            group = _extract_group(args, kwargs)
            in_sig = _staged_sig_list(input)
            with _suspended():
                work = orig(output, input, *args, **kwargs)
            # The output reduction is enqueued after the (synchronous) collective, so it
            # reads the completed result. Requires comm overlap OFF (see the deferred note).
            _stash(op_name, group, in_sig, _staged_sig_list(output))
            return work

        return wrapper

    return factory


def _wrap_all_to_all(orig):
    def wrapper(output_tensor_list, input_tensor_list, *args, **kwargs):
        if not (_S.enabled and _S.active) or _S.suspend:
            return orig(output_tensor_list, input_tensor_list, *args, **kwargs)
        group = _extract_group(args, kwargs)
        in_sig = _staged_sig_list(input_tensor_list)
        with _suspended():
            work = orig(output_tensor_list, input_tensor_list, *args, **kwargs)
        _stash("all_to_all", group, in_sig, _staged_sig_list(output_tensor_list))
        return work

    return wrapper


# (real_attr_name, wrapper_factory)
_TARGETS = [
    ("all_reduce", _wrap_all_reduce),
    ("reduce_scatter_tensor", _wrap_out_in("reduce_scatter_tensor")),
    ("_reduce_scatter_base", _wrap_out_in("_reduce_scatter_base")),
    ("all_gather_into_tensor", _wrap_out_in("all_gather_into_tensor")),
    ("_all_gather_base", _wrap_out_in("_all_gather_base")),
    ("all_to_all_single", _wrap_out_in("all_to_all_single")),
    ("all_to_all", _wrap_all_to_all),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def enable(out_dir: str) -> None:
    """Install collective wrappers and open this rank's stream file.

    Signatures are staged as async GPU reductions and drained (``.item()`` + written) at the
    step boundary — no mid-iteration host sync — so this is HybridEP-safe. Correctness
    requires the traced collectives to be synchronous: run with comm overlap OFF.

    Args:
        out_dir: Directory for per-rank ``.fp`` JSONL streams (shared FS for multi-node).
    """
    if _S.enabled:
        logger.warning("collective_trace already enabled; ignoring re-enable")
        return

    coords = _logical_coords()
    config = _parallel_config()
    os.makedirs(out_dir, exist_ok=True)
    # File name keyed by the MINIMAL COMPLETE unique GPU coordinate (pp, tp, cp, dp)
    # so two jobs' files line up regardless of physical node placement. ep/etp/edp
    # are derived from these and live in the header, not the name. dp is in the key
    # because data (and thus the whole stream) differs per dp rank.
    fname = "stream_pp{pp}_tp{tp}_cp{cp}_dp{dp}.fp".format(**coords)
    fh = open(os.path.join(out_dir, fname), "w")
    fh.write(json.dumps({"_header": True, "coords": coords, "config": config}) + "\n")

    for attr, factory in _TARGETS:
        orig = getattr(dist, attr, None)
        if orig is None:
            continue
        _S.originals[attr] = orig
        setattr(dist, attr, factory(orig))

    _patch_captured_refs()

    _S.enabled = True
    _S.active = False
    _S.out_dir = out_dir
    _S.fh = fh
    _S.seq_id = 0
    _S.op_counter.clear()
    _S.group_names.clear()
    _S.fallback_names.clear()
    _S.fallback_size_counter.clear()
    _S.pending.clear()
    logger.info(
        "collective_trace enabled: wrapped %d dist collectives + %d captured refs → %s",
        len(_S.originals),
        len(_S.extra_patches),
        fname,
    )


def _patch_captured_refs() -> None:
    """Wrap Megatron-Core's IMPORT-TIME captured collective refs (sequence-parallel path).

    ``mappings.py`` does ``dist_reduce_scatter_func = torch.distributed._reduce_scatter_base``
    at import, so it holds the ORIGINAL callable and bypasses our ``dist.*`` monkeypatch.
    Wrap those module-level names directly (all have an ``(output, input, ...)`` signature)
    so SP reduce_scatter/all_gather are captured too. Best-effort + version-specific.
    """
    try:
        from megatron.core.tensor_parallel import mappings as _mappings
    except Exception:
        return
    for attr, label in (
        ("dist_reduce_scatter_func", "reduce_scatter_tensor"),
        ("dist_all_gather_func", "all_gather_into_tensor"),
    ):
        orig = getattr(_mappings, attr, None)
        if orig is None or getattr(orig, "_det_wrapped", False):
            continue
        wrapped = _wrap_out_in(getattr(orig, "__name__", label))(orig)
        try:
            wrapped._det_wrapped = True  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass
        _S.extra_patches.append((_mappings, attr, orig))
        setattr(_mappings, attr, wrapped)


def set_active(active: bool, window: Optional[int] = None) -> None:
    """Turn capture on/off (wrap the iteration window you care about).

    Args:
        active: Whether to capture collectives from now on.
        window: Window id (e.g. the training step) stamped on each record so the
            diff aligns by ``(window, group, op, align_idx)``. Pass the step when
            tracing multiple iterations so iter-N's ``all_reduce#0`` does not
            collide with iter-(N+1)'s in the alignment key.
    """
    if not active:
        # Drain the window we just finished: the iteration's collectives and HybridEP
        # kernels have retired, so .item()-ing the staged scalars now is safe (no hang).
        flush_pending()
    _S.active = active
    _S.window = window
    if active:
        # reset alignment counters at the start of each window so align_idx is
        # comparable across jobs that start the window at the same iteration
        _S.op_counter.clear()
        # clear any stale module scope so per-step attribution starts fresh (best-effort
        # backward hooks can leave the stack slightly unbalanced across a step boundary)
        _S.scope_stack.clear()


def disable() -> None:
    """Restore original collectives and flush/close the stream."""
    if not _S.enabled:
        return
    # Drain any records still staged (e.g. capture ended without a trailing set_active(False)).
    flush_pending()
    for attr, orig in _S.originals.items():
        setattr(dist, attr, orig)
    _S.originals.clear()
    for module, attr, orig in _S.extra_patches:
        setattr(module, attr, orig)
    _S.extra_patches.clear()
    if _S.fh is not None:
        _S.fh.flush()
        _S.fh.close()
        _S.fh = None
    _S.enabled = False
    _S.active = False
    logger.info("collective_trace disabled (%d records written)", _S.seq_id)
