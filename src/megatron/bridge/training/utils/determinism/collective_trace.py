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
import traceback
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist

from megatron.bridge.training.utils.determinism.signature import signature_to_jsonable, tensor_signature


logger = logging.getLogger(__name__)


@dataclass
class _TraceState:
    """Module-level tracer state."""

    enabled: bool = False
    active: bool = False
    force_sync: bool = True
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
    """Cheap call-site tag (first frame outside torch/this file) to aid attribution."""
    for frame in traceback.extract_stack()[::-1]:
        fn = frame.filename
        if "/torch/" in fn or fn.endswith("collective_trace.py") or fn.endswith("op_trace.py"):
            continue
        # keep last two path components for brevity
        short = "/".join(fn.split("/")[-2:])
        return f"{short}:{frame.lineno}"
    return "?"


# ---------------------------------------------------------------------------
# Record emission
# ---------------------------------------------------------------------------
def record_event(op_name: str, group_name: str, input_sigs, output_sigs) -> None:
    """Write one ordered stream record (JSONL) using an already-resolved group name.

    Shared by the collective wrappers (group resolved from the process group) and the
    op-level tracer (``group_name="aten"``). Records interleave in one ordered stream
    per rank; the diff finds the first divergent ``output`` — which, in execution
    order, is the root cause (everything before it, including this record's inputs,
    matched).

    Args:
        op_name: Operation name (collective name or ``str(aten_func)``).
        group_name: Resolved logical group label (e.g. ``"tp"``, ``"dp"``, ``"aten"``).
        input_sigs: List of input signatures (may be empty for op records).
        output_sigs: List of output signatures.
    """
    ckey = (group_name, op_name)
    align_idx = _S.op_counter.get(ckey, 0)
    _S.op_counter[ckey] = align_idx + 1

    record = {
        "seq_id": _S.seq_id,
        "window": _S.window,
        "op": op_name,
        "group": group_name,
        # (window, group, op, align_idx) is the cross-process alignment key
        "align_idx": align_idx,
        # enclosing layer/module (per-layer attribution); None outside any module scope
        "scope": _S.scope_stack[-1] if _S.scope_stack else None,
        "caller": _caller_tag(),
        "input": [signature_to_jsonable(s) for s in input_sigs],
        "output": [signature_to_jsonable(s) for s in output_sigs],
    }
    _S.seq_id += 1
    if _S.fh is not None:
        _S.fh.write(json.dumps(record) + "\n")
        # Flush per record: a determinism debug run may end in a hard crash (NCCL abort),
        # and a buffered tail would lose exactly the trace the operator launched for.
        _S.fh.flush()


def _emit(op_name: str, group, input_sigs, output_sigs) -> None:
    """Resolve the process group to a logical name and write a stream record."""
    record_event(op_name, _group_name(group), input_sigs, output_sigs)


def _sig_list(x):
    """Signature(s) for a tensor or a list of tensors.

    Sets ``_S.suspend`` so that, when op_trace is also active, the tensor ops this
    signature computation performs are NOT recorded as 'aten' events (they would
    desync the two jobs' streams — see ``_S.suspend``).
    """
    with _suspended():
        if isinstance(x, (list, tuple)):
            return [tensor_signature(t) for t in x]
        return [tensor_signature(x)]


@contextlib.contextmanager
def _suspended():
    """Suspend op-level recording for the body (nesting-safe).

    Used around the collective wrappers' internal work — the input clone, the actual
    ``orig()`` collective (which dispatches as a C10d ATen op and, being async, would be
    fingerprinted mid-flight → garbage), and the signature computation. None of these
    are model ops; the collective is represented by its own record from ``_emit``.
    """
    prev = _S.suspend
    _S.suspend = True
    try:
        yield
    finally:
        _S.suspend = prev


# ---------------------------------------------------------------------------
# Wrappers — capture input (clone in-place sources first), call real, capture output
# ---------------------------------------------------------------------------
def _maybe_sync(work):
    """For debug runs, wait on async work so the captured output is complete.

    Preserves *values* (what determinism cares about) at the cost of overlap timing.
    The returned work is handed back to the caller; a second .wait() is a no-op.
    """
    if _S.force_sync and work is not None and hasattr(work, "wait"):
        # Let wait() exceptions propagate: a faulting collective during the debug
        # window is exactly what the operator wants to see — swallowing it would
        # record a garbage output buffer and mask the failure.
        work.wait()
    return work


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
        if not (_S.enabled and _S.active):
            return orig(tensor, *args, **kwargs)
        group = _extract_group(args, kwargs)
        with _suspended():
            # all_reduce is in-place — clone the input for its signature. Guard the clone:
            # it is an extra full-size allocation present only in the debug window, and on a
            # memory-tight run it could OOM. If it does, skip the input signature but STILL
            # issue the collective — otherwise this rank would never all_reduce while peers
            # do, hanging the job (the debug tool must not cause the hang it diagnoses).
            try:
                in_sig = _sig_list(tensor.clone())
            except torch.cuda.OutOfMemoryError:
                in_sig = [None]
            work = orig(tensor, *args, **kwargs)
            _maybe_sync(work)
            _emit("all_reduce", group, in_sig, _sig_list(tensor))
        return work

    return wrapper


def _wrap_out_in(op_name):
    """Factory for collectives with signature ``(output, input, ...)`` (out separate)."""

    def factory(orig):
        def wrapper(output, input, *args, **kwargs):
            if not (_S.enabled and _S.active):
                return orig(output, input, *args, **kwargs)
            group = _extract_group(args, kwargs)
            with _suspended():
                in_sig = _sig_list(input)
                work = orig(output, input, *args, **kwargs)
                _maybe_sync(work)
                _emit(op_name, group, in_sig, _sig_list(output))
            return work

        return wrapper

    return factory


def _wrap_all_to_all(orig):
    def wrapper(output_tensor_list, input_tensor_list, *args, **kwargs):
        if not (_S.enabled and _S.active):
            return orig(output_tensor_list, input_tensor_list, *args, **kwargs)
        group = _extract_group(args, kwargs)
        with _suspended():
            in_sig = _sig_list(input_tensor_list)
            work = orig(output_tensor_list, input_tensor_list, *args, **kwargs)
            _maybe_sync(work)
            _emit("all_to_all", group, in_sig, _sig_list(output_tensor_list))
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
def enable(out_dir: str, force_sync: bool = True) -> None:
    """Install collective wrappers and open this rank's stream file.

    Args:
        out_dir: Directory for per-rank ``.fp`` JSONL streams (shared FS for multi-node).
        force_sync: Wait on async collectives so captured outputs are complete.
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
    _S.force_sync = force_sync
    _S.out_dir = out_dir
    _S.fh = fh
    _S.seq_id = 0
    _S.op_counter.clear()
    _S.group_names.clear()
    _S.fallback_names.clear()
    _S.fallback_size_counter.clear()
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
