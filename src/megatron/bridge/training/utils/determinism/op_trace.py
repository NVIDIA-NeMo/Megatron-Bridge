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

"""Op-level tracer for cross-process determinism debugging (prototype, layer 2).

This is **layer 2** of the design: a ``TorchDispatchMode`` that fingerprints the
*output* (and input) of every ATen op and writes it to the same ordered per-rank stream
that ``collective_trace`` writes to. Together they form one totally-ordered stream of
(ops + collectives) per logical rank.

Why op-level matters: non-determinism that originates in a *compute* kernel
(fused cross-entropy backward, ``scatter_add`` in embedding-grad, a Mamba/attention
Triton kernel) is invisible to ``nn.Module`` hooks and to the collective tracer — the
collective tracer only sees it *downstream*, when the bad gradient reaches a collective.
The op tracer names the exact ``aten`` op.

Root-cause property: in a single totally-ordered stream, **the first op whose output
diverges while its inputs matched is the root cause** — everything earlier matched, so
this op introduced the divergence. Capturing inputs *and* outputs lets the diff make that
distinction (inputs differ → look upstream; inputs match / output differs → this op).

Cost / memory: the naive approach did a full ``.cpu()`` + hash per op — tens of thousands
of host copies per iteration → host-RAM OOM and serialization. This layer uses the SAME
deferred, HybridEP-safe signature as ``collective_trace``: an async GPU reduction staged
per op and drained (``.item()``) at the step boundary. Nothing leaves the GPU except tiny
scalars, so there is no host copy and no mid-iteration sync. Opt-in (``DET_TRACE_OPS``).

Usage::

    from megatron.bridge.training.utils.determinism import collective_trace as ct, op_trace
    ct.enable(out_dir=...)   # opens the shared stream
    op_trace.enable()        # start fingerprinting ATen ops into the same stream
    ct.set_active(True, window=step)
    ...                      # forward/backward
    ct.set_active(False)     # drains the window (op + collective records)
    op_trace.disable()
    ct.disable()

Limitations:
- **Compares only runs with the SAME op stream.** ``align_idx`` counts every recorded ATen
  op, so the two jobs must dispatch the same ops in the same order. Same code + same
  dtype/backend (the intended A/A comparison, incl. cross-topology at identical config)
  gives that. Comparing runs whose op *decompositions* differ — fp8 vs bf16, or a fused
  kernel available on one path and decomposed to primitives on the other — shifts
  ``align_idx`` from that point and can flag a spurious root cause. ``diff_streams`` already
  refuses mismatched parallel configs; keep dtype/backend identical too.
- **Incompatible with CUDA-graph capture.** This mode runs Python per ATen op, which
  breaks ``torch.cuda.graph`` / TE graphed-callables. Do not set ``DET_TRACE_OPS`` on a
  run with CUDA graphs enabled. The collective-only layer is graph-safe.
- Op records are staged as async GPU reductions and finalized at the step boundary, so
  they are only valid once ``collective_trace.set_active(False)`` (or ``disable``) drains
  the window — same as the collective layer.
- ``"empty"``/``"c10d"`` are substring skips (empty-family + c10d collective ops).
"""

import logging
from typing import Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from megatron.bridge.training.utils.determinism import collective_trace as ct
from megatron.bridge.training.utils.determinism.signature import stage_tensor


logger = logging.getLogger(__name__)


def _is_inplace(func_name: str) -> bool:
    """True for ATen ops that MUTATE a buffer — in-place (``aten.add_.Tensor``) or the
    ``.out`` variants (``aten.add.out``).

    Such ops write a buffer autograd may have saved for backward; their output must be
    cloned before fingerprinting or the reduction aliases the just-mutated storage and
    trips the saved-variable version check. In-place ops have a base name ending in ``_``;
    out-variants have the overload ``out`` (the base doesn't end in ``_``, so check both).
    """
    # Normalize both renderings: 'aten.add_.Tensor' and 'aten::add_.Tensor'.
    parts = func_name.replace("::", ".").split(".")
    base = parts[1] if len(parts) >= 2 else parts[0]
    overload = parts[-1] if len(parts) >= 3 else ""
    return base.endswith("_") or overload == "out"


def _collect(obj: object, sigs: list, clone: bool = False) -> None:
    """Append a staged GPU-reduction signature for every non-empty tensor in ``obj``.

    ``clone=True`` (used for in-place op outputs) copies the tensor first so the
    fingerprint never aliases an autograd-saved buffer's storage/version counter.
    """
    if isinstance(obj, torch.Tensor):
        if obj.numel() > 0:
            t = obj.detach()
            if clone:
                try:
                    t = t.clone()
                except Exception:
                    pass
            sigs.append(stage_tensor(t))
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _collect(x, sigs, clone)


def _fingerprint_outputs(out: object, clone: bool = False) -> list:
    """Staged signatures of ALL non-empty tensor outputs of an op.

    Every dtype is fingerprinted, not just float: integer/bool divergences (routing
    indices, masks, argmax results, counts) are real non-determinism and must not be
    dropped. Empty tensors are skipped (no content to compare).
    """
    sigs: list = []
    _collect(out, sigs, clone=clone)
    return sigs


def _fingerprint_inputs(args: tuple, kwargs: dict) -> list:
    """Staged signatures of the op's INPUT tensors (positional + keyword).

    Together with the output sigs, this lets the diff classify each op:
      inputs match, output differs -> THIS op introduced the non-determinism (root cause);
      inputs differ               -> the divergence arrived from upstream (look earlier).
    Read before the op runs (see ``__torch_dispatch__``) so in-place ops' inputs are the
    pre-mutation values.
    """
    sigs: list = []
    _collect(args, sigs)
    for v in kwargs.values():
        _collect(v, sigs)
    return sigs


class OpTraceMode(TorchDispatchMode):
    """Fingerprint every ATen op's input+output into the shared collective_trace stream.

    Honors ``collective_trace`` activation state (``_S.enabled``/``_S.active``) so the same
    ``set_active(window=...)`` toggles both layers. Signature computation runs while this
    mode is popped off the dispatch stack (standard ``TorchDispatchMode`` behavior), so it
    does not recurse, and under ``no_grad`` so it never extends the autograd graph.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Record every op EXCEPT: while suspended (tracer-internal ops), "empty" family
        # (uninitialized memory -> spurious diff), and "c10d" (the collective layer's job).
        # IN-PLACE ops ARE recorded (they can be the actual non-determinism seed); their
        # output is cloned and the fingerprint runs under no_grad so it never touches an
        # autograd-saved buffer's version counter.
        S = ct._S
        record = S.enabled and S.active and not S.suspend
        name = str(func) if record else ""
        record = record and "empty" not in name and "c10d" not in name
        inplace = record and _is_inplace(name)

        in_sigs = []
        if record:
            # Fingerprint INPUTS *before* the op — required for in-place ops, whose args are
            # the buffers they mutate; reading them afterwards would capture the post-mutation
            # value and hide whether the inputs actually matched.
            with ct._suspended(), torch.no_grad():
                in_sigs = _fingerprint_inputs(args, kwargs)

        out = func(*args, **kwargs)

        if record:
            with ct._suspended(), torch.no_grad():
                # Clone in-place outputs (the mutated buffer) so the reduction never aliases
                # an autograd-saved tensor. no_grad + detach keeps it off the graph.
                out_sigs = _fingerprint_outputs(out, clone=inplace)
                if out_sigs:
                    # Deferred: stash in+out staged sigs into the shared pending stream;
                    # the diff classifies the first output divergence (inputs match -> this
                    # op is the root cause; inputs differ -> upstream, look earlier).
                    ct._stash_named(name, "aten", in_sigs, out_sigs)
        return out


_active_mode: Optional[OpTraceMode] = None


def enable() -> None:
    """Install the op-trace dispatch mode (writes into the shared stream).

    ``collective_trace.enable(out_dir=...)`` must have been called first to open the
    stream file. Activation is shared via ``collective_trace.set_active``.
    """
    global _active_mode
    if _active_mode is not None:
        logger.warning("op_trace already enabled; ignoring re-enable")
        return
    _active_mode = OpTraceMode()
    _active_mode.__enter__()
    logger.info("op_trace enabled: fingerprinting ATen op inputs+outputs into the shared stream")


def disable() -> None:
    """Remove the op-trace dispatch mode."""
    global _active_mode
    if _active_mode is None:
        return
    _active_mode.__exit__(None, None, None)
    _active_mode = None
    logger.info("op_trace disabled")
