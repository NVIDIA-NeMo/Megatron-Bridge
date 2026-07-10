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

This is **layer 2** of the design in
``scripts/performance/perf_leaderboard/design_determinism_debug_tool.md``: a
``TorchDispatchMode`` that fingerprints the *output* of every ATen op and writes it
to the same ordered per-rank stream that ``collective_trace`` writes to. Together they
form one totally-ordered stream of (ops + collectives) per logical rank.

Why op-level matters: non-determinism that originates in a *compute* kernel
(fused cross-entropy backward, ``scatter_add`` in embedding-grad, flash-attn backward)
is invisible to ``nn.Module`` hooks and to the collective tracer — the collective
tracer only sees it *downstream*, when the bad gradient reaches a collective. The op
tracer names the exact ``aten`` op.

Root-cause property: in a single totally-ordered stream, **the first op whose output
diverges is the root cause** — everything earlier matched, so this op's inputs
matched and the op itself introduced the divergence. Output-only fingerprints are
therefore sufficient (no need to fingerprint inputs).

Cost: fingerprinting every ATen op is expensive (CPU copy + crc per op), so this is a
separate opt-in layer (``DET_TRACE_OPS``). Use it scoped to the suspect iteration after
the cheap collective trace has narrowed the region.

Usage::

    from megatron.bridge.training.utils.determinism import collective_trace as ct, op_trace
    ct.enable(out_dir=...)   # opens the shared stream
    op_trace.enable()        # start fingerprinting ATen ops into the same stream
    ct.set_active(True, window=step)
    ...                      # forward/backward
    ct.set_active(False)
    op_trace.disable()
    ct.disable()

Limitations (known; see design_determinism_debug_tool.md §10):
- **Incompatible with CUDA-graph capture.** This mode runs Python per ATen op, which breaks
  ``torch.cuda.graph`` / TE graphed-callables. Do not set ``DET_TRACE_OPS`` on a run with CUDA
  graphs enabled (capture will error). The collective-only layer is graph-safe.
- **Do not combine with ``determinism_debug_enabled`` (the Class-A replay path).** That path
  reruns the step N times without re-arming capture, so per-replay records would accumulate
  under one window. Use one debug mode at a time.
- ``collective_trace.enable(force_sync=True)`` (the default) is required for correctness; with
  ``force_sync=False`` async collective outputs are fingerprinted before completion.
- ``"empty"``/``"c10d"`` are substring skips; they reliably catch the empty-family and c10d
  collective ops in practice but are not a formal op-class check.
"""

import logging
from typing import Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from megatron.bridge.training.utils.determinism import collective_trace as ct
from megatron.bridge.training.utils.determinism.signature import tensor_signature


logger = logging.getLogger(__name__)


def _fingerprint_outputs(out: object) -> list:
    """Signatures of ALL non-empty tensor outputs of an op (full comparison).

    Every dtype is fingerprinted, not just float: integer/bool divergences (routing
    indices, masks, argmax results, counts) are real non-determinism and must not be
    dropped. Empty tensors are skipped (no content to compare). This is heavier but is
    a complete comparison.
    """
    sigs: list = []

    def add(x: object) -> None:
        if isinstance(x, torch.Tensor) and x.numel() > 0:
            sigs.append(tensor_signature(x))

    if isinstance(out, torch.Tensor):
        add(out)
    elif isinstance(out, (list, tuple)):
        for x in out:
            add(x)
    return sigs


class OpTraceMode(TorchDispatchMode):
    """Fingerprint every ATen op's output into the shared collective_trace stream.

    Honors ``collective_trace`` activation state (``_S.enabled``/``_S.active``) so the
    same ``set_active(window=...)`` toggles both layers. Signature computation runs
    while this mode is popped off the dispatch stack (standard ``TorchDispatchMode``
    behavior), so it does not recurse.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        # Skip when suspended: collective_trace sets _S.suspend while computing its own
        # signatures, so those tracer-internal ops are not recorded as 'aten' events
        # (recording them would desync the two jobs' streams — see _S.suspend).
        # Also skip the empty-family ops: torch.empty/empty_like/... return UNINITIALIZED
        # memory whose contents differ run-to-run by definition (and use_deterministic_
        # algorithms fills it, so they'd diverge only on the nondet side) — fingerprinting
        # them yields a spurious "root cause" that is just uninitialized scratch, not the
        # first op that computes a divergent value.
        if ct._S.enabled and ct._S.active and not ct._S.suspend:
            name = str(func)
            # Skip two op classes that produce spurious divergences at the op level:
            #   - "empty": uninitialized memory (contents differ run-to-run by definition).
            #   - "c10d": collective ATen ops. These are the collective-tracer's job — it
            #     fingerprints them AFTER waiting (_maybe_sync). op_trace sees the async op
            #     return immediately, so it would fingerprint an in-flight/garbage buffer.
            #     They also reach here via Megatron paths that bypass the dist.* wrappers
            #     (so _S.suspend is not set), which is why the wrapper-side guard alone is
            #     insufficient. Collectives belong to the collective layer, not the op layer.
            if "empty" not in name and "c10d" not in name:
                prev = ct._S.suspend
                ct._S.suspend = True
                try:
                    sigs = _fingerprint_outputs(out)
                    if sigs:
                        ct.record_event(name, "aten", [], sigs)
                finally:
                    ct._S.suspend = prev
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
    logger.info("op_trace enabled: fingerprinting ATen op outputs into the shared stream")


def disable() -> None:
    """Remove the op-trace dispatch mode."""
    global _active_mode
    if _active_mode is None:
        return
    _active_mode.__exit__(None, None, None)
    _active_mode = None
    logger.info("op_trace disabled")
