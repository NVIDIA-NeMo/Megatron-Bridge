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

"""Opt-in diagnostic: the tokens each rank's local experts receive, per MoE layer and micro-batch.

It answers one measurement question about expert parallelism: which rank holds the most routed tokens in each MoE
layer, and does that rank change from layer to layer. It is off by default and is switched on by the environment, for
diagnostic runs only:

* ``ERNIE55_LOG_EXPERT_LOAD=1`` turns it on (``0``, ``false``, ``off``, empty or unset leave it off; other values
  are rejected).
* ``ERNIE55_EXPERT_LOAD_DIR`` (required when on) is the directory of the per-rank JSON files.
* ``ERNIE55_EXPERT_LOAD_STEPS=<start>:<end>`` records the iterations that begin with ``start`` to ``end - 1``
  completed steps (``train_state.step``), the convention of ``profiling.profile_step_start`` and
  ``profiling.profile_step_end``, so the same values select the iterations an nsys capture holds. Unset, the window
  is Bridge's profiling window when a profiler is configured; otherwise the first training micro-batch raises. The
  window holds at most :data:`MAX_WINDOW_STEPS` steps.

Off, nothing is installed: no wrapper, tensor, kernel or host synchronization, and the forward and backward are
unchanged. :func:`note_microbatch`, which the ERNIE 5.5 training step calls once per micro-batch, returns at once.

On, every :class:`~megatron.bridge.models.ernie.modeling_ernie55.moe.Ernie55MoELayer` wraps two methods of its own
instances (:func:`attach_expert_load_probe`):

* the router's ``routing``: this rank's routed count per expert, ``routing_map.sum(0)`` without padding rows, one
  reduction kept on the device;
* the token dispatcher's ``dispatch_postprocess``, when the layer has one: ``tokens_per_expert``, the tokens this
  rank's local experts receive (with any alignment padding the dispatcher adds for low-precision GEMMs; none in BF16). A host tensor
  (the all-gather and alltoall dispatchers synchronize before returning one) is read as a list, which needs no
  synchronization; a device tensor is cloned on the device (``test_ernie55_expert_load_gpu.py`` logs which kind
  HybridEP returns).

A megakernel backend (``moe_megakernel_backend``, MoK) builds no token dispatcher: its layers record ``routed`` as
soon as the router runs, with ``received`` null; the global per-expert load is the sum of ``routed`` over the ranks.

The wrappers only read, so outputs and gradients are bitwise unchanged. No host synchronization is added inside the
window: the device counts move to the host once, when the window ends. Summing ``routed`` over every rank's file
gives the global count per expert; for each local expert it must equal that expert's ``received`` count.

One record per MoE layer and training micro-batch. A training-mode forward without autograd (the first pass of
Megatron-Core's MoE recompute, ``recompute_modules`` with ``moe``, or of full recompute) is recorded and queued; the
replay with autograd in the backward pops the queue and is not recorded again, so a layer counts once per
micro-batch with or without recompute, as Megatron-Core's expert-bias token counts do. Micro-batches are numbered in
the order the training step announces them (``0, 1, ...`` per step; evaluation forwards are ignored), which supposes
one model chunk per rank (no virtual pipeline). Forwards under CUDA graph capture are not recorded.

Output: ``<dir>/expert_load_rank<global rank, 5 digits>.json``, written by every rank at the first training
micro-batch after the window, or at process exit if training ends inside it (``flush_expert_load`` writes it on
demand)::

    {"format": "ernie55_expert_load/1", "rank": 9, "world_size": 64, "window": [50, 52],
     "num_experts": 512, "topk": 8, "ep_rank": 9, "ep_size": 64, "num_local_experts": 8,
     "local_expert_indices": [72, ..., 79],
     "layers": [{"slot": 0, "layer_number": 4, "is_mtp_layer": false}, ...],
     "records": [{"step": 50, "microbatch": 0, "slot": 0, "received": [8 counts], "routed": [512 counts]}, ...]}

``slot`` numbers the MoE layers in construction order; records follow the forward execution order. The files hold
counts and layer numbers only.
"""

from __future__ import annotations

import atexit
import collections
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from megatron.core.utils import get_pg_rank, get_pg_size, log_single_rank


logger = logging.getLogger(__name__)

__all__ = [
    "ENV_DIR",
    "ENV_STEPS",
    "ENV_SWITCH",
    "FORMAT_VERSION",
    "MAX_WINDOW_STEPS",
    "ExpertLoadRecorder",
    "attach_expert_load_probe",
    "expert_load_logging_enabled",
    "expert_load_recorder",
    "flush_expert_load",
    "note_microbatch",
]

ENV_SWITCH = "ERNIE55_LOG_EXPERT_LOAD"
"""Environment switch of the diagnostic (``1`` on)."""

ENV_DIR = "ERNIE55_EXPERT_LOAD_DIR"
"""Environment variable naming the output directory (required when on)."""

ENV_STEPS = "ERNIE55_EXPERT_LOAD_STEPS"
"""Environment variable with the recorded window, ``<start>:<end>`` in completed steps."""

FORMAT_VERSION = "ernie55_expert_load/1"
"""``format`` field of the output files."""

MAX_WINDOW_STEPS = 64
"""Longest window: its device counts stay on the GPU until the window ends (about 4 KB per record)."""

_ON = ("1", "true", "on", "yes")
_OFF = ("", "0", "false", "off", "no")


def expert_load_logging_enabled() -> bool:
    """Whether ``ERNIE55_LOG_EXPERT_LOAD`` switches the diagnostic on.

    Raises:
        ValueError: If the variable holds anything but an on or off value.
    """
    value = os.environ.get(ENV_SWITCH, "").strip().lower()
    if value in _OFF:
        return False
    if value in _ON:
        return True
    raise ValueError(f"{ENV_SWITCH} must be 1 or 0 (or true/false, on/off), got {os.environ[ENV_SWITCH]!r}.")


def _parse_window(text: str, what: str) -> tuple[int, int]:
    start, sep, end = text.strip().partition(":")
    try:
        if not sep:
            raise ValueError
        window = int(start), int(end)
    except ValueError:
        raise ValueError(f"{what} must be '<start>:<end>' (completed steps), got {text!r}.") from None
    if not 0 <= window[0] < window[1]:
        raise ValueError(f"{what} must satisfy 0 <= start < end, got {text!r}.")
    if window[1] - window[0] > MAX_WINDOW_STEPS:
        raise ValueError(f"{what} spans {window[1] - window[0]} steps; at most {MAX_WINDOW_STEPS} are recorded.")
    return window


def _profiling_window(profiling: Any) -> tuple[int, int] | None:
    """Bridge's profiler window ``(profile_step_start, profile_step_end)``, if a profiler is configured."""
    if profiling is None:
        return None
    if not (getattr(profiling, "use_nsys_profiler", False) or getattr(profiling, "use_pytorch_profiler", False)):
        return None
    start, end = getattr(profiling, "profile_step_start", None), getattr(profiling, "profile_step_end", None)
    if start is None or end is None:
        return None
    return _parse_window(f"{start}:{end}", "the profiling window (profile_step_start:profile_step_end)")


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _global_rank_and_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _to_host(values: list[Any]) -> list[Any]:
    """Host lists of ``values`` (tensors, lists or None), with one copy per group of alike device tensors."""
    out = [value if value is None or isinstance(value, torch.Tensor) else list(value) for value in values]
    groups: dict[tuple, list[int]] = collections.defaultdict(list)
    for index, value in enumerate(out):
        if isinstance(value, torch.Tensor):
            groups[(value.device, value.dtype, tuple(value.shape))].append(index)
    for indices in groups.values():
        stacked = torch.stack([out[index] for index in indices]).cpu().tolist()
        for index, host in zip(indices, stacked):
            out[index] = host
    return out


@dataclass
class _LayerProbe:
    """Per-layer state: the layer, its slot, the pass in flight and the queue of forwards awaiting their replay."""

    layer: torch.nn.Module
    slot: int
    routed: bool = False
    current: tuple[tuple[int, int], torch.Tensor] | None = None
    replays: collections.deque = field(default_factory=collections.deque)
    direct: bool = False


class ExpertLoadRecorder:
    """Records the routed and received token counts of the attached MoE layers inside a window of steps.

    Processes normally get one from the environment (:func:`attach_expert_load_probe`); tests may build their own.

    Args:
        output_dir: Directory of the per-rank JSON file.
        window: ``(start, end)`` in completed steps, or ``None`` to take Bridge's profiling window at the first
            micro-batch.
        dump_at_exit: Write the file at process exit if the window has not ended by then.
    """

    def __init__(self, output_dir: str, *, window: tuple[int, int] | None = None, dump_at_exit: bool = False) -> None:
        if not output_dir:
            raise ValueError(f"{ENV_SWITCH}=1 needs {ENV_DIR}=<directory> for the per-rank expert-load files.")
        if window is not None:
            window = _parse_window(f"{window[0]}:{window[1]}", "window")
        self.output_dir = os.path.expanduser(output_dir)
        self.window = window
        self.step: int | None = None
        self.microbatch = -1
        self.dumped_path: str | None = None
        self._probes: list[_LayerProbe] = []
        self._records: list[tuple[int, int, int, Any, Any]] = []
        self._unrouted_dispatches = 0
        self._identity: dict[str, int] | None = None
        if dump_at_exit:
            atexit.register(self._dump_at_exit)

    # ------------------------------------------------------------------------------------------------------------
    # Wiring

    def attach(self, layer: torch.nn.Module) -> None:
        """Wrap ``layer.router.routing`` and ``layer.token_dispatcher.dispatch_postprocess`` of these instances.

        Args:
            layer: A Megatron-Core ``MoELayer`` (an ERNIE 5.5 MoE layer).
        """
        probe = _LayerProbe(layer=layer, slot=len(self._probes))
        self._probes.append(probe)
        router_routing: Callable = layer.router.routing

        def routing(logits, padding_mask=None, **kwargs):
            probs, routing_map = router_routing(logits, padding_mask=padding_mask, **kwargs)
            self._on_routing(probe, routing_map, padding_mask)
            return probs, routing_map

        layer.router.routing = routing
        dispatcher = getattr(layer, "token_dispatcher", None)
        if dispatcher is None:
            # A megakernel backend (MoK) dispatches inside its kernel and builds no token dispatcher: record the
            # routed counts as soon as the router has run, with no received counts.
            probe.direct = True
            return
        dispatch_postprocess: Callable = dispatcher.dispatch_postprocess

        def postprocess(*args, **kwargs):
            outputs = dispatch_postprocess(*args, **kwargs)
            self._on_dispatch(probe, outputs[1])
            return outputs

        dispatcher.dispatch_postprocess = postprocess

    def begin_microbatch(self, step: int, *, training: bool = True, profiling: Any = None) -> None:
        """Announce the next micro-batch; the ERNIE 5.5 training step calls this through :func:`note_microbatch`.

        Args:
            step: Completed training steps (``train_state.step``) of the iteration the micro-batch belongs to.
            training: False for evaluation forwards, which are ignored.
            profiling: Bridge's ``ProfilingConfig``, whose window applies when none was given.

        Raises:
            ValueError: If no window was given and no profiler is configured.
        """
        if not training:
            return
        # Read the ranks while the process groups exist (a file written at exit could not).
        self._identify()
        if self.window is None:
            self.window = _profiling_window(profiling)
            if self.window is None:
                raise ValueError(
                    f"{ENV_SWITCH}=1 needs {ENV_STEPS}=<start>:<end> (completed steps), or a configured profiler "
                    "whose profile_step_start/profile_step_end window it then records."
                )
        if step != self.step:
            self.step, self.microbatch = step, 0
        else:
            self.microbatch += 1
        if self.dumped_path is None and step >= self.window[1]:
            self.dump()

    @property
    def recording(self) -> bool:
        """Whether the current micro-batch lies in the window (and the file is not written yet)."""
        return (
            self.dumped_path is None
            and self.window is not None
            and self.step is not None
            and self.window[0] <= self.step < self.window[1]
        )

    # ------------------------------------------------------------------------------------------------------------
    # Recording

    def _on_routing(self, probe: _LayerProbe, routing_map: torch.Tensor, padding_mask: torch.Tensor | None) -> None:
        probe.routed, probe.current = True, None
        if not (probe.layer.training and self.recording) or _capturing():
            return
        label = (self.step, self.microbatch)
        if torch.is_grad_enabled():
            # Queue entries of an earlier step belong to forwards that were never replayed.
            while probe.replays and probe.replays[0][0] != self.step:
                probe.replays.popleft()
            if probe.replays:
                # The recompute replay of a forward recorded without autograd.
                probe.replays.popleft()
                return
        else:
            probe.replays.append(label)
        with torch.no_grad():
            routing_map = routing_map.reshape(-1, routing_map.shape[-1])
            if padding_mask is not None:
                routing_map = routing_map & ~padding_mask.reshape(-1, 1).bool()
            probe.current = (label, routing_map.sum(dim=0))
        if probe.direct:
            # No dispatcher follows (megakernel backend): this is the layer's record for the micro-batch.
            (step, microbatch), routed = probe.current
            probe.current, probe.routed = None, False
            self._records.append((step, microbatch, probe.slot, None, routed))

    def _on_dispatch(self, probe: _LayerProbe, tokens_per_expert: Any) -> None:
        current, routed = probe.current, probe.routed
        probe.current, probe.routed = None, False
        if current is None:
            if not routed and probe.layer.training and self.recording and not _capturing():
                # The router did not run in this pass (e.g. a CUDA-graphed router): nothing can be recorded.
                self._unrouted_dispatches += 1
            return
        (step, microbatch), routed = current
        if isinstance(tokens_per_expert, torch.Tensor):
            if tokens_per_expert.device.type == "cpu":
                received = tokens_per_expert.tolist()
            else:
                received = tokens_per_expert.detach().clone()
        else:
            received = None if tokens_per_expert is None else list(tokens_per_expert)
        self._records.append((step, microbatch, probe.slot, received, routed))

    # ------------------------------------------------------------------------------------------------------------
    # Output

    def records(self) -> list[dict[str, Any]]:
        """The records so far, with host counts (copies device counts to the host)."""
        received = _to_host([record[3] for record in self._records])
        routed = _to_host([record[4] for record in self._records])
        return [
            {"step": step, "microbatch": microbatch, "slot": slot, "received": rec, "routed": rout}
            for (step, microbatch, slot, _, _), rec, rout in zip(self._records, received, routed)
        ]

    def _identify(self) -> dict[str, int]:
        """Global and expert-parallel rank and size, read once (the first micro-batch, or the first file)."""
        if self._identity is None:
            rank, world_size = _global_rank_and_size()
            identity = {"rank": rank, "world_size": world_size}
            if self._probes:
                group = self._probes[0].layer.ep_group
                identity.update(ep_rank=get_pg_rank(group), ep_size=get_pg_size(group))
            self._identity = identity
        return self._identity

    def payload(self) -> dict[str, Any]:
        """The JSON document of this rank."""
        identity = self._identify()
        payload: dict[str, Any] = {"format": FORMAT_VERSION, "rank": identity["rank"]}
        payload["world_size"] = identity["world_size"]
        payload["window"] = list(self.window) if self.window is not None else None
        if self._probes:
            layer = self._probes[0].layer
            payload.update(
                num_experts=layer.config.num_moe_experts,
                topk=layer.config.moe_router_topk,
                ep_rank=identity.get("ep_rank"),
                ep_size=identity.get("ep_size"),
                num_local_experts=layer.num_local_experts,
                local_expert_indices=list(layer.local_expert_indices),
            )
        payload["layers"] = [
            {
                "slot": probe.slot,
                "layer_number": probe.layer.layer_number,
                "is_mtp_layer": bool(probe.layer.is_mtp_layer),
            }
            for probe in self._probes
        ]
        payload["records"] = self.records()
        if self._unrouted_dispatches:
            payload["unrouted_dispatches"] = self._unrouted_dispatches
        return payload

    def dump(self) -> str:
        """Write this rank's file (atomically), free the records and stop recording.

        Returns:
            The path written.
        """
        payload = self.payload()
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"expert_load_rank{payload['rank']:05d}.json")
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
        self.dumped_path = path
        self._records.clear()
        for probe in self._probes:
            probe.routed, probe.current = False, None
            probe.replays.clear()
        if not payload["records"]:
            logger.warning("ERNIE 5.5 expert load: no record in window %s; wrote an empty %s", self.window, path)
        log_single_rank(
            logger,
            logging.INFO,
            "ERNIE 5.5 expert load: %d records per rank (window %s) written to %s",
            len(payload["records"]),
            self.window,
            self.output_dir,
        )
        return path

    def _dump_at_exit(self) -> None:
        if self.dumped_path is not None:
            return
        try:
            self.dump()
        except Exception as error:  # noqa: BLE001 - never fail the interpreter exit for a diagnostic
            logger.warning("ERNIE 5.5 expert load: could not write the file at exit: %s", error)


_RECORDER: ExpertLoadRecorder | None = None


def expert_load_recorder() -> ExpertLoadRecorder | None:
    """The process's recorder, or ``None`` when the diagnostic is off (or no MoE layer was built yet)."""
    return _RECORDER


def attach_expert_load_probe(layer: torch.nn.Module) -> None:
    """Attach ``layer`` to the process's recorder, creating it from the environment on first use.

    Called by :class:`~megatron.bridge.models.ernie.modeling_ernie55.moe.Ernie55MoELayer` only when
    :func:`expert_load_logging_enabled` is true.

    Raises:
        ValueError: On a missing ``ERNIE55_EXPERT_LOAD_DIR`` or a malformed ``ERNIE55_EXPERT_LOAD_STEPS``.
    """
    global _RECORDER
    if _RECORDER is None:
        steps = os.environ.get(ENV_STEPS, "").strip()
        _RECORDER = ExpertLoadRecorder(
            os.environ.get(ENV_DIR, "").strip(),
            window=_parse_window(steps, ENV_STEPS) if steps else None,
            dump_at_exit=True,
        )
        log_single_rank(
            logger,
            logging.INFO,
            "ERNIE 5.5 expert load: recording window %s (completed steps) to %s",
            _RECORDER.window or "of the profiler",
            _RECORDER.output_dir,
        )
    _RECORDER.attach(layer)


def note_microbatch(state: Any, *, training: bool) -> None:
    """Announce a micro-batch of the training step to the recorder; returns at once when the diagnostic is off.

    Args:
        state: Bridge's ``GlobalState`` (``train_state.step`` and ``cfg.profiling`` are read only when on).
        training: Whether the model is in training mode (evaluation forwards are ignored).
    """
    recorder = _RECORDER
    if recorder is None:
        return
    cfg = getattr(state, "cfg", None)
    recorder.begin_microbatch(
        state.train_state.step, training=training, profiling=getattr(cfg, "profiling", None) if cfg else None
    )


def flush_expert_load() -> str | None:
    """Write the process's file now, if the diagnostic is on and the file is not written yet.

    Returns:
        The path written, or ``None``.
    """
    if _RECORDER is None or _RECORDER.dumped_path is not None:
        return None
    return _RECORDER.dump()
