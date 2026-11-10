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

"""Helpers shared by the ERNIE 5.5 CPU unit tests."""

from __future__ import annotations

import importlib.util
import sys
import types

import torch


_REAL_CURRENT_DEVICE = torch.cuda.current_device
_LORA_LAYERS_MODULE = "megatron.bridge.peft.lora_layers"
# The checkpoint-conversion modules the ERNIE 5.5 conversion tests use.
_CONVERSION_MODULES = (
    "megatron.bridge.models.conversion.model_bridge",
    "megatron.bridge.models.conversion.mapping_registry",
    "megatron.bridge.models.conversion.param_mapping",
    "megatron.bridge.models.hf_pretrained.state",
)


def allow_conversion_imports_without_te() -> None:
    """Import Bridge's checkpoint-conversion modules on a machine without Transformer Engine.

    ``megatron.bridge.models.conversion`` imports Bridge's PEFT bridge, and through it the LoRA layer
    module ``megatron.bridge.peft.lora_layers``, which imports Transformer Engine at module level.
    Without TE (CPU-only test machines) this imports the conversion modules with a placeholder in
    ``sys.modules`` for that one module, and removes the placeholder again afterwards. Its classes are
    only used in ``isinstance`` checks on LoRA adapters, which the ERNIE 5.5 conversion tests never
    create. Only the modules imported here see the placeholder: a later import of the LoRA layers fails
    without TE as it would without this helper. Transformer Engine itself is not stubbed, so
    Megatron-Core still sees no TE. With TE installed this does nothing.
    """
    if importlib.util.find_spec("transformer_engine") is not None:
        return
    if all(name in sys.modules for name in _CONVERSION_MODULES):
        return
    placeholder = None
    if _LORA_LAYERS_MODULE not in sys.modules:
        placeholder = types.ModuleType(_LORA_LAYERS_MODULE)
        placeholder.__doc__ = "Placeholder for Bridge's LoRA layers on a machine without Transformer Engine."

        def missing_class(name: str) -> type:
            if name.startswith("__"):
                raise AttributeError(name)
            cls = type(name, (), {"__module__": _LORA_LAYERS_MODULE})
            setattr(placeholder, name, cls)
            return cls

        placeholder.__getattr__ = missing_class
        sys.modules[_LORA_LAYERS_MODULE] = placeholder
    try:
        for name in _CONVERSION_MODULES:
            importlib.import_module(name)
    finally:
        if placeholder is not None and sys.modules.get(_LORA_LAYERS_MODULE) is placeholder:
            del sys.modules[_LORA_LAYERS_MODULE]


# Callers that must see the real device: torch itself and the Triton runtime it drives.
_DEVICE_AWARE_PACKAGES = ("torch", "triton")


def cpu_current_device() -> int | str:
    """``torch.cuda.current_device`` for CPU tests: ``"cpu"`` for model code, the real index for torch.

    Megatron-Core allocates router buffers, RoPE tables, the compressor APE and the DSA loss tracker
    on ``torch.cuda.current_device()``. The ERNIE 5.5 CPU tests therefore patch it to return ``"cpu"``,
    so they run on CPU even when a GPU is visible. Callers inside torch and Triton get the real device
    index instead. With a visible GPU, the first ``torch.compile`` call checks the device's Triton
    and TMA support through ``current_device()`` and uses the answer as a list index or compares it
    with integers. A blanket ``"cpu"`` would make every compiled Megatron-Core helper fail to trace.
    Without CUDA, those checks never call ``current_device()``.

    Use as ``monkeypatch.setattr(torch.cuda, "current_device", cpu_current_device)``.

    Returns:
        ``"cpu"``, or the real current CUDA device index when the caller is in torch or Triton.
    """
    caller = sys._getframe(1).f_globals.get("__name__", "")
    if caller.partition(".")[0] in _DEVICE_AWARE_PACKAGES:
        return _REAL_CURRENT_DEVICE()
    return "cpu"


def mtp_row_loss_oracle(
    logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor, cu_seqlens: list[int], *, depth: int = 1
) -> float:
    """fp64 per-row loss of MTP ``depth`` (1-based) of a packed ``b = 1`` row, written from the MTP definition.

    Depth ``k`` predicts the target ``k`` positions ahead inside the same document: labels and mask move ``k``
    positions left within each document of ``cu_seqlens``, and the last ``k`` positions of a document get no target.
    The row loss is the masked mean of the token cross entropies.

    Args:
        logits: ``[s, 1, vocab]`` logits of the depth.
        labels: ``[1, s]`` targets aligned with the inputs (``-100`` or any id where ``loss_mask`` is 0).
        loss_mask: ``[1, s]`` 1 for supervised tokens.
        cu_seqlens: Document boundaries of the row.
        depth: MTP depth, 1-based.

    Returns:
        The row's mean MTP cross entropy.
    """
    rolled_labels, rolled_mask = labels.clone(), loss_mask.clone()
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        tail = min(depth, end - start)
        rolled_labels[0, start:end] = torch.cat([labels[0, start + depth : end], labels.new_zeros(tail)])
        rolled_mask[0, start:end] = torch.cat([loss_mask[0, start + depth : end], loss_mask.new_zeros(tail)])
    log_probs = torch.log_softmax(logits.double()[:, 0], dim=-1)
    nll = -log_probs.gather(-1, rolled_labels[0].clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return float((nll * rolled_mask[0].double()).sum() / rolled_mask.double().sum().clamp(min=1))


def mtp_tracked_losses() -> torch.Tensor | None:
    """Every MTP depth's loss in Megatron-Core's MTP tracker (``loss_sums`` on dev, ``loss_values`` on main)."""
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    tracker = MTPLossLoggingHelper.tracker
    return tracker.get("loss_sums", tracker.get("loss_values"))


class CallRecorder:
    """Stand-in for a kernel entry point: records every call and returns ``result(*args, **kwargs)``."""

    def __init__(self, result) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result(*args, **kwargs)


def isolate_fused_dispatch(monkeypatch) -> types.ModuleType:
    """Give a test a clean ERNIE 5.5 fused-op dispatch; ``monkeypatch`` restores the process state afterwards.

    Megatron-Core's ``situ_glu`` bindings and ``cp_utils.apply_thd_cp_local_rope_unfused`` go back to the
    Megatron-Core functions (unwrapped), and the recorded SiTU-GLU scales, the one-time log keys and
    ``FUSED_OP_CALLS`` start empty. Building an ERNIE 5.5 model installs the dispatchers process-wide; without this, a test would
    see the dispatchers and counts of the tests before it.

    Args:
        monkeypatch: The test's ``monkeypatch`` fixture.

    Returns:
        The ``fused_dispatch`` module.
    """
    import collections

    from megatron.bridge.models.ernie.modeling_ernie55 import fused_dispatch

    for site in fused_dispatch.SITU_GLU_SITES:
        module = importlib.import_module(site)
        current = module.situ_glu
        monkeypatch.setattr(module, "situ_glu", getattr(current, "ernie55_eager", current))
    cp_utils = importlib.import_module(fused_dispatch.CP_UTILS_MODULE)
    current = cp_utils.apply_thd_cp_local_rope_unfused
    monkeypatch.setattr(cp_utils, "apply_thd_cp_local_rope_unfused", getattr(current, "ernie55_unfused", current))
    monkeypatch.setattr(fused_dispatch, "_STATE", fused_dispatch._DispatchState())
    monkeypatch.setattr(fused_dispatch, "FUSED_OP_CALLS", collections.Counter())
    return fused_dispatch
