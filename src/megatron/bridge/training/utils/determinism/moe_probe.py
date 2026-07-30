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

"""Targeted MoE-layer probe for the determinism tracer.

Emits three ordered fingerprint records per MoE layer so the offline diff can localize the
first divergence to a specific sub-op of the MoE forward — in particular to distinguish the
expert GEMM (e.g. the CuteDSL fused grouped MLP) from the token *combine*:

  1. ``expert_input``    — post-dispatch tokens entering the experts (grouped MLP)
  2. ``expert_gemm_out`` — the experts' output, BEFORE combine (grouped-MLP result)
  3. ``combine_out``     — the MoE layer output, AFTER the (HybridEP) combine

Motivation: the flex/HybridEP combine runs in a custom CUDA/RDMA extension that is neither a
``torch.distributed`` call nor an ATen op, so ``collective_trace``/``op_trace`` cannot see it
(a divergence there only surfaces one ATen op later, on the tensor that consumes the combined
output). These three points bracket it with plain ``nn.Module`` forward hooks on the experts
module and its parent MoE layer, so the combine's *output tensor* — an ordinary tensor — is
fingerprinted directly. The probe is dispatcher-agnostic (works for ``alltoall`` too).

Records enter the same ordered per-rank stream as every other record (group ``"moe"``) and
flush at the step boundary via ``collective_trace.flush_pending``. They are emitted output-only
(``input=[]``), so ``diff_streams`` reads ``input_match=True`` and returns the FIRST probe point
whose output diverges — everything earlier in the ordered stream having matched. Thus:
``expert_gemm_out`` diverging (``expert_input`` matched) implicates the grouped MLP;
``combine_out`` diverging (``expert_gemm_out`` matched) implicates the combine.

Registered entirely from the Bridge side — it never edits ``3rdparty/Megatron-LM``. Gated by
``DET_TRACE_MOE``; inert otherwise. See docs/determinism-debug-tool.md.
"""

import logging

import torch

from megatron.bridge.training.utils.determinism import collective_trace as ct


logger = logging.getLogger(__name__)

_handles: list = []


def _primary_tensor(x):
    """The hidden-state tensor from a module's in/out (often an ``(output, bias)`` tuple)."""
    if isinstance(x, (list, tuple)):
        return next((item for item in x if isinstance(item, torch.Tensor)), None)
    return x if isinstance(x, torch.Tensor) else None


def _emit(op_name: str, value) -> None:
    """Stash one output-only ``moe`` record for ``value``'s primary tensor.

    No-op unless the tracer is actively capturing this iteration (mirrors the guard in
    ``collective_trace``'s wrappers), and unless ``value`` carries a tensor.
    """
    if not (ct._S.enabled and ct._S.active) or ct._S.suspend:
        return
    t = _primary_tensor(value)
    if t is None:
        return
    ct._stash_named(op_name, "moe", [], ct._staged_sig_list(t))


def _pre_hook(op_name: str):
    def hook(module, args):
        _emit(op_name, args[0] if args else None)

    return hook


def _post_hook(op_name: str):
    def hook(module, args, output):
        _emit(op_name, output)

    return hook


def register(model, prefix: str = "", experts_suffix: str = "experts") -> int:
    """Attach probe hooks to every experts module and its parent MoE layer.

    Args:
        model: the (unwrapped) model chunk to instrument.
        prefix: informational per-chunk prefix (VPP/multi-chunk); the record's layer identity
            comes from ``align_idx`` + the ``module_scope`` tag, so this is not embedded in
            the key — accepted for call-site symmetry with ``module_scope.register``.
        experts_suffix: submodule name of the grouped-MLP experts container to match.

    Returns:
        The number of MoE layers instrumented.
    """
    mods = dict(model.named_modules())
    count = 0
    for name, module in mods.items():
        if not (name == experts_suffix or name.endswith("." + experts_suffix)):
            continue
        # experts (grouped MLP): post-dispatch input (pre-hook) + pre-combine output (post-hook).
        _handles.append(module.register_forward_pre_hook(_pre_hook("expert_input")))
        _handles.append(module.register_forward_hook(_post_hook("expert_gemm_out")))
        # Parent MoE layer: post-combine output (its forward returns after the combine).
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = mods.get(parent_name)
        if parent is not None:
            _handles.append(parent.register_forward_hook(_post_hook("combine_out")))
        count += 1
    logger.info(
        "moe_probe: instrumented %d MoE layer(s) (prefix=%r, experts_suffix=%r)",
        count,
        prefix,
        experts_suffix,
    )
    return count


def unregister() -> None:
    """Remove all MoE probe hooks."""
    for h in _handles:
        h.remove()
    _handles.clear()
