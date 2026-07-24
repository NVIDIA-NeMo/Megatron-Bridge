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

"""Per-layer attribution for the determinism tracer (layer 1 of the design).

Maintains a module-scope stack (in ``collective_trace._S.scope_stack``) by pushing each
module's qualified name on entry and popping it on exit, for BOTH the forward and
backward passes. Every record emitted by ``collective_trace``/``op_trace`` then carries
the enclosing layer in its ``scope`` field — turning an opaque ``aten.view (caller=?)``
into ``decoder.layers.7.self_attention[bwd] | aten.view``.

Why forward AND backward hooks: a divergence in the backward grad computation runs under
autograd with only torch C++ frames on the Python stack (so ``caller`` resolves to ``?``
and no module forward is executing). ``register_full_backward_pre_hook`` gives the
enclosing module during backward. Under activation recompute the forward re-runs inside
backward, so the forward hooks also fire there.

Best-effort: ``full_backward_hook`` does not fire for every module (e.g. no grad input),
so the stack can drift slightly within a step; ``collective_trace.set_active`` clears it
at each step boundary to bound the drift. The scope is a strong hint, used alongside
``caller``/``op``, not a guarantee.
"""

import logging

from megatron.bridge.training.utils.determinism import collective_trace as ct


logger = logging.getLogger(__name__)

_handles: list = []


def _push(name: str):
    def hook(module, *args):
        # Only track scope while capturing — otherwise non-firing backward pops would let
        # the stack grow unbounded across untraced iterations (set_active clears it).
        if ct._S.active:
            ct._S.scope_stack.append(name)

    return hook


def _pop(name: str):
    def hook(module, *args):
        # Conservative pop: only remove if our name is on top, so a misfiring sibling
        # hook cannot pop someone else's scope. Stale entries are cleared per step.
        if ct._S.active and ct._S.scope_stack and ct._S.scope_stack[-1] == name:
            ct._S.scope_stack.pop()

    return hook


def register(model, prefix: str = "", backward_scope: bool = False) -> int:
    """Register FORWARD scope hooks on every named submodule (PP-safe by default).

    Args:
        model: The (unwrapped) model module to attribute.
        prefix: Disambiguating prefix for the module names (e.g. ``"chunk0."``). With
            virtual pipeline / multiple model chunks, each chunk's ``named_modules()``
            yields the same relative names; without a per-chunk prefix two physical
            layers would push identical scope strings and the diff would mis-attribute.
        backward_scope: If True, ALSO register full backward hooks so pure-backward ops
            get a ``[bwd]`` layer tag. **DANGER — PP-incompatible.** PyTorch's
            ``register_full_backward_hook`` wraps module OUTPUTS as views (``_base`` becomes
            non-None) merely by being registered; Megatron's pipeline
            ``deallocate_output_tensor`` asserts ``out._base is None`` (with
            ``deallocate_pipeline_outputs=True``), so this CRASHES any PP>1 run on the first
            step. Only enable for TP-only debugging where you want backward layer labels.
            Default False → forward hooks only → outputs stay non-views → PP-safe.

    Returns:
        The number of modules instrumented.
    """
    count = 0
    for name, module in model.named_modules():
        if not name:
            continue
        qname = f"{prefix}{name}"
        # Forward hooks return None → do NOT modify/wrap the output → PP-safe.
        _handles.append(module.register_forward_pre_hook(_push(qname)))
        _handles.append(module.register_forward_hook(_pop(qname)))
        if backward_scope:
            # OPT-IN only: full backward hooks alias module outputs into views and break
            # Megatron PP output deallocation. Backward ops still get captured either way;
            # this only adds the [bwd] layer LABEL. See the warning in the docstring.
            try:
                _handles.append(module.register_full_backward_pre_hook(_push(qname + "[bwd]")))
                _handles.append(module.register_full_backward_hook(_pop(qname + "[bwd]")))
            except Exception:
                pass
        count += 1
    logger.info(
        "module_scope: registered forward scope hooks on %d modules (prefix=%r, backward_scope=%s)",
        count,
        prefix,
        backward_scope,
    )
    return count


def unregister() -> None:
    """Remove all scope hooks and clear the stack."""
    for h in _handles:
        h.remove()
    _handles.clear()
    ct._S.scope_stack.clear()
