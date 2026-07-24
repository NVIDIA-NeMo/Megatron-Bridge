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

"""Zero-source-edit bootstrap for the determinism tracer (read-only Megatron-LM case).

CPython auto-imports ``sitecustomize`` at interpreter startup if its directory is on
``PYTHONPATH``. This module is **inert unless ``DET_TRACE_OUT_DIR`` is set**; when set it
installs a post-import hook that wraps ``megatron.training.training.train_step`` to drive
the tracer (``megatron.bridge.training.utils.determinism``) WITHOUT editing the read-only
``3rdparty/Megatron-LM`` source — the repo boundary forbids modifying it. For a
Megatron-Bridge training run the ``train.py`` wiring already covers this; use this bootstrap
for a bare Megatron-LM run (``pretrain_gpt.py`` etc.) where you cannot edit the loop.

Usage (prepend this dir to the container PYTHONPATH for the traced run only)::

    PYTHONPATH=/opt/Megatron-Bridge/scripts/determinism:$PYTHONPATH \\
    DET_TRACE_OUT_DIR=/lustre/.../det_streams/run_A DET_TRACE_ITERS=1-50 \\
        <your Megatron-LM launch command>

Env knobs (same names as the Bridge train.py wiring):
    DET_TRACE_OUT_DIR   (required to activate)  per-logical-rank .fp stream dir
    DET_TRACE_ITERS     default "1"             "all" | "a-b" | comma list (1-based train_step calls)
    DET_TRACE_OPS       set -> also fingerprint every ATen op (op-level root cause)
    DET_TRACE_BWD_SCOPE set -> add [bwd] layer labels (TP-only; PP-incompatible)
"""

import os
import sys


def _parse_iters(raw):
    raw = (raw or "1").strip()
    if raw == "all":
        return None  # all steps
    out = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            if "-" in tok:
                a, b = (int(v) for v in tok.split("-", 1))
                out.update(range(min(a, b), max(a, b) + 1))
            else:
                out.add(int(tok))
        except ValueError:
            print(f"[det-boot] ignoring malformed DET_TRACE_ITERS token: {tok!r}", flush=True)
    return out


def _install_post_import(name, callback):
    """Run ``callback(module)`` once, right after ``name`` is first imported."""
    if name in sys.modules:
        callback(sys.modules[name])
        return
    import importlib.abc
    import importlib.util

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != name:
                return None
            # Remove self BEFORE resolving the real spec to avoid infinite recursion.
            try:
                sys.meta_path.remove(self)
            except ValueError:
                return None
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                try:
                    callback(module)
                except Exception as e:  # never break the training job
                    print(f"[det-boot] post-import callback failed: {e!r}", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


def _wrap_train_step(training_mod):
    """Wrap ``megatron.training.training.train_step`` to drive the tracer."""
    out_dir = os.environ["DET_TRACE_OUT_DIR"]
    iters = _parse_iters(os.environ.get("DET_TRACE_ITERS", "1"))
    orig_train_step = training_mod.train_step
    state = {"enabled": False, "step": 0}

    def _extract_model(args, kwargs):
        # train_step(forward_step_func, data_iterator, model, optimizer, ...)
        if "model" in kwargs:
            return kwargs["model"]
        return args[2] if len(args) > 2 else None

    def wrapped(*args, **kwargs):
        from megatron.bridge.training.utils.determinism import collective_trace as ct

        # Lazy one-time enable: parallel_state is initialized and the model exists by the
        # first train_step — exactly what the tracer needs.
        if not state["enabled"]:
            state["enabled"] = True
            ct.enable(out_dir=out_dir)
            model = _extract_model(args, kwargs)
            try:
                from megatron.bridge.training.utils.determinism import module_scope as ms

                chunks = model if isinstance(model, (list, tuple)) else [model]
                multi = len(chunks) > 1
                bwd = bool(os.environ.get("DET_TRACE_BWD_SCOPE"))
                for i, ch in enumerate(chunks):
                    if ch is None:
                        continue
                    ms.register(getattr(ch, "module", ch), prefix=(f"chunk{i}." if multi else ""), backward_scope=bwd)
            except Exception as e:
                print(f"[det-boot] module_scope register failed: {e!r}", flush=True)
            if os.environ.get("DET_TRACE_OPS"):
                try:
                    from megatron.bridge.training.utils.determinism import op_trace as ot

                    ot.enable()
                except Exception as e:
                    print(f"[det-boot] op_trace enable failed: {e!r}", flush=True)
            import atexit

            atexit.register(ct.disable)
            print(
                f"[det-boot] tracer enabled -> {out_dir}; "
                f"iters={'all' if iters is None else sorted(iters)}; "
                f"op-level={'ON' if os.environ.get('DET_TRACE_OPS') else 'off'}",
                flush=True,
            )

        state["step"] += 1  # 1-based to match Megatron iteration numbering / DET_TRACE_ITERS
        step = state["step"]
        on = (iters is None) or (step in iters)
        if on:
            ct.set_active(True, window=step)
        try:
            return orig_train_step(*args, **kwargs)
        finally:
            if on:
                ct.set_active(False)  # drains this window's staged records (HybridEP-safe)

    training_mod.train_step = wrapped
    print("[det-boot] wrapped megatron.training.training.train_step", flush=True)


def _chain_shadowed_sitecustomize():
    """Run any ``sitecustomize`` this file shadows (deeper on ``sys.path``).

    This dir is prepended to PYTHONPATH only for traced runs, so if the container ships its
    own ``sitecustomize`` (cuda/env setup) ours would suppress it. Load and exec the shadowed
    one so its side effects still happen. Best-effort.
    """
    import importlib.machinery
    import importlib.util

    my_dir = os.path.dirname(os.path.abspath(__file__))
    other = [p for p in sys.path if p and os.path.abspath(p) != my_dir]
    try:
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", other)
    except Exception:
        spec = None
    if spec is None or not spec.origin or spec.loader is None:
        return
    if os.path.abspath(os.path.dirname(spec.origin)) == my_dir:
        return
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"[det-boot] chained sitecustomize failed: {e!r}", flush=True)


# -- activate only when requested ---------------------------------------------------------
if os.environ.get("DET_TRACE_OUT_DIR"):
    try:
        _install_post_import("megatron.training.training", _wrap_train_step)
        print("[det-boot] armed: will wrap train_step on import", flush=True)
    except Exception as e:
        print(f"[det-boot] arming failed (training will run untraced): {e!r}", flush=True)
    # Preserve any container sitecustomize we shadow (only on PYTHONPATH for traced runs).
    _chain_shadowed_sitecustomize()
