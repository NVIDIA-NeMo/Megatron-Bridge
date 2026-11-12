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

"""Run CPU test cases on several ``torch.distributed`` gloo ranks from inside one pytest process.

The ERNIE 5.5 context-parallel tests compare CP > 1 with CP = 1 on CPU, which needs real
collectives between processes. :func:`run_cases` spawns one process per rank; every rank runs a
setup function and then every case of a test module in the same order (the cases run collectives)
and writes ``{case: {"ok", "error"}}`` to a JSON file, which the pytest process reads, so each
pytest test can assert its own case.

Spawned processes import the test module again. This module imports nothing from Megatron-Bridge
at module level, because a test session may register the ``megatron.bridge`` packages without
running their ``__init__`` (a lightweight session that skips the full package import); the
workers register them the same way before importing the test module (:func:`bare_package_layout`).
"""

from __future__ import annotations

import datetime
import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch


__all__ = ["bare_package_layout", "run_cases"]

_BRIDGE_PACKAGES = ("megatron.bridge", "megatron.bridge.models", "megatron.bridge.models.ernie")
_COLLECTIVE_TIMEOUT = datetime.timedelta(seconds=120)


def bare_package_layout() -> dict[str, list[str]]:
    """Megatron-Bridge packages this process registered without running their ``__init__``.

    Returns:
        ``{package: search path}`` for every such package; empty after a normal import.
    """
    layout = {}
    for name in _BRIDGE_PACKAGES:
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None) is None and hasattr(module, "__path__"):
            layout[name] = list(module.__path__)
    return layout


def _register_bare_packages(layout: dict[str, list[str]]) -> None:
    if not layout:
        return
    import megatron

    for name in _BRIDGE_PACKAGES:
        if name not in layout or name in sys.modules:
            continue
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = list(layout[name])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        parent_name, _, child = name.rpartition(".")
        setattr(sys.modules.get(parent_name, megatron), child, module)


def _rank_entry(
    rank: int,
    world_size: int,
    init_file: str,
    module_name: str,
    setup_name: str,
    case_names: list[str],
    layout: dict[str, list[str]],
    results_dir: str,
) -> None:
    """Spawned rank: set up gloo, run every case in order and write the results.

    pytest's outcomes (``pytest.fail``, or a ``pytest.raises`` block that saw no exception) derive
    from ``BaseException``, not ``Exception``, so the rank catches ``BaseException``: a failed check
    is reported as that case's failure and the later cases still run. Only ``KeyboardInterrupt`` and
    ``SystemExit`` end the rank.
    """
    _register_bare_packages(layout)
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=_COLLECTIVE_TIMEOUT
    )
    results = {}
    try:
        module = importlib.import_module(module_name)
        context = getattr(module, setup_name)(world_size)
        for case_name in case_names:
            try:
                getattr(module, case_name)(context)
                results[case_name] = {"ok": True, "error": ""}
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:  # noqa: BLE001 - every failure, pytest outcomes included, becomes that case's report
                results[case_name] = {"ok": False, "error": traceback.format_exc()}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # noqa: BLE001 - a setup failure fails every case
        error = traceback.format_exc()
        results = {case_name: {"ok": False, "error": f"setup failed:\n{error}"} for case_name in case_names}
    finally:
        with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as handle:
            json.dump(results, handle)
        torch.distributed.destroy_process_group()


def run_cases(
    *,
    world_size: int,
    module_name: str,
    setup_name: str,
    case_names: list[str],
    workdir: Path,
    timeout: float = 900.0,
) -> dict[str, dict[int, dict]]:
    """Run ``case_names`` of ``module_name`` on ``world_size`` spawned gloo ranks.

    Every rank calls ``module.<setup_name>(world_size)`` once, then ``module.<case>(context)`` for
    every case in order.

    Args:
        world_size: Number of ranks.
        module_name: Importable name of the module with the setup and the cases.
        setup_name: Name of the setup function.
        case_names: Names of the case functions, run in this order on every rank.
        workdir: Empty directory for the rendezvous file and the results.
        timeout: Seconds before the ranks are killed.

    Returns:
        ``{case: {rank: {"ok": bool, "error": str}}}``.

    Raises:
        RuntimeError: If the ranks time out or a rank dies without results.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    context = torch.multiprocessing.start_processes(
        _rank_entry,
        args=(
            world_size,
            str(workdir / "rendezvous"),
            module_name,
            setup_name,
            list(case_names),
            bare_package_layout(),
            str(workdir),
        ),
        nprocs=world_size,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + timeout
    while not context.join(timeout=5):
        if time.monotonic() > deadline:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise RuntimeError(f"{world_size} gloo ranks did not finish {module_name} within {timeout} s.")
    results: dict[str, dict[int, dict]] = {case_name: {} for case_name in case_names}
    for rank in range(world_size):
        path = workdir / f"rank{rank}.json"
        if not path.exists():
            raise RuntimeError(f"Rank {rank} of {world_size} wrote no results ({path}).")
        for case_name, result in json.loads(path.read_text()).items():
            results[case_name][rank] = result
    return results
