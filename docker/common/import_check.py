# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Smoke-import every installed distribution's top-level modules.

This guards against shipping an environment whose installed packages cannot
actually be imported -- for example an ABI or version skew between a vendored
shared library and its Python bindings. ``pip check`` only validates declared
version ranges; it never exercises an import, so a nominally-satisfied
dependency that crashes the moment it is imported slips through. This script
closes that gap: it enumerates installed distributions, resolves each one's
top-level importable modules, imports every module in an isolated subprocess,
and exits non-zero if any import raises (or hangs).

Run with no arguments to check the active interpreter's environment::

    python import_check.py

Because an image is built without a GPU, packages that require a driver or a
host library at import time are not importable at build time. List those in a
skip file (one module per line, ``#`` comments allowed) and pass it via
``--skip-file``; every other import error then fails the build.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportResult:
    """Outcome of importing a single top-level module.

    Attributes:
        module: The top-level module name that was imported.
        ok: True when the import subprocess exited cleanly.
        detail: Trailing diagnostic output when the import failed, else "".
    """

    module: str
    ok: bool
    detail: str


def _clean_module_name(filename: str) -> str | None:
    """Reduce a top-level filename to the module name Python would import.

    Extension modules carry an ABI tag (``foo.cpython-312-x86_64-linux-gnu.so``)
    that is not part of the import name, so everything from the first dot is
    dropped. Pure-Python files lose their ``.py`` suffix.

    Args:
        filename: The basename of a top-level file.

    Returns:
        The import name, or None when the file is not an importable module.
    """
    if filename.endswith((".so", ".pyd")):
        return filename.split(".", 1)[0]
    if filename.endswith(".py"):
        return filename[:-3]
    return None


def _modules_from_files(dist: md.Distribution) -> set[str]:
    """Derive importable top-level modules from a distribution's file list.

    Deriving from the recorded files (rather than ``top_level.txt``) reflects
    what is actually on disk, which avoids both stale metadata entries that name
    non-importable helpers and ABI-tagged extension filenames.

    Args:
        dist: The installed distribution to inspect.

    Returns:
        The set of importable top-level module names.
    """
    modules: set[str] = set()
    for entry in dist.files or []:
        parts = entry.parts
        if not parts:
            continue
        head = parts[0]
        if head.endswith((".dist-info", ".data", ".egg-info")) or head == "__pycache__":
            continue
        if len(parts) == 1:
            name = _clean_module_name(head)
            candidate = name if name and name != "__init__" else None
        else:
            candidate = head if entry.suffix in {".py", ".so", ".pyd"} else None
        if candidate and candidate.isidentifier():
            modules.add(candidate)
    return modules


def _modules_from_top_level_txt(dist: md.Distribution) -> set[str]:
    """Return importable module names declared in ``top_level.txt``.

    Used only when the file list is unavailable.

    Args:
        dist: The installed distribution to inspect.

    Returns:
        The set of top-level module names that are valid identifiers.
    """
    text = dist.read_text("top_level.txt")
    if not text:
        return set()
    return {line.strip() for line in text.splitlines() if line.strip().isidentifier()}


def discover_modules(skip: set[str]) -> dict[str, list[str]]:
    """Map each importable top-level module to the distributions providing it.

    Args:
        skip: Module names to exclude from the result.

    Returns:
        A mapping of module name to the sorted list of distribution names that
        provide it, excluding any module in ``skip``.
    """
    providers: dict[str, set[str]] = {}
    for dist in md.distributions():
        name = dist.metadata["Name"] or "<unknown>"
        modules = _modules_from_files(dist) or _modules_from_top_level_txt(dist)
        for module in modules:
            if module in skip:
                continue
            providers.setdefault(module, set()).add(name)
    return {module: sorted(names) for module, names in providers.items()}


def import_one(module: str, timeout: float) -> ImportResult:
    """Import a single module in a fresh subprocess.

    Isolation in a subprocess keeps a hard crash (segfault, ``os._exit``) or a
    hang in one module from aborting the whole sweep.

    Args:
        module: The top-level module name to import.
        timeout: Seconds to allow before treating the import as hung.

    Returns:
        The :class:`ImportResult` for this module.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ImportResult(module, False, f"timed out after {timeout:.0f}s")
    if proc.returncode == 0:
        return ImportResult(module, True, "")
    tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-3:])
    return ImportResult(module, False, tail or f"exit code {proc.returncode}")


def load_skip(skip_file: Path | None) -> set[str]:
    """Read a newline-delimited skip list, ignoring blanks and comments.

    Args:
        skip_file: Path to the skip file, or None.

    Returns:
        The set of module names to skip.
    """
    if skip_file is None or not skip_file.exists():
        return set()
    skip: set[str] = set()
    for line in skip_file.read_text().splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            skip.add(token)
    return skip


def run(skip: set[str], jobs: int, timeout: float) -> list[ImportResult]:
    """Import every discovered module and collect results.

    Args:
        skip: Module names to exclude.
        jobs: Maximum number of concurrent import subprocesses.
        timeout: Per-import timeout in seconds.

    Returns:
        The import results sorted by module name.
    """
    modules = discover_modules(skip)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = pool.map(lambda m: import_one(m, timeout), sorted(modules))
    return list(results)


def report(results: list[ImportResult], skipped: set[str]) -> int:
    """Print a summary and return the process exit code.

    Args:
        results: The import results to summarize.
        skipped: Module names that were skipped.

    Returns:
        1 if any import failed, otherwise 0.
    """
    failures = [r for r in results if not r.ok]
    for failure in failures:
        print(f"FAIL  {failure.module}")
        for line in failure.detail.splitlines():
            print(f"        {line}")
    passed = len(results) - len(failures)
    print(f"\nimport check: {passed} ok, {len(failures)} failed, {len(skipped)} skipped, {len(results)} total")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the import-check CLI.

    Args:
        argv: Argument vector, defaulting to ``sys.argv``.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description="Smoke-import installed packages.")
    parser.add_argument("--skip-file", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    skip = load_skip(args.skip_file)
    results = run(skip, args.jobs, args.timeout)
    return report(results, skip)


if __name__ == "__main__":
    raise SystemExit(main())
