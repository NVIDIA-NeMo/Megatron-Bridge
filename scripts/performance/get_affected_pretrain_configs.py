# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Determine which pretrain config functions are affected by a git diff.

Usage (precise, default):
    git diff origin/main...HEAD | python get_affected_pretrain_configs.py

Usage (conservative fallback, file-names only):
    git diff --name-only origin/main...HEAD | python get_affected_pretrain_configs.py --files
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal


SCRIPT_DIR = Path(__file__).parent
CONFIGS_DIR = SCRIPT_DIR / "configs"
UTILS_DIR = SCRIPT_DIR / "utils"
REPO_ROOT = SCRIPT_DIR.parent.parent


# ---------------------------------------------------------------------------
# Phase 1: Parse unified diff
# ---------------------------------------------------------------------------


def parse_diff(
    diff_text: str,
) -> tuple[dict[str, set[int]], dict[str, list[tuple[str, str]]]]:
    """Parse unified diff into touched line sets and raw changed lines.

    Returns:
        touched_lines: {relative_filepath: set_of_line_numbers}
            Positive integers = added lines (new file line numbers).
            Negative integers = deleted lines (old file line numbers).
        raw_lines: {relative_filepath: [(sign, text), ...]}
            sign is '+' or '-'; text is the line content without the sign prefix.
    """
    touched: dict[str, set[int]] = defaultdict(set)
    raw: dict[str, list[tuple[str, str]]] = defaultdict(list)

    current_file: str | None = None
    old_lineno = 0
    new_lineno = 0

    for line in diff_text.splitlines():
        if line.startswith("--- "):
            pass  # old file tracked via +++ below
        elif line.startswith("+++ "):
            if line == "+++ /dev/null":
                current_file = None
            else:
                m = re.match(r"^\+\+\+ [ab]/(.+)$", line)
                current_file = m.group(1) if m else None
        elif line.startswith("@@ ") and current_file is not None:
            m = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                old_lineno = int(m.group(1))
                new_lineno = int(m.group(2))
        elif current_file is not None:
            if line.startswith("+") and not line.startswith("+++"):
                touched[current_file].add(new_lineno)
                raw[current_file].append(("+", line[1:]))
                new_lineno += 1
            elif line.startswith("-") and not line.startswith("---"):
                touched[current_file].add(-old_lineno)
                raw[current_file].append(("-", line[1:]))
                old_lineno += 1
            elif line.startswith(" "):
                old_lineno += 1
                new_lineno += 1

    return dict(touched), dict(raw)


def parse_files_list(text: str) -> list[str]:
    """Parse newline-separated file paths (for --files mode)."""
    return [line.strip() for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Phase 2: Classify changed files
# ---------------------------------------------------------------------------


def classify_file(path: str) -> Literal["utils", "wbc", "pretrain", "init", "other"]:
    """Classify a repo-relative filepath into one of the tracked change categories."""
    if path.startswith("scripts/performance/utils/") and path.endswith(".py"):
        return "utils"
    if "_workload_base_configs.py" in path:
        return "wbc"
    if "_pretrain" in path and path.endswith(".py") and "configs/" in path:
        return "pretrain"
    if path.endswith("__init__.py") and "scripts/performance/configs/" in path:
        return "init"
    return "other"


# ---------------------------------------------------------------------------
# Phase 3: Build indices
# ---------------------------------------------------------------------------


def discover_pretrain_files() -> list[Path]:
    """Find all *_pretrain*.py files under configs/."""
    return sorted(CONFIGS_DIR.rglob("*_pretrain*.py"))


def build_pretrain_index(pretrain_files: list[Path]) -> dict[Path, list[str]]:
    """Map each pretrain file → list of *_pretrain_config_* function names."""
    index: dict[Path, list[str]] = {}
    for pf in pretrain_files:
        funcs: list[str] = []
        try:
            tree = ast.parse(pf.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and "_pretrain_config_" in node.name:
                    funcs.append(node.name)
        except SyntaxError:
            pass
        index[pf] = funcs
    return index


def build_pretrain_utils_deps(pretrain_files: list[Path]) -> dict[Path, set[str]]:
    """Map each pretrain file → set of utils module stems it directly imports."""
    deps: dict[Path, set[str]] = {}
    for pf in pretrain_files:
        stems: set[str] = set()
        try:
            tree = ast.parse(pf.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("utils."):
                        stems.add(node.module.split(".")[1])
        except SyntaxError:
            pass
        deps[pf] = stems
    return deps


def build_utils_transitive_deps() -> dict[str, set[str]]:
    """Compute transitive closure of utils→utils imports.

    Returns {module_stem: set_of_all_utils_stems_it_depends_on_transitively}.
    """
    direct: dict[str, set[str]] = {}
    for py_file in UTILS_DIR.glob("*.py"):
        stem = py_file.stem
        if stem == "__init__":
            continue
        stems: set[str] = set()
        try:
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("utils."):
                        stems.add(node.module.split(".")[1])
        except SyntaxError:
            pass
        direct[stem] = stems

    # BFS for transitive closure: each module's full dependency set
    transitive: dict[str, set[str]] = {}
    for start in direct:
        visited: set[str] = set()
        queue = list(direct.get(start, []))
        while queue:
            n = queue.pop()
            if n in visited:
                continue
            visited.add(n)
            queue.extend(direct.get(n, []))
        transitive[start] = visited

    return transitive


def build_pretrain_static_wbc(pretrain_files: list[Path]) -> dict[Path, set[str]]:
    """Map each pretrain file → set of wbc module stems it statically imports (Pattern B)."""
    result: dict[Path, set[str]] = {}
    for pf in pretrain_files:
        wbc_stems: set[str] = set()
        try:
            tree = ast.parse(pf.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    # e.g. "from .qwen3_vl_workload_base_configs import ..."
                    mod = node.module.lstrip(".")
                    if "_workload_base_configs" in mod:
                        wbc_stems.add(mod)
        except SyntaxError:
            pass
        result[pf] = wbc_stems
    return result


# ---------------------------------------------------------------------------
# AST helpers for wbc analysis
# ---------------------------------------------------------------------------


def build_reverse_deps(wbc_file: Path) -> dict[str, set[str]]:
    """Build reverse dep graph: {const: set_of_constants_that_reference_it_on_their_RHS}."""
    rdeps: dict[str, set[str]] = defaultdict(set)
    try:
        tree = ast.parse(wbc_file.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if not node.targets or not isinstance(node.targets[0], ast.Name):
                    continue
                lhs = node.targets[0].id
                for n in ast.walk(node.value):
                    if isinstance(n, ast.Name):
                        rdeps[n.id].add(lhs)
    except SyntaxError:
        pass
    return dict(rdeps)


def bfs_propagate(seeds: set[str], rdeps: dict[str, set[str]]) -> set[str]:
    """BFS through reverse dep graph from seed constants."""
    visited = set(seeds)
    queue = list(seeds)
    while queue:
        n = queue.pop()
        for dep in rdeps.get(n, set()):
            if dep not in visited:
                visited.add(dep)
                queue.append(dep)
    return visited


def get_assignment_names_at_lines(wbc_file: Path, line_nos: set[int]) -> set[str]:
    """Find top-level Assign target names whose AST span overlaps line_nos."""
    names: set[str] = set()
    try:
        tree = ast.parse(wbc_file.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if not node.targets or not isinstance(node.targets[0], ast.Name):
                    continue
                span = set(range(node.lineno, node.end_lineno + 1))
                if span & line_nos:
                    names.add(node.targets[0].id)
    except SyntaxError:
        pass
    return names


# ---------------------------------------------------------------------------
# Naming convention: constant → function name + precision
# ---------------------------------------------------------------------------


def pretrain_const_to_func_and_precision(const: str) -> tuple[str, str | None] | None:
    """Map a PRETRAIN_CONFIG constant → (func_name, precision).

    Precision is None when the constant has no explicit precision token
    (e.g. DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1 has no precision between GPU and version).

    e.g. GPT_OSS_120B_PRETRAIN_CONFIG_B200_BF16_V1       → ("gpt_oss_120b_pretrain_config_b200", "BF16")
         QWEN3_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_CS_V1 → ("qwen3_235b_a22b_pretrain_config_gb300", "FP8_CS")
         DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1            → ("deepseek_v3_pretrain_config_gb300", None)
    """
    if "_PRETRAIN_CONFIG_" not in const:
        return None
    parts = const.split("_PRETRAIN_CONFIG_")
    model_recipe = parts[0].lower()
    tokens = parts[1].split("_")
    gpu = tokens[0].lower()
    func_name = f"{model_recipe}_pretrain_config_{gpu}"
    # Strip gpu (tokens[0]) and version suffix (last token matching V\d+)
    precision_tokens = tokens[1:]
    if precision_tokens and re.match(r"^V\d+$", precision_tokens[-1]):
        precision_tokens = precision_tokens[:-1]
    precision = "_".join(precision_tokens) if precision_tokens else None
    return func_name, precision


def build_func_precisions_index(
    pretrain_index: dict[Path, list[str]],
) -> dict[tuple[str, str], set[str]]:
    """Build index: (file.name, func_name) → set of all known precisions from wbc files."""
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for wbc_file in CONFIGS_DIR.rglob("*_workload_base_configs.py"):
        family_dir = wbc_file.parent
        try:
            tree = ast.parse(wbc_file.read_text())
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    if not node.targets or not isinstance(node.targets[0], ast.Name):
                        continue
                    result = pretrain_const_to_func_and_precision(node.targets[0].id)
                    if result is None:
                        continue
                    func_name, precision = result
                    if precision is None:
                        continue  # no explicit precision token — rely on bare fallback
                    pf = find_pretrain_file_for_func(func_name, family_dir, pretrain_index)
                    if pf is not None:
                        index[(pf.name, func_name)].add(precision)
        except SyntaxError:
            pass
    return dict(index)


def expand_with_precisions(
    file_name: str,
    func_name: str,
    func_precisions: dict[tuple[str, str], set[str]],
) -> set[str]:
    """Return one entry per known precision; fall back to bare func entry if none known."""
    precisions = func_precisions.get((file_name, func_name), set())
    if precisions:
        return {f"{file_name}::{func_name}::{p}" for p in precisions}
    return {f"{file_name}::{func_name}"}


def find_pretrain_file_for_func(
    func_name: str,
    family_dir: Path,
    pretrain_index: dict[Path, list[str]],
) -> Path | None:
    """Find the pretrain file in family_dir containing func_name."""
    for pf, funcs in pretrain_index.items():
        if pf.parent == family_dir and func_name in funcs:
            return pf
    return None


# ---------------------------------------------------------------------------
# Phase 4: Compute affected functions
# ---------------------------------------------------------------------------


def compute_affected_from_utils(
    changed_module: str,
    utils_transitive: dict[str, set[str]],
    pretrain_utils_deps: dict[Path, set[str]],
    pretrain_index: dict[Path, list[str]],
    func_precisions: dict[tuple[str, str], set[str]],
) -> set[str]:
    """Return all pretrain functions affected by a changed utils module."""
    # All utils modules that (directly or transitively) depend on changed_module
    effective = {changed_module}
    for m, all_deps in utils_transitive.items():
        if changed_module in all_deps:
            effective.add(m)

    affected: set[str] = set()
    for pf, deps in pretrain_utils_deps.items():
        if deps & effective:
            for func in pretrain_index.get(pf, []):
                affected |= expand_with_precisions(pf.name, func, func_precisions)
    return affected


def compute_affected_from_wbc(
    wbc_file: Path,
    touched_lines: set[int],
    raw_lines: list[tuple[str, str]],
    pretrain_index: dict[Path, list[str]],
    pretrain_static_wbc: dict[Path, set[str]],
    func_precisions: dict[tuple[str, str], set[str]],
) -> set[str]:
    """Compute affected pretrain functions from changes in a wbc file."""
    affected: set[str] = set()
    family_dir = wbc_file.parent

    # Step 1: Find directly-changed assignment names
    pos_lines = {ln for ln in touched_lines if ln > 0}
    directly_changed = get_assignment_names_at_lines(wbc_file, pos_lines) if pos_lines else set()

    # For deleted lines: regex scan of raw '-' lines (fallback, may over-report)
    for sign, text in raw_lines:
        if sign == "-":
            m = re.match(r"^\s*([A-Z][A-Z0-9_]+)\s*=", text)
            if m:
                directly_changed.add(m.group(1))

    # Step 2: BFS through reverse dep graph
    rdeps = build_reverse_deps(wbc_file)
    all_affected = bfs_propagate(directly_changed, rdeps)

    # Step 3: Filter to PRETRAIN_CONFIG constants
    pretrain_constants = {c for c in all_affected if "_PRETRAIN_CONFIG_" in c}

    # Step 4: Map each constant → pretrain function + precision via naming convention
    for const in pretrain_constants:
        result = pretrain_const_to_func_and_precision(const)
        if result is None:
            continue
        func_name, precision = result
        if precision is None:
            continue  # base config constant (no precision token) — tagged derivatives caught separately
        pf = find_pretrain_file_for_func(func_name, family_dir, pretrain_index)
        if pf is not None:
            affected.add(f"{pf.name}::{func_name}::{precision}")
        else:
            print(
                f"WARNING: no function matching '{func_name}' for constant '{const}'",
                file=sys.stderr,
            )

    # Step 5: Pattern B — pretrain files with static imports from this wbc
    wbc_stem = wbc_file.stem
    for pf, wbc_stems in pretrain_static_wbc.items():
        if wbc_stem in wbc_stems:
            try:
                tree = ast.parse(pf.read_text())
                for func in ast.walk(tree):
                    if isinstance(func, ast.FunctionDef) and "_pretrain_config_" in func.name:
                        body_names = {n.id for n in ast.walk(func) if isinstance(n, ast.Name)}
                        if body_names & pretrain_constants:
                            affected |= expand_with_precisions(pf.name, func.name, func_precisions)
            except SyntaxError:
                pass

    return affected


def compute_affected_from_pretrain(
    pretrain_file: Path,
    touched_lines: set[int],
    pretrain_index: dict[Path, list[str]],
    func_precisions: dict[tuple[str, str], set[str]],
) -> set[str]:
    """Compute affected pretrain functions from changed lines in a pretrain file."""
    affected: set[str] = set()

    try:
        tree = ast.parse(pretrain_file.read_text())
    except (SyntaxError, FileNotFoundError):
        for func in pretrain_index.get(pretrain_file, []):
            affected |= expand_with_precisions(pretrain_file.name, func, func_precisions)
        return affected

    pos_lines = {ln for ln in touched_lines if ln > 0}
    all_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

    changed_funcs: list[ast.FunctionDef] = []
    for func in all_funcs:
        func_lines = set(range(func.lineno, func.end_lineno + 1))
        if func_lines & pos_lines:
            changed_funcs.append(func)

    for func in changed_funcs:
        if "_pretrain_config_" in func.name:
            affected |= expand_with_precisions(pretrain_file.name, func.name, func_precisions)
        else:
            # Changed helper — find all *_pretrain_config_* callers in same file
            for other_func in all_funcs:
                if "_pretrain_config_" not in other_func.name:
                    continue
                for call in ast.walk(other_func):
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == func.name:
                        affected |= expand_with_precisions(pretrain_file.name, other_func.name, func_precisions)
                        break

    return affected


def compute_affected_from_init(
    init_file: Path,
    raw_lines: list[tuple[str, str]],
    pretrain_index: dict[Path, list[str]],
    func_precisions: dict[tuple[str, str], set[str]],
) -> set[str]:
    """Compute affected pretrain functions from changed lines in an __init__.py."""
    family_dir = init_file.parent

    all_family_funcs: set[str] = set()
    for pf, funcs in pretrain_index.items():
        if pf.parent == family_dir:
            for f in funcs:
                all_family_funcs |= expand_with_precisions(pf.name, f, func_precisions)

    # touched_entries: specific (func_name, precision_or_None) pairs from changed lines
    touched_entries: set[tuple[str, str | None]] = set()
    non_import_change = False

    for _sign, text in raw_lines:
        # lowercase pretrain_config function name references (no precision info here)
        names = re.findall(r"\b\w+_pretrain_config_\w+\b", text)
        if names:
            for n in names:
                touched_entries.add((n, None))
        elif re.search(r"^\s*(from|import)\s+", text):
            pass  # import line — not a structural change
        elif re.search(r"\b\w+_PRETRAIN_CONFIG_\w+\b", text):
            # uppercase constant name in a multi-line import/export block — map to (func, precision)
            for const in re.findall(r"\b(\w+_PRETRAIN_CONFIG_\w+)\b", text):
                result = pretrain_const_to_func_and_precision(const)
                if result:
                    touched_entries.add(result)  # (func_name, precision_or_None)
        elif text.strip() and not text.strip().startswith("#"):
            non_import_change = True

    if non_import_change:
        return all_family_funcs

    if touched_entries:
        affected: set[str] = set()
        for func_name, precision in touched_entries:
            for pf, funcs in pretrain_index.items():
                if pf.parent == family_dir and func_name in funcs:
                    if precision is not None:
                        affected.add(f"{pf.name}::{func_name}::{precision}")
                    else:
                        affected |= expand_with_precisions(pf.name, func_name, func_precisions)
        return affected if affected else all_family_funcs

    return all_family_funcs


def compute_affected_from_wbc_files_mode(
    wbc_file: Path,
    pretrain_index: dict[Path, list[str]],
    pretrain_static_wbc: dict[Path, set[str]],
    func_precisions: dict[tuple[str, str], set[str]],
) -> set[str]:
    """Conservative fallback (--files mode): report all pretrain functions for a wbc file."""
    affected: set[str] = set()
    family_dir = wbc_file.parent

    try:
        tree = ast.parse(wbc_file.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if not node.targets or not isinstance(node.targets[0], ast.Name):
                    continue
                result = pretrain_const_to_func_and_precision(node.targets[0].id)
                if result is None:
                    continue
                func_name, precision = result
                if precision is None:
                    continue  # base config constant — skip, tagged derivatives are separate entries
                pf = find_pretrain_file_for_func(func_name, family_dir, pretrain_index)
                if pf is not None:
                    affected.add(f"{pf.name}::{func_name}::{precision}")
    except SyntaxError:
        pass

    # Pattern B
    wbc_stem = wbc_file.stem
    for pf, wbc_stems in pretrain_static_wbc.items():
        if wbc_stem in wbc_stems:
            for func in pretrain_index.get(pf, []):
                affected |= expand_with_precisions(pf.name, func, func_precisions)

    return affected


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_path(filepath: str) -> Path | None:
    """Resolve a repo-relative filepath to an absolute path (returns None if missing)."""
    abs_path = REPO_ROOT / filepath
    return abs_path if abs_path.exists() else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse stdin diff or file list and print affected pretrain config functions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        action="store_true",
        help="Conservative mode: read newline-separated file names (not diff) from stdin",
    )
    args = parser.parse_args()

    stdin_text = sys.stdin.read()

    # Build indices (relative to script location — works from any CWD)
    pretrain_files = discover_pretrain_files()
    pretrain_index = build_pretrain_index(pretrain_files)
    pretrain_utils_deps = build_pretrain_utils_deps(pretrain_files)
    utils_transitive = build_utils_transitive_deps()
    pretrain_static_wbc = build_pretrain_static_wbc(pretrain_files)
    func_precisions = build_func_precisions_index(pretrain_index)

    # wbc analysis is authoritative for precision: when wbc changes identify specific precisions
    # for a (file, func) pair, non-wbc paths (pretrain/init/utils) must not expand that pair to
    # all precisions — they would add noise (e.g. BF16) when only FP8_MX was actually changed.
    wbc_affected: set[str] = set()
    other_affected: set[str] = set()

    if args.files:
        for filepath in parse_files_list(stdin_text):
            kind = classify_file(filepath)
            abs_path = resolve_path(filepath)

            if kind == "utils":
                module_stem = Path(filepath).stem
                other_affected |= compute_affected_from_utils(
                    module_stem, utils_transitive, pretrain_utils_deps, pretrain_index, func_precisions
                )
            elif kind == "wbc":
                if abs_path:
                    wbc_affected |= compute_affected_from_wbc_files_mode(
                        abs_path, pretrain_index, pretrain_static_wbc, func_precisions
                    )
            elif kind == "pretrain":
                if abs_path:
                    for func in pretrain_index.get(abs_path, []):
                        other_affected |= expand_with_precisions(abs_path.name, func, func_precisions)
            elif kind == "init":
                if abs_path:
                    family_dir = abs_path.parent
                    for pf, funcs in pretrain_index.items():
                        if pf.parent == family_dir:
                            for f in funcs:
                                other_affected |= expand_with_precisions(pf.name, f, func_precisions)
    else:
        touched_lines_map, raw_lines_map = parse_diff(stdin_text)

        for filepath, touched_lines in touched_lines_map.items():
            kind = classify_file(filepath)
            abs_path = resolve_path(filepath)
            raw_lines = raw_lines_map.get(filepath, [])

            if kind == "utils":
                module_stem = Path(filepath).stem
                other_affected |= compute_affected_from_utils(
                    module_stem, utils_transitive, pretrain_utils_deps, pretrain_index, func_precisions
                )
            elif kind == "wbc":
                if abs_path:
                    wbc_affected |= compute_affected_from_wbc(
                        abs_path, touched_lines, raw_lines, pretrain_index, pretrain_static_wbc, func_precisions
                    )
                else:
                    # Deleted file: conservative — report all funcs+precisions in this family
                    family_dir = CONFIGS_DIR / Path(filepath).parent.name
                    for pf, funcs in pretrain_index.items():
                        if pf.parent == family_dir:
                            for f in funcs:
                                wbc_affected |= expand_with_precisions(pf.name, f, func_precisions)
            elif kind == "pretrain":
                if abs_path:
                    other_affected |= compute_affected_from_pretrain(
                        abs_path, touched_lines, pretrain_index, func_precisions
                    )
            elif kind == "init":
                if abs_path:
                    other_affected |= compute_affected_from_init(
                        abs_path, raw_lines, pretrain_index, func_precisions
                    )

    # For (file, func) pairs already covered by wbc with specific precisions, suppress the
    # all-precision expansion that non-wbc paths produce — it would add irrelevant precisions.
    wbc_funcs = {tuple(e.split("::")[:2]) for e in wbc_affected}
    affected = wbc_affected | {e for e in other_affected if tuple(e.split("::")[:2]) not in wbc_funcs}

    if affected:
        for entry in sorted(affected):
            print(entry)
    else:
        print("(no pretrain configs affected)")


if __name__ == "__main__":
    main()
