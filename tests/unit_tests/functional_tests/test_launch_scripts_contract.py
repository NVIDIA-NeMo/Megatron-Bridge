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

"""Static contracts for functional-test launch scripts."""

from __future__ import annotations

import re
import shlex
import stat
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FUNCTIONAL_ROOT = REPO_ROOT / "tests" / "functional_tests"
LAUNCH_ROOT = FUNCTIONAL_ROOT / "launch_scripts"
TEST_GROUP_ROOT = FUNCTIONAL_ROOT / "test_groups"
EXPECTED_HARDWARE = {"gb200", "h100"}
EXPECTED_STATUSES = {"active", "flaky"}
LAUNCH_NAME_RE = re.compile(r"^L[0-2]_[A-Za-z0-9_]+\.sh$")
CI_TIMEOUT_RE = re.compile(r"^# CI_TIMEOUT=([1-9][0-9]*)$")
GPU_COUNT_RE = re.compile(r"^# GPU_COUNT=x([1-9][0-9]*)$")
NPROC_RE = re.compile(r"--nproc[-_]per[-_]node(?:=|\s+)([^\s\\]+)")
CUDA_VISIBLE_DEVICES_RE = re.compile(r"CUDA_VISIBLE_DEVICES=([^\s]+)")
TEST_GROUP_REF_RE = re.compile(r'(?:tests/functional_tests/)?test_groups/[^\s"\'\\|&;()]+')
APPROVED_DIRECT_PYTEST = "uv run python -m pytest"
DIRECT_PYTEST_FORMS = (
    ("uv", "run", "python", "-m", "pytest"),
    ("uv", "run", "pytest"),
    ("python", "-m", "pytest"),
    ("pytest",),
)
SHELL_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
STALE_FUNCTIONAL_MODEL_PATHS = ("tests/functional_tests/models/", "tests.functional_tests.models")
MAX_GPUS = 2


def _launch_scripts() -> list[Path]:
    return sorted(LAUNCH_ROOT.rglob("*.sh"))


def _repo_relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _format_paths(paths: list[Path]) -> str:
    return "\n".join(f"- {_repo_relative(path)}" for path in paths)


def _format_items(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _referenced_test_group_paths(script: Path) -> list[Path]:
    referenced_paths = []
    for _, logical_line in _logical_shell_lines(script.read_text(encoding="utf-8")):
        comment_index = _unquoted_comment_index(logical_line)
        shell_code = logical_line if comment_index is None else logical_line[:comment_index]
        for match in TEST_GROUP_REF_RE.findall(shell_code):
            normalized = match.removeprefix("tests/functional_tests/")
            normalized = normalized.split("::", maxsplit=1)[0]
            referenced_paths.append(FUNCTIONAL_ROOT / normalized)
    return referenced_paths


def _referenced_test_group_files() -> set[Path]:
    referenced_files = set()
    for script in _launch_scripts():
        for referenced_path in _referenced_test_group_paths(script):
            if referenced_path.is_dir():
                referenced_files.update(referenced_path.rglob("test_*.py"))
            elif referenced_path.is_file() and referenced_path.match("test_*.py"):
                referenced_files.add(referenced_path)
    return referenced_files


def _strip_shell_prefixes(tokens: list[str]) -> list[str]:
    tokens = list(tokens)
    while tokens:
        while tokens and SHELL_ASSIGNMENT_RE.match(tokens[0]):
            tokens.pop(0)
        command_name = Path(tokens[0]).name if tokens else ""
        if command_name in {"!", "command", "do", "if", "then"}:
            tokens.pop(0)
            continue
        if command_name == "env":
            tokens.pop(0)
            while tokens and tokens[0].startswith("-"):
                option = tokens.pop(0)
                if option in {"-C", "--chdir", "-u", "--unset"} and tokens:
                    tokens.pop(0)
                elif option in {"-S", "--split-string"} and tokens:
                    tokens = shlex.split(tokens.pop(0)) + tokens
                    break
                elif option.startswith("--split-string="):
                    tokens = shlex.split(option.split("=", maxsplit=1)[1]) + tokens
                    break
                elif option.startswith("-S") and len(option) > 2:
                    tokens = shlex.split(option[2:]) + tokens
                    break
            continue
        if command_name == "time":
            tokens.pop(0)
            while tokens and tokens[0].startswith("-"):
                option = tokens.pop(0)
                if option in {"-f", "--format", "-o", "--output"} and tokens:
                    tokens.pop(0)
            continue
        break
    return tokens


def _classify_direct_pytest_command(tokens: list[str]) -> str | None:
    normalized_tokens = list(tokens)
    if normalized_tokens:
        normalized_tokens[0] = Path(normalized_tokens[0]).name
    if normalized_tokens and normalized_tokens[0] == "pytest":
        return "pytest"

    for command_tokens in DIRECT_PYTEST_FORMS:
        if tuple(normalized_tokens[: len(command_tokens)]) == command_tokens:
            return " ".join(command_tokens)
    if "pytest" in normalized_tokens and not _is_nested_pytest(normalized_tokens):
        return "unclassified pytest"
    return None


def _is_nested_pytest(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if token != "pytest" or index == 0 or tokens[index - 1] != "-m":
            continue
        runner_tokens = {Path(runner).name for runner in tokens[: index - 1]}
        if any(runner in runner_tokens for runner in ("coverage", "ft_launcher", "torch.distributed.run")):
            return True
    return False


def _logical_shell_lines(contents: str) -> list[tuple[int, str]]:
    logical_lines = []
    fragments = []
    start_line = 1
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not fragments:
            start_line = line_number
        stripped = line.rstrip()
        if stripped.endswith("\\") and not _has_unquoted_comment(line):
            fragments.append(stripped[:-1])
            continue
        fragments.append(line)
        logical_lines.append((start_line, " ".join(fragments)))
        fragments = []
    if fragments:
        logical_lines.append((start_line, " ".join(fragments)))
    return logical_lines


def _has_unquoted_comment(line: str) -> bool:
    return _unquoted_comment_index(line) is not None


def _unquoted_comment_index(line: str) -> int | None:
    in_single_quote = False
    in_double_quote = False
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and not in_single_quote:
            escaped = True
            continue
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if (
            character == "#"
            and not in_single_quote
            and not in_double_quote
            and (index == 0 or line[index - 1].isspace() or line[index - 1] in ";&|()")
        ):
            return index
    return None


def _shell_command_segments(command: str) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    segments = []
    current = []
    for token in lexer:
        if token and all(character in ";&|()" for character in token):
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _wrapped_shell_command(tokens: list[str]) -> str | None:
    if not tokens or Path(tokens[0]).name not in {"bash", "sh"}:
        return None

    index = 1
    while index < len(tokens):
        option = tokens[index]
        if option in {"-O", "+O", "-o", "+o"}:
            index += 2
            continue
        if option == "-c" or (
            option.startswith("-") and not option.startswith("--") and "c" in option.removeprefix("-")
        ):
            return tokens[index + 1] if index + 1 < len(tokens) else ""
        index += 1
    return None


def _direct_pytest_commands(contents: str) -> list[tuple[int, str]]:
    commands = []
    for line_number, logical_line in _logical_shell_lines(contents):
        try:
            segments = _shell_command_segments(logical_line)
        except ValueError:
            if "pytest" in logical_line:
                commands.append((line_number, "unparseable pytest command"))
            continue
        for segment in segments:
            stripped_segment = _strip_shell_prefixes(segment)
            wrapped_command = _wrapped_shell_command(stripped_segment)
            if wrapped_command is not None:
                for _, command in _direct_pytest_commands(wrapped_command):
                    commands.append((line_number, command))
                continue
            command = _classify_direct_pytest_command(stripped_segment)
            if command is not None:
                commands.append((line_number, command))
    return commands


def _declared_gpu_counts(contents: str) -> tuple[list[int], list[str]]:
    counts = []
    unresolved = []

    for value in NPROC_RE.findall(contents):
        normalized = value.strip("\"'")
        if normalized.isdigit():
            counts.append(int(normalized))
        else:
            unresolved.append(f"nproc={value}")

    for value in CUDA_VISIBLE_DEVICES_RE.findall(contents):
        normalized = value.strip("\"'")
        if re.fullmatch(r"[0-9]+(?:,[0-9]+)*", normalized):
            counts.append(len(normalized.split(",")))
        else:
            unresolved.append(f"CUDA_VISIBLE_DEVICES={value}")

    return counts, unresolved


def _gpu_policy_error(contents: str) -> str | None:
    gpu_count_lines = [line for line in contents.splitlines() if line.startswith("# GPU_COUNT")]
    header_matches = [GPU_COUNT_RE.fullmatch(line) for line in gpu_count_lines]
    malformed_headers = [line for line, match in zip(gpu_count_lines, header_matches) if match is None]
    header_counts = [int(match.group(1)) for match in header_matches if match is not None]
    declared_counts, unresolved = _declared_gpu_counts(contents)
    has_valid_header = len(header_counts) == 1 and not malformed_headers and header_counts[0] <= MAX_GPUS

    problems = []
    if len(gpu_count_lines) > 1:
        problems.append("multiple GPU_COUNT headers")
    if malformed_headers:
        problems.append(f"malformed headers: {malformed_headers!r}")
    over_limit = [count for count in [*header_counts, *declared_counts] if count > MAX_GPUS]
    if over_limit:
        problems.append(f"counts above {MAX_GPUS}: {over_limit!r}")
    if unresolved and not has_valid_header:
        problems.append(f"unresolved without valid header: {unresolved!r}")
    return "; ".join(problems) or None


def test_launch_script_inventory_is_nonempty() -> None:
    assert LAUNCH_ROOT.is_dir(), f"Launch-script directory does not exist: {_repo_relative(LAUNCH_ROOT)}"
    assert _launch_scripts(), f"No launch scripts found under {_repo_relative(LAUNCH_ROOT)}"


def test_launch_scripts_use_supported_layout_and_tiered_names() -> None:
    invalid = []
    for script in _launch_scripts():
        relative_parts = script.relative_to(LAUNCH_ROOT).parts
        valid_layout = (
            len(relative_parts) == 3
            and relative_parts[0] in EXPECTED_HARDWARE
            and relative_parts[1] in EXPECTED_STATUSES
        )
        if not valid_layout or not LAUNCH_NAME_RE.fullmatch(script.name):
            invalid.append(script)

    assert not invalid, "Launch scripts must use <h100|gb200>/<active|flaky>/L{0,1,2}_*.sh layout:\n" + _format_paths(
        invalid
    )


def test_launch_scripts_are_executable() -> None:
    not_executable = [script for script in _launch_scripts() if not script.stat().st_mode & stat.S_IXUSR]

    assert not not_executable, "Launch scripts must be executable:\n" + _format_paths(not_executable)


def test_launch_scripts_use_valid_optional_timeout_headers() -> None:
    invalid = []
    for script in _launch_scripts():
        timeout_lines = [
            line for line in script.read_text(encoding="utf-8").splitlines() if line.startswith("# CI_TIMEOUT")
        ]
        if len(timeout_lines) > 1 or any(CI_TIMEOUT_RE.fullmatch(line) is None for line in timeout_lines):
            invalid.append(script)

    assert not invalid, "Optional timeout headers must use one '# CI_TIMEOUT=<positive integer>' line:\n" + (
        _format_paths(invalid)
    )


def test_launch_scripts_do_not_reference_old_functional_model_path() -> None:
    stale_references = []
    for script in _launch_scripts():
        contents = script.read_text(encoding="utf-8")
        if any(stale_path in contents for stale_path in STALE_FUNCTIONAL_MODEL_PATHS):
            stale_references.append(script)

    assert not stale_references, "Launch scripts must use tests/functional_tests/test_groups/models paths:\n" + (
        _format_paths(stale_references)
    )


def test_launch_scripts_reference_existing_test_group_paths() -> None:
    missing = []
    for script in _launch_scripts():
        for referenced_path in _referenced_test_group_paths(script):
            if not referenced_path.exists():
                missing.append(f"{_repo_relative(script)} -> {_repo_relative(referenced_path)}")

    assert not missing, "Launch scripts reference missing test_groups paths:\n" + _format_items(missing)


def test_test_group_reference_parser_ignores_shell_comments(tmp_path: Path) -> None:
    script = tmp_path / "launcher.sh"
    script.write_text(
        "# tests/functional_tests/test_groups/models/commented.py\n"
        "uv run python -m pytest tests/functional_tests/test_groups/models/active.py "
        "# test_groups/models/trailing_comment.py\n",
        encoding="utf-8",
    )

    assert _referenced_test_group_paths(script) == [FUNCTIONAL_ROOT / "test_groups/models/active.py"]


def test_all_test_group_files_have_a_launcher() -> None:
    """Every functional test group must be reachable from at least one launcher."""
    test_group_files = set(TEST_GROUP_ROOT.rglob("test_*.py"))
    unreferenced_files = sorted(test_group_files - _referenced_test_group_files())

    assert not unreferenced_files, "Functional test groups without a launcher:\n" + _format_paths(unreferenced_files)


def test_direct_pytest_commands_use_approved_module_form() -> None:
    invalid = []
    for script in _launch_scripts():
        for line_number, command in _direct_pytest_commands(script.read_text(encoding="utf-8")):
            if command != APPROVED_DIRECT_PYTEST:
                invalid.append(f"{_repo_relative(script)}:{line_number} -> {command}")

    assert not invalid, f"Direct pytest commands must use '{APPROVED_DIRECT_PYTEST}':\n" + _format_items(invalid)


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("pytest -q", [(1, "pytest")]),
        ("FOO=bar pytest -q", [(1, "pytest")]),
        ("env -i FOO=bar uv run pytest -q", [(1, "uv run pytest")]),
        ("time -p python -m pytest -q", [(1, "python -m pytest")]),
        ("FOO=bar uv run python -m pytest -q", [(1, APPROVED_DIRECT_PYTEST)]),
        ("FOO=bar && pytest -q", [(1, "pytest")]),
        ("setup; uv run pytest -q", [(1, "uv run pytest")]),
        ("uv run python \\\n            -m pytest -q", [(1, APPROVED_DIRECT_PYTEST)]),
        ("# comment \\\npytest -q", [(2, "pytest")]),
        ("echo setup # comment \\\npytest -q", [(2, "pytest")]),
        ("/usr/bin/pytest -q", [(1, "pytest")]),
        ("bash -c 'pytest -q'", [(1, "pytest")]),
        ("bash --norc -c 'pytest -q'", [(1, "pytest")]),
        ("bash -O extglob -c 'pytest -q'", [(1, "pytest")]),
        ("env -S 'pytest -q'", [(1, "pytest")]),
        ("env -S'pytest -q'", [(1, "pytest")]),
        ("env --split-string='pytest -q'", [(1, "pytest")]),
        ("env -C /tmp uv run python -m pytest -q", [(1, APPROVED_DIRECT_PYTEST)]),
        ("time -o /tmp/timing uv run python -m pytest -q", [(1, APPROVED_DIRECT_PYTEST)]),
        ("/usr/bin/env FOO=bar uv run python -m pytest -q", [(1, APPROVED_DIRECT_PYTEST)]),
        ("/usr/bin/time -p uv run python -m pytest -q", [(1, APPROVED_DIRECT_PYTEST)]),
        ("uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest", []),
        ("uv run ft_launcher --nproc-per-node=2 -m pytest", []),
        ("/opt/bin/ft_launcher --nproc-per-node=2 -m pytest", []),
        ("uv run python -m coverage run -m pytest", []),
        ("/usr/bin/coverage run -m pytest", []),
    ],
)
def test_direct_pytest_parser_handles_shell_syntax(contents: str, expected: list[tuple[int, str]]) -> None:
    assert _direct_pytest_commands(contents) == expected


def test_launch_scripts_use_at_most_two_gpus() -> None:
    invalid = []
    for script in _launch_scripts():
        contents = script.read_text(encoding="utf-8")
        error = _gpu_policy_error(contents)
        if error is not None:
            invalid.append(f"{_repo_relative(script)} ({error})")

    assert not invalid, f"Launch scripts must use at most {MAX_GPUS} GPUs:\n" + _format_items(invalid)


@pytest.mark.parametrize(
    ("contents", "expected_counts", "expected_unresolved"),
    [
        ("--nproc_per_node=2", [2], []),
        ('--nproc_per_node="$WORLD_SIZE"', [], ['nproc="$WORLD_SIZE"']),
        ('CUDA_VISIBLE_DEVICES="0,1"', [2], []),
        ('CUDA_VISIBLE_DEVICES="$DEVICES"', [], ['CUDA_VISIBLE_DEVICES="$DEVICES"']),
    ],
)
def test_gpu_count_parser_fails_closed(
    contents: str,
    expected_counts: list[int],
    expected_unresolved: list[str],
) -> None:
    assert _declared_gpu_counts(contents) == (expected_counts, expected_unresolved)


@pytest.mark.parametrize(
    ("contents", "is_valid"),
    [
        ("# GPU_COUNT=x2\n--nproc_per_node=2", True),
        ("# GPU_COUNT=x4", False),
        ('--nproc_per_node="$WORLD_SIZE"', False),
        ('# GPU_COUNT=x2\n--nproc_per_node="$WORLD_SIZE"', True),
        ("# GPU_COUNT=x2\n--nproc_per_node=4", False),
    ],
)
def test_gpu_policy_uses_headers_and_fails_closed(contents: str, is_valid: bool) -> None:
    assert (_gpu_policy_error(contents) is None) is is_valid
