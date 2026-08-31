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
"""Doc-consistency regression tests for shipped model verification cards.

Every verification card publishes literal shell commands a reader is meant to
copy and run. These tests check the published argv against the launchers that
own it: GPU conversion topologies must decompose over the published world size,
and an inference command must not carry a flag that belongs to the conversion
launcher's namespace.

Deliberately stdlib-only apart from PyYAML (no torch / megatron import) so it
scans source files directly and runs anywhere, including without the GPU stack.
"""

import ast
import shlex
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CARDS_DIR = REPO_ROOT / "examples" / "model_verification_cards"
CONVERSION_ARGUMENTS = REPO_ROOT / "scripts" / "conversion" / "arguments.py"
SETUP_CONVERSION = REPO_ROOT / "scripts" / "conversion" / "setup_conversion.py"
SETUP_INFERENCE = REPO_ROOT / "scripts" / "inference" / "setup_inference.py"


def _declared_options(source_path: Path, *, append_only: bool = False) -> set[str]:
    """Return option strings an argparse parser in the module declares."""
    options: set[str] = set()
    for node in ast.walk(ast.parse(source_path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if append_only:
            actions = [kw.value for kw in node.keywords if kw.arg == "action"]
            if not any(isinstance(a, ast.Constant) and a.value == "append" for a in actions):
                continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("-"):
                    options.add(argument.value)
    return options


def _inference_tasks() -> dict[str, Path]:
    """Return the launcher's task name to repository entry point mapping."""
    tree = ast.parse(SETUP_INFERENCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "INFERENCE_TASKS" for t in node.targets):
            continue
        return {key.value: REPO_ROOT / value.args[0].value for key, value in zip(node.value.keys, node.value.values)}
    raise AssertionError(f"INFERENCE_TASKS is no longer a module-level mapping in {SETUP_INFERENCE}")


def _walk_commands(node, path: str = ""):
    """Yield (card-relative location, command) for every published command."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("command", "commands"):
                if isinstance(value, str):
                    yield f"{path}/{key}", value
                elif isinstance(value, list):
                    for index, entry in enumerate(value):
                        if isinstance(entry, str):
                            yield f"{path}/{key}[{index}]", entry
            else:
                yield from _walk_commands(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, entry in enumerate(node):
            yield from _walk_commands(entry, f"{path}[{index}]")


def _published_commands():
    """Yield (leaf identifier, argv tokens) for every command in every card."""
    for card in sorted(CARDS_DIR.glob("*/card.yaml")):
        document = yaml.safe_load(card.read_text(encoding="utf-8"))
        for location, command in _walk_commands(document):
            tokens = shlex.split(command)
            if tokens:
                yield f"{card.parent.name}{location}", tokens


def _flag_value(tokens: list[str], flag: str, default: int | None) -> int | None:
    """Return the last integer value passed to a flag, or the launcher default."""
    value = default
    for index, token in enumerate(tokens[:-1]):
        if token == flag:
            value = int(tokens[index + 1])
    return value


def _selected_task(tokens: list[str]) -> str:
    task = "text-generation"
    for index, token in enumerate(tokens[:-1]):
        if token == "--task":
            task = tokens[index + 1]
    return task


def test_published_gpu_conversion_commands_decompose_over_the_published_world_size():
    offenders = []
    for leaf, tokens in _published_commands():
        if "convert.sh" not in tokens[0]:
            continue
        device = "gpu"
        for index, token in enumerate(tokens[:-1]):
            if token == "--device":
                device = tokens[index + 1]
        if device != "gpu":
            continue
        gpus_per_node = _flag_value(tokens, "--gpus-per-node", None)
        assert gpus_per_node is not None, f"{leaf}: GPU conversion requires --gpus-per-node"
        world_size = _flag_value(tokens, "--nodes", 1) * gpus_per_node
        pipeline = _flag_value(tokens, "--pp", 1)
        model_parallel_size = _flag_value(tokens, "--tp", 1) * pipeline
        expert_parallel_size = _flag_value(tokens, "--etp", 1) * _flag_value(tokens, "--ep", 1) * pipeline
        if world_size % model_parallel_size or world_size % expert_parallel_size:
            offenders.append(
                f"{leaf}: world size {world_size} is not divisible by TP*PP={model_parallel_size} "
                f"or ETP*EP*PP={expert_parallel_size}"
            )
    assert not offenders, "setup_conversion.py refuses these published commands: " + "; ".join(offenders)


def test_published_inference_commands_carry_no_conversion_launcher_flag():
    launcher_options = _declared_options(SETUP_INFERENCE)
    conversion_options = _declared_options(CONVERSION_ARGUMENTS) | _declared_options(SETUP_CONVERSION)
    tasks = _inference_tasks()
    offenders = []
    for leaf, tokens in _published_commands():
        if "infer.sh" not in tokens[0]:
            continue
        accepted = launcher_options | _declared_options(tasks[_selected_task(tokens)])
        for token in tokens[1:]:
            name = token.split("=", 1)[0]
            # Restricted to the conversion namespace because task scripts may
            # declare options through helpers this static scan does not follow.
            if name.startswith("--") and name not in accepted and name in conversion_options:
                offenders.append(f"{leaf}: {name} belongs to setup_conversion.py, not to infer.sh")
    assert not offenders, "these published inference commands die in the container: " + "; ".join(offenders)


def test_published_card_commands_do_not_repeat_a_flag():
    repeatable = _declared_options(CONVERSION_ARGUMENTS, append_only=True) | _declared_options(
        SETUP_INFERENCE, append_only=True
    )
    offenders = []
    for leaf, tokens in _published_commands():
        flags = [token.split("=", 1)[0] for token in tokens[1:] if token.startswith("--")]
        repeated = sorted({flag for flag in flags if flags.count(flag) > 1 and flag not in repeatable})
        if repeated:
            offenders.append(f"{leaf}: {', '.join(repeated)}")
    assert not offenders, "published commands repeat a flag: " + "; ".join(offenders)


if __name__ == "__main__":
    # Allow standalone RED-GREEN without pytest/torch:  python3 test_verification_card_published_commands.py
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
