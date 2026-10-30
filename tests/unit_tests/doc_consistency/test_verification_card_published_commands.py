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
an inference command must not carry a flag that belongs to the conversion
launcher's namespace, and a checkpoint a command loads (``checkpoint.load=``,
``dataset.dataloader_load=`` or ``--load_dir``) must be written by a command in the
same card.

Uses only pytest, PyYAML and the standard library (no torch / megatron import) so it
scans source files directly and runs anywhere, including without the GPU stack.
"""

import ast
import shlex
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import overload

import pytest
import yaml


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CARDS_DIR = REPO_ROOT / "examples" / "model_verification_cards"
CONVERSION_ARGUMENTS = REPO_ROOT / "scripts" / "conversion" / "arguments.py"
SETUP_CONVERSION = REPO_ROOT / "scripts" / "conversion" / "setup_conversion.py"
SETUP_INFERENCE = REPO_ROOT / "scripts" / "inference" / "setup_inference.py"


def _declared_options(source_path: Path, *, append_only: bool = False, function: str | None = None) -> set[str]:
    """Read parser options, including called argument helpers in Bridge source."""
    options: set[str] = set()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    scope: ast.AST = tree
    if function is not None:
        scope = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    called = {
        node.func.id for node in ast.walk(scope) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for statement in tree.body:
        if (
            not isinstance(statement, ast.ImportFrom)
            or not statement.module
            or not statement.module.startswith("megatron.bridge.")
        ):
            continue
        helper = REPO_ROOT / "src" / (statement.module.replace(".", "/") + ".py")
        for alias in statement.names:
            if alias.name.startswith("add_") and (alias.asname or alias.name) in called:
                options |= _declared_options(helper, append_only=append_only, function=alias.name)
    for node in ast.walk(scope):
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
        assert isinstance(node.value, ast.Dict)
        tasks = {}
        for key, value in zip(node.value.keys, node.value.values):
            assert isinstance(key, ast.Constant) and isinstance(key.value, str)
            assert isinstance(value, ast.Call) and value.args
            argument = value.args[0]
            assert isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            tasks[key.value] = REPO_ROOT / argument.value
        return tasks
    raise AssertionError(f"INFERENCE_TASKS is no longer a module-level mapping in {SETUP_INFERENCE}")


def _walk_commands(node: object, path: str = "") -> Iterator[tuple[str, str]]:
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


def _published_commands() -> Iterator[tuple[str, list[str]]]:
    """Yield (leaf identifier, argv tokens) for every command in every card."""
    for card in sorted(CARDS_DIR.glob("*/card.yaml")):
        document = yaml.safe_load(card.read_text(encoding="utf-8"))
        for location, command in _walk_commands(document):
            tokens = shlex.split(command)
            if tokens:
                yield f"{card.parent.name}{location}", tokens


@overload
def _flag_value(tokens: list[str], flag: str, default: int) -> int: ...


@overload
def _flag_value(tokens: list[str], flag: str, default: None) -> int | None: ...


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


def test_published_gpu_conversion_commands_decompose_over_the_published_world_size() -> None:
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


def test_published_inference_commands_carry_no_conversion_launcher_flag() -> None:
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
            # Restrict this check to conversion options; it is not a complete
            # validator for every task's dynamic argument declarations.
            if name.startswith("--") and name not in accepted and name in conversion_options:
                offenders.append(f"{leaf}: {name} belongs to setup_conversion.py, not to infer.sh")
    assert not offenders, "these published inference commands die in the container: " + "; ".join(offenders)


def test_published_card_commands_do_not_repeat_a_flag() -> None:
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


def _token_value(tokens: list[str], index: int) -> tuple[str, str]:
    name, _, value = tokens[index].partition("=")
    if not value and index + 1 < len(tokens):
        value = tokens[index + 1]
    return name, value.rstrip("/")


def _produced_by_card() -> dict[str, set[str]]:
    """Paths each card's published commands write, resolved per command.

    run_recipe.py applies convenience flags before trailing overrides, so a trailing
    ``checkpoint.save=`` replaces ``--save_dir`` in the same command.
    """
    producers = ("checkpoint.save", "dataset.dataloader_save", "--save_dir")
    conversion_outputs = {"import": "--megatron-path", "export": "--hf-path"}
    produced: dict[str, set[str]] = {}
    for leaf, tokens in _published_commands():
        card = leaf.split("/", 1)[0]
        is_conversion = len(tokens) > 1 and Path(tokens[0]).name == "convert.sh"
        output_flag = conversion_outputs.get(tokens[1]) if is_conversion else None
        outputs: dict[str, str] = {}
        for index in range(len(tokens)):
            name, value = _token_value(tokens, index)
            if name in producers or name == output_flag:
                outputs[name] = value
        if "checkpoint.save" in outputs:
            outputs.pop("--save_dir", None)
        produced.setdefault(card, set()).update(outputs.values())
    return produced


def _is_produced(path: str, written: set[str]) -> bool:
    return path in written or (Path(path).name.startswith("iter_") and str(Path(path).parent) in written)


def test_published_checkpoint_loads_come_from_the_same_card() -> None:
    """Every non-null load override reads a path that a command in the same card writes."""
    loads = ("checkpoint.load", "dataset.dataloader_load")
    produced = _produced_by_card()
    offenders = []
    for leaf, tokens in _published_commands():
        card = leaf.split("/", 1)[0]
        for index, token in enumerate(tokens):
            name, value = _token_value(tokens, index)
            if (
                name in loads
                and "=" in token
                and value != "null"
                and not _is_produced(value, produced.get(card, set()))
            ):
                offenders.append(f"{leaf}: {name}={value}")
    assert not offenders, "published commands load a path no command in their card produces: " + "; ".join(offenders)


def test_published_load_dirs_come_from_the_same_card() -> None:
    """Every published --load_dir reads a path that a command in the same card writes."""
    produced = _produced_by_card()
    offenders = []
    for leaf, tokens in _published_commands():
        card = leaf.split("/", 1)[0]
        for index in range(len(tokens)):
            name, value = _token_value(tokens, index)
            if name == "--load_dir" and not _is_produced(value, produced.get(card, set())):
                offenders.append(f"{leaf}: --load_dir {value}")
    assert not offenders, "published commands load a checkpoint no command in their card writes: " + "; ".join(
        offenders
    )


def test_declared_options_follow_only_called_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    helper = tmp_path / "src/megatron/bridge/arguments.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "def add_parallelism(parser):\n"
        "    parser.add_argument('--tp', type=int)\n"
        "    parser.add_argument('--mount', action='append')\n"
        "def add_unused(parser):\n"
        "    parser.add_argument('--executor')\n",
        encoding="utf-8",
    )
    script = tmp_path / "inference.py"
    script.write_text(
        "from megatron.bridge.arguments import add_parallelism as add_args, add_unused\n"
        "def build_parser(parser):\n"
        "    add_args(parser)\n",
        encoding="utf-8",
    )
    assert _declared_options(script) == {"--tp", "--mount"}
    assert _declared_options(script, append_only=True) == {"--mount"}


@pytest.mark.parametrize(
    ("command", "check_name", "message"),
    [
        (
            "./scripts/conversion/convert.sh export --device gpu --nodes 2 --gpus-per-node 4 --ep 8 --pp 4",
            "test_published_gpu_conversion_commands_decompose_over_the_published_world_size",
            "ETP\\*EP\\*PP=32",
        ),
        (
            "./scripts/inference/infer.sh --task model-comparison --executor slurm",
            "test_published_inference_commands_carry_no_conversion_launcher_flag",
            "--executor belongs to setup_conversion.py",
        ),
        (
            "./scripts/inference/infer.sh --task model-comparison --ep 4 --ep 4",
            "test_published_card_commands_do_not_repeat_a_flag",
            "repeat a flag",
        ),
        (
            "./scripts/training/train.sh --mode pretrain checkpoint.load=work/unproduced",
            "test_published_checkpoint_loads_come_from_the_same_card",
            "checkpoint.load=work/unproduced",
        ),
        (
            "./scripts/training/train.sh --mode sft --load_dir work/unproduced",
            "test_published_load_dirs_come_from_the_same_card",
            "--load_dir work/unproduced",
        ),
        (
            "./scripts/training/train.sh --mode sft --load_dir=work/unproduced/iter_0000250",
            "test_published_load_dirs_come_from_the_same_card",
            "--load_dir work/unproduced/iter_0000250",
        ),
    ],
)
def test_published_command_checks_reject_original_errors(
    command: str, check_name: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_published_commands", lambda: iter([("regression", shlex.split(command))]))
    with pytest.raises(AssertionError, match=message):
        getattr(module, check_name)()


def _check_loads(monkeypatch: pytest.MonkeyPatch, *commands: tuple[str, str]) -> None:
    module = sys.modules[__name__]
    monkeypatch.setattr(
        module, "_published_commands", lambda: iter((leaf, shlex.split(cmd)) for leaf, cmd in commands)
    )
    module.test_published_checkpoint_loads_come_from_the_same_card()


def test_checkpoint_load_rejects_export_only_source(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AssertionError, match="checkpoint.load=work/unproduced"):
        _check_loads(
            monkeypatch,
            (
                "card/export",
                "./scripts/conversion/convert.sh export --megatron-path work/unproduced --hf-path work/hf",
            ),
            ("card/train", "./scripts/training/train.sh checkpoint.load=work/unproduced"),
        )


def test_checkpoint_load_rejects_import_hf_source(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AssertionError, match="checkpoint.load=work/hf"):
        _check_loads(
            monkeypatch,
            ("card/import", "./scripts/conversion/convert.sh import --hf-path work/hf --megatron-path work/mcore"),
            ("card/train", "./scripts/training/train.sh checkpoint.load=work/hf"),
        )


@pytest.mark.parametrize("load", ["work/mcore", "work/mcore/iter_0000000"])
def test_checkpoint_load_accepts_import_output_and_iteration_child(load: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _check_loads(
        monkeypatch,
        ("card/import", "./scripts/conversion/convert.sh import --hf-path work/hf --megatron-path work/mcore"),
        ("card/train", f"./scripts/training/train.sh checkpoint.load={load} dataset.dataloader_load=null"),
    )


def test_checkpoint_load_rejects_import_from_another_card(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AssertionError, match="checkpoint.load=work/mcore"):
        _check_loads(
            monkeypatch,
            ("other/import", "./scripts/conversion/convert.sh import --hf-path work/hf --megatron-path work/mcore"),
            ("card/train", "./scripts/training/train.sh checkpoint.load=work/mcore"),
        )


def _check_load_dirs(monkeypatch: pytest.MonkeyPatch, *commands: tuple[str, str]) -> None:
    module = sys.modules[__name__]
    monkeypatch.setattr(
        module, "_published_commands", lambda: iter((leaf, shlex.split(cmd)) for leaf, cmd in commands)
    )
    module.test_published_load_dirs_come_from_the_same_card()


@pytest.mark.parametrize("load_dir", ["work/sft", "work/sft/iter_0000250"])
def test_published_load_dirs_accept_a_save_dir_from_the_same_card(
    load_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _check_load_dirs(
        monkeypatch,
        ("card/items/sft/command", "./scripts/training/train.sh --mode sft --save_dir work/sft"),
        ("card/items/resume/command", f"./scripts/training/train.sh --mode sft --load_dir {load_dir}"),
    )


@pytest.mark.parametrize(
    "save",
    [
        "--save_dir work/unwritten checkpoint.save=work/actual",
        "checkpoint.save=work/actual --save_dir work/unwritten",
    ],
)
def test_published_load_dirs_honor_checkpoint_save_override(save: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AssertionError, match="--load_dir work/unwritten"):
        _check_load_dirs(
            monkeypatch,
            ("card/train", f"./scripts/training/train.sh --mode sft {save}"),
            ("card/resume", "./scripts/training/train.sh --mode sft --load_dir work/unwritten"),
        )
    for load in ("work/actual", "work/actual/iter_0000250"):
        _check_load_dirs(
            monkeypatch,
            ("card/train", f"./scripts/training/train.sh --mode sft {save}"),
            ("card/resume", f"./scripts/training/train.sh --mode sft --load_dir {load}"),
        )


def test_published_load_dirs_accept_an_independent_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    _check_load_dirs(
        monkeypatch,
        ("card/train", "./scripts/training/train.sh --mode sft --save_dir work/unwritten checkpoint.save=work/actual"),
        ("card/other", "./scripts/training/train.sh --mode sft --save_dir work/unwritten"),
        ("card/resume", "./scripts/training/train.sh --mode sft --load_dir work/unwritten"),
    )


def test_checkpoint_load_honors_checkpoint_save_override(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AssertionError, match="checkpoint.load=work/unwritten"):
        _check_loads(
            monkeypatch,
            ("card/train", "./scripts/training/train.sh --save_dir work/unwritten checkpoint.save=work/actual"),
            ("card/resume", "./scripts/training/train.sh checkpoint.load=work/unwritten"),
        )
