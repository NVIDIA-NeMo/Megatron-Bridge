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

"""Tests for the single environment-aware performance entry point."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

import run_script


def test_setup_experiment_uses_run_script_for_every_perf_workload():
    setup_path = _PERF_SCRIPTS_DIR / "setup_experiment.py"
    tree = ast.parse(setup_path.read_text())
    entrypoints = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.startswith("ENTRYPOINT")
    }

    assert entrypoints == {
        "ENTRYPOINT_PERFORMANCE": "run_script.py",
        "ENTRYPOINT_RECIPE": "run_recipe.py",
    }
    setup_source = setup_path.read_text()
    argument_parser_source = (_PERF_SCRIPTS_DIR / "argument_parser.py").read_text()
    assert "--require-env-bootstrap" not in setup_source
    assert "--require-env-bootstrap" not in argument_parser_source
    assert not (_PERF_SCRIPTS_DIR / "run_script_with_env.py").exists()


def test_run_script_exports_recipe_environment_before_self_exec(monkeypatch):
    args = SimpleNamespace(
        model_family_name="qwen",
        model_recipe_name="qwen3_30b_a3b",
        task="pretrain",
        num_gpus=8,
        gpu="h100",
        compute_dtype="bf16",
        config_variant=None,
        moe_a2a_overlap=False,
        tensor_model_parallel_size=None,
        pipeline_model_parallel_size=None,
        context_parallel_size=None,
        expert_model_parallel_size=None,
        deterministic=False,
        env=[("NCCL_NVLS_ENABLE", "explicit"), ("USER_FLAG", "enabled")],
    )
    parser = SimpleNamespace(parse_known_args=lambda: (args, []))
    recipe = SimpleNamespace()
    workload = SimpleNamespace()
    calls = []

    monkeypatch.setattr(run_script, "parse_cli_args", lambda: parser)

    def get_recipe(**kwargs):
        calls.append(("recipe", kwargs))
        return recipe

    def build_workload(config, *, num_gpus):
        calls.append(("workload", config, num_gpus))
        return workload

    class FakePerfEnvPlugin:
        def __init__(self, **kwargs):
            calls.append(("plugin", kwargs))

        def setup_recipe_environment(self, task, executor, config):
            calls.append(("environment", task, executor, config))
            executor.env_vars["NCCL_NVLS_ENABLE"] = "plugin"
            executor.env_vars["PLUGIN_FLAG"] = "enabled"

    def exec_runner(executable, argv, env):
        calls.append(("exec", executable, argv, env))

    monkeypatch.setattr(run_script, "get_perf_recipe_for_environment", get_recipe)
    monkeypatch.setattr(run_script, "_workload_base_config_from_recipe", build_workload)
    monkeypatch.setattr(run_script, "PerfEnvPlugin", FakePerfEnvPlugin)
    monkeypatch.setattr(run_script.os, "execvpe", exec_runner)

    run_script.main()

    assert [call[0] for call in calls] == ["recipe", "workload", "plugin", "environment", "exec"]
    assert calls[1] == ("workload", recipe, 8)
    assert calls[3][3] is workload
    assert calls[4][2][1].endswith("run_script.py")
    assert calls[4][3][run_script.ENV_BOOTSTRAP_MARKER] == str(run_script.os.getpid())
    assert calls[4][3]["NCCL_NVLS_ENABLE"] == "explicit"
    assert calls[4][3]["USER_FLAG"] == "enabled"
    assert calls[4][3]["PLUGIN_FLAG"] == "enabled"


def test_run_script_trains_only_after_environment_bootstrap(monkeypatch):
    args = SimpleNamespace()
    cli_overrides = ["model.num_layers=1"]
    parser = SimpleNamespace(parse_known_args=lambda: (args, cli_overrides))
    calls = []

    monkeypatch.setattr(run_script, "parse_cli_args", lambda: parser)
    monkeypatch.setenv(run_script.ENV_BOOTSTRAP_MARKER, str(run_script.os.getpid()))
    monkeypatch.setattr(run_script, "_bootstrap_recipe_environment", lambda _: calls.append("bootstrap"))
    monkeypatch.setattr(
        run_script, "_run_training", lambda parsed_args, overrides: calls.append((parsed_args, overrides))
    )

    run_script.main()

    assert calls == [(args, cli_overrides)]


def test_run_script_ignores_stale_bootstrap_marker(monkeypatch):
    args = SimpleNamespace()
    parser = SimpleNamespace(parse_known_args=lambda: (args, []))
    calls = []

    monkeypatch.setattr(run_script, "parse_cli_args", lambda: parser)
    monkeypatch.setenv(run_script.ENV_BOOTSTRAP_MARKER, "stale-pid")
    monkeypatch.setattr(run_script, "_bootstrap_recipe_environment", lambda parsed_args: calls.append(parsed_args))
    monkeypatch.setattr(run_script, "_run_training", lambda *_: calls.append("training"))

    run_script.main()

    assert calls == [args]


def test_run_script_defers_training_framework_imports():
    tree = ast.parse((_PERF_SCRIPTS_DIR / "run_script.py").read_text())
    top_level_imports = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    top_level_imports.update(
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert "torch" not in top_level_imports
    assert not any(module.startswith("megatron.bridge") for module in top_level_imports)
