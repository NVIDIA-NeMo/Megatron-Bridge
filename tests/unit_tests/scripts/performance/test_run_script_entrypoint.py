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
import setup_experiment


def test_gpu_tuning_options_are_not_forwarded_to_rank_local_scripts():
    assert setup_experiment._filter_run_script_args(
        [
            "--enable_vboost",
            "true",
            "--lock_gpu_freq=1200",
            "--deterministic",
        ]
    ) == ["--deterministic"]


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
    )
    parser = SimpleNamespace(parse_known_args=lambda: (args, []))
    recipe = SimpleNamespace(env_vars={"CUDA_DEVICE_MAX_CONNECTIONS": 1})
    calls = []

    monkeypatch.setattr(run_script, "parse_cli_args", lambda: parser)

    def get_recipe(**kwargs):
        calls.append(("recipe", kwargs))
        return recipe

    def exec_runner(executable, argv, env):
        calls.append(("exec", executable, argv, env))

    monkeypatch.setattr(run_script, "get_perf_recipe_for_environment", get_recipe)
    monkeypatch.setattr(run_script, "_apply_perf_recipe_overrides", lambda config, _overrides, _args: config)
    monkeypatch.setattr(run_script, "_apply_recipe_environment", lambda config: calls.append(("environment", config)))
    monkeypatch.setattr(run_script.os, "execvpe", exec_runner)

    run_script.main()

    assert [call[0] for call in calls] == ["recipe", "environment", "exec"]
    assert calls[1] == ("environment", recipe)
    assert calls[2][2][1].endswith("run_script.py")
    assert calls[2][3][run_script.ENV_BOOTSTRAP_MARKER] == str(run_script.os.getpid())


def test_gpu_tuning_options_are_applied_directly_to_slurm_executor():
    executor = SimpleNamespace(
        nodes=2,
        tunnel=SimpleNamespace(job_dir="/job/dir"),
        setup_lines="existing setup\n",
    )

    setup_experiment._configure_slurm_gpu_tuning(
        executor,
        enable_vboost=True,
        lock_gpu_freq=1200,
    )

    assert executor.setup_lines.startswith("existing setup\n")
    assert "sudo nvidia-smi boost-slider --vboost 1" in executor.setup_lines
    assert "--ntasks=2" in executor.setup_lines
    assert "sudo nvidia-smi -lgc 1200" in executor.setup_lines
    assert "--ntasks-per-node=1" in executor.setup_lines


def test_run_script_trains_only_after_environment_bootstrap(monkeypatch):
    args = SimpleNamespace()
    cli_overrides = ["model.num_layers=1"]
    parser = SimpleNamespace(parse_known_args=lambda: (args, cli_overrides))
    calls = []

    monkeypatch.setattr(run_script, "parse_cli_args", lambda: parser)
    monkeypatch.setenv(run_script.ENV_BOOTSTRAP_MARKER, str(run_script.os.getpid()))
    monkeypatch.setattr(run_script, "_bootstrap_recipe_environment", lambda *_: calls.append("bootstrap"))
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
    monkeypatch.setattr(
        run_script,
        "_bootstrap_recipe_environment",
        lambda parsed_args, overrides: calls.append((parsed_args, overrides)),
    )
    monkeypatch.setattr(run_script, "_run_training", lambda *_: calls.append("training"))

    run_script.main()

    assert calls == [(args, [])]


def test_run_script_defers_training_framework_imports():
    tree = ast.parse((_PERF_SCRIPTS_DIR / "run_script.py").read_text())
    top_level_imports = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    top_level_imports.update(
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert "torch" not in top_level_imports
    assert not any(module.startswith("megatron.bridge") for module in top_level_imports)


def test_run_recipe_defers_training_framework_imports():
    tree = ast.parse((_PERF_SCRIPTS_DIR / "run_recipe.py").read_text())
    top_level_imports = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    top_level_imports.update(
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert "torch" not in top_level_imports
    assert not any(module.startswith("megatron.bridge") for module in top_level_imports)
