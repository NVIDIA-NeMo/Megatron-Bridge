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

"""Tests for the rank-local performance bootstrap and training entrypoints."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

import bootstrap
import run_recipe
import run_script
import setup_experiment
from utils import utils


def test_nemo_ci_legacy_variants_select_the_default_flat_recipe_name():
    assert utils._recipe_variant_suffix("v1") == ""
    assert utils._recipe_variant_name("v1") is None
    assert utils._recipe_variant_suffix("v2") == ""
    assert utils._recipe_variant_name("v2") is None
    assert utils._recipe_variant_suffix("large_scale") == "_large_scale"


def test_gpu_tuning_options_are_not_forwarded_to_rank_local_scripts():
    assert setup_experiment._filter_run_script_args(
        [
            "--enable_vboost",
            "true",
            "--lock_gpu_freq=1200",
            "--deterministic",
        ]
    ) == ["--deterministic"]


def test_setup_experiment_uses_one_bootstrap_entrypoint():
    setup_path = _PERF_SCRIPTS_DIR / "setup_experiment.py"
    tree = ast.parse(setup_path.read_text())
    entrypoints = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.startswith("ENTRYPOINT")
    }

    assert entrypoints == {"ENTRYPOINT_BOOTSTRAP": "bootstrap.py"}
    setup_source = setup_path.read_text()
    argument_parser_source = (_PERF_SCRIPTS_DIR / "argument_parser.py").read_text()
    assert "--require-env-bootstrap" not in setup_source
    assert "--require-env-bootstrap" not in argument_parser_source
    assert not (_PERF_SCRIPTS_DIR / "run_script_with_env.py").exists()


def test_bootstrap_exports_flat_recipe_environment_before_exec(monkeypatch):
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
        use_recipes=False,
    )
    parser = SimpleNamespace(parse_known_args=lambda: (args, []))
    recipe = SimpleNamespace(env_vars={"CUDA_DEVICE_MAX_CONNECTIONS": 1})
    calls = []

    monkeypatch.setattr(bootstrap, "parse_cli_args", lambda: parser)

    def get_recipe(parsed_args, overrides):
        calls.append(("recipe", parsed_args, overrides))
        return recipe

    def exec_runner(executable, argv, env):
        calls.append(("exec", executable, argv, env))

    monkeypatch.setattr(run_script, "_prepare_perf_recipe", get_recipe)
    monkeypatch.setattr(bootstrap, "_apply_recipe_environment", lambda config: calls.append(("environment", config)))
    monkeypatch.setattr(bootstrap.os, "execvpe", exec_runner)

    bootstrap.main()

    assert [call[0] for call in calls] == ["recipe", "environment", "exec"]
    assert calls[1] == ("environment", recipe)
    assert calls[2][2][1].endswith("run_script.py")


def test_bootstrap_exec_preserves_argv_and_environment(monkeypatch):
    original_argv = [
        "bootstrap.py",
        "--wandb_experiment_name",
        "name with spaces",
        "+env_vars.EXTRA={nested:value}",
    ]
    calls = []

    monkeypatch.setattr(bootstrap.sys, "argv", original_argv)
    monkeypatch.setenv("BOOTSTRAP_SENTINEL", "preserved")
    expected_environment = dict(bootstrap.os.environ)
    monkeypatch.setattr(bootstrap.os, "execvpe", lambda executable, argv, env: calls.append((executable, argv, env)))

    bootstrap._exec_training(bootstrap.ENTRYPOINT_RECIPE)

    assert len(calls) == 1
    executable, argv, environment = calls[0]
    assert executable == bootstrap.sys.executable
    assert argv == [
        bootstrap.sys.executable,
        str((_PERF_SCRIPTS_DIR / "run_recipe.py").resolve()),
        *original_argv[1:],
    ]
    assert environment == expected_environment
    assert environment is not bootstrap.os.environ


def test_compatibility_overrides_preserve_legacy_manual_gc_defaults():
    from utils.overrides import _set_common_perf_overrides

    recipe = SimpleNamespace(
        train=SimpleNamespace(train_iters=0, eval_iters=1, manual_gc=False, manual_gc_interval=0),
        checkpoint=SimpleNamespace(save="checkpoint"),
        logger=SimpleNamespace(log_interval=10, tensorboard_dir="tensorboard"),
        ddp=SimpleNamespace(check_for_nan_in_grad=True, check_for_large_grads=True),
        rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=True),
        scheduler=SimpleNamespace(lr_decay_iters=0, lr_warmup_iters=0),
        model=SimpleNamespace(
            apply_rope_fusion=False,
            cross_entropy_fusion_impl="native",
            moe_flex_dispatcher_backend=None,
        ),
    )

    _set_common_perf_overrides(recipe)

    assert recipe.train.manual_gc is True
    assert recipe.train.manual_gc_interval == 100


def test_gpu_tuning_options_are_applied_directly_to_slurm_executor():
    executor = SimpleNamespace(
        nodes=2,
        tunnel=SimpleNamespace(job_dir="/job/dir"),
        setup_lines="existing setup\n",
    )

    utils.configure_slurm_gpu_tuning(
        executor,
        enable_vboost=True,
        lock_gpu_freq=1200,
    )

    assert executor.setup_lines.startswith("existing setup\n")
    assert "sudo nvidia-smi boost-slider --vboost 1" in executor.setup_lines
    assert "--ntasks=2" in executor.setup_lines
    assert "sudo nvidia-smi -lgc 1200" in executor.setup_lines
    assert "--ntasks-per-node=1" in executor.setup_lines


def test_run_script_main_runs_training_once(monkeypatch):
    args = SimpleNamespace()
    cli_overrides = ["model.num_layers=1"]
    parser = SimpleNamespace(parse_known_args=lambda: (args, cli_overrides))
    calls = []

    monkeypatch.setattr(run_script, "parse_cli_args", lambda: parser)
    monkeypatch.setattr(
        run_script, "_run_training", lambda parsed_args, overrides: calls.append((parsed_args, overrides))
    )

    run_script.main()

    assert calls == [(args, cli_overrides)]


def test_run_recipe_main_runs_training_once(monkeypatch):
    args = SimpleNamespace()
    parser = SimpleNamespace(parse_known_args=lambda: (args, []))
    calls = []

    monkeypatch.setattr(run_recipe, "parse_cli_args", lambda: parser)
    monkeypatch.setattr(
        run_recipe, "_run_training", lambda parsed_args, overrides: calls.append((parsed_args, overrides))
    )

    run_recipe.main()

    assert calls == [(args, [])]


def test_training_entrypoints_do_not_self_exec():
    for script_name in ("run_script.py", "run_recipe.py"):
        source = (_PERF_SCRIPTS_DIR / script_name).read_text()
        assert "ENV_BOOTSTRAP_MARKER" not in source
        assert "execvpe" not in source


def test_training_entrypoints_leave_process_group_lifecycle_to_training():
    for script_name in ("run_script.py", "run_recipe.py"):
        source = (_PERF_SCRIPTS_DIR / script_name).read_text()
        assert "destroy_process_group" not in source
        assert "torch.distributed.barrier" not in source


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
