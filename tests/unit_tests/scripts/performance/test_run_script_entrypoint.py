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
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))


def _install_launcher_stubs() -> None:
    """Install import-time stubs for launcher-only optional dependencies."""
    nemo_run = types.ModuleType("nemo_run")

    class Script:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def to_command(self):
            return []

    class _RunObject:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    nemo_run.Plugin = object
    nemo_run.Script = Script
    nemo_run.SlurmExecutor = _RunObject
    nemo_run.LocalTunnel = _RunObject
    nemo_run.GitArchivePackager = _RunObject
    nemo_run.Packager = _RunObject
    nemo_run.KubeflowExecutor = _RunObject
    nemo_run.Torchrun = _RunObject

    nemo_run_config = types.ModuleType("nemo_run.config")
    nemo_run_config.get_nemorun_home = lambda: str(Path.home() / ".nemo_run")
    nemo_run_config.set_nemorun_home = lambda _path: None
    nemo_run.config = nemo_run_config

    launcher_module = types.ModuleType("nemo_run.core.execution.launcher")
    launcher_module.SlurmTemplate = _RunObject

    run_plugins = types.ModuleType("megatron.bridge.recipes.run_plugins")
    run_plugins.PreemptionPlugin = object

    for package_name in (
        "megatron",
        "megatron.bridge",
        "megatron.bridge.recipes",
        "megatron.bridge.recipes.deepseek",
        "megatron.bridge.recipes.kimi",
        "megatron.bridge.recipes.utils",
        "megatron.bridge.training",
        "megatron.bridge.training.utils",
        "megatron.bridge.utils",
    ):
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules.setdefault(package_name, package)

    deepseek_v3 = types.ModuleType("megatron.bridge.recipes.deepseek.deepseek_v3")
    deepseek_v3.set_deepseek_v3_pipeline_model_parallel_layout = lambda *_args, **_kwargs: None

    kimi_k2 = types.ModuleType("megatron.bridge.recipes.kimi.kimi_k2")
    kimi_k2._get_kimi_k2_pipeline_layout = lambda *_args, **_kwargs: None

    determinism_utils = types.ModuleType("megatron.bridge.recipes.utils.determinism_utils")
    determinism_utils.apply_determinism_overrides = lambda _config: None

    comm_overlap = types.ModuleType("megatron.bridge.training.comm_overlap")

    class CommOverlapConfig:
        pass

    comm_overlap.CommOverlapConfig = CommOverlapConfig

    config_module = types.ModuleType("megatron.bridge.training.config")

    class TokenizerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config_module.ConfigContainer = object
    config_module.TokenizerConfig = TokenizerConfig

    flex_dispatcher = types.ModuleType("megatron.bridge.training.flex_dispatcher_backend")

    def apply_flex_dispatcher_backend(model, backend):
        model.moe_token_dispatcher_type = "flex"
        model.moe_flex_dispatcher_backend = backend

    flex_dispatcher.apply_flex_dispatcher_backend = apply_flex_dispatcher_backend

    moe_token_drop = types.ModuleType("megatron.bridge.training.utils.moe_token_drop")
    moe_token_drop.apply_moe_token_drop = lambda *_args, **_kwargs: None

    omegaconf_utils = types.ModuleType("megatron.bridge.training.utils.omegaconf_utils")
    omegaconf_utils.apply_overrides = lambda config, *_args, **_kwargs: config
    omegaconf_utils.create_omegaconf_dict_config = lambda *_args, **_kwargs: {}
    omegaconf_utils.parse_hydra_overrides = lambda overrides: overrides

    cuda_graph = types.ModuleType("megatron.bridge.utils.cuda_graph")
    cuda_graph.cuda_graph_module_names = lambda *_args, **_kwargs: []
    cuda_graph.is_full_iteration_cuda_graph = lambda *_args, **_kwargs: False
    cuda_graph.set_cuda_graph_modules = lambda *_args, **_kwargs: None
    cuda_graph.set_full_iteration_cuda_graph = lambda *_args, **_kwargs: None
    cuda_graph.validate_cuda_graph_configuration = lambda *_args, **_kwargs: None

    sys.modules.setdefault("nemo_run", nemo_run)
    sys.modules.setdefault("nemo_run.config", nemo_run_config)
    sys.modules.setdefault("nemo_run.core", types.ModuleType("nemo_run.core"))
    sys.modules.setdefault("nemo_run.core.execution", types.ModuleType("nemo_run.core.execution"))
    sys.modules.setdefault("nemo_run.core.execution.launcher", launcher_module)
    sys.modules.setdefault("megatron.bridge.recipes.run_plugins", run_plugins)
    sys.modules.setdefault("megatron.bridge.recipes.deepseek.deepseek_v3", deepseek_v3)
    sys.modules.setdefault("megatron.bridge.recipes.kimi.kimi_k2", kimi_k2)
    sys.modules.setdefault("megatron.bridge.recipes.utils.determinism_utils", determinism_utils)
    sys.modules.setdefault("megatron.bridge.training.comm_overlap", comm_overlap)
    sys.modules.setdefault("megatron.bridge.training.config", config_module)
    sys.modules.setdefault("megatron.bridge.training.flex_dispatcher_backend", flex_dispatcher)
    sys.modules.setdefault("megatron.bridge.training.utils.moe_token_drop", moe_token_drop)
    sys.modules.setdefault("megatron.bridge.training.utils.omegaconf_utils", omegaconf_utils)
    sys.modules.setdefault("megatron.bridge.utils.cuda_graph", cuda_graph)


_install_launcher_stubs()

import run_script
import setup_experiment
from utils import utils


def test_nemo_ci_legacy_variants_select_the_default_flat_recipe_name():
    assert utils._recipe_variant_suffix(None) == ""
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
    assert "in_container_training_script_dir" in setup_source
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


def test_run_script_delegates_training_to_shared_recipe_runner(monkeypatch):
    args = SimpleNamespace(
        model_recipe_name="qwen3_vl_30b_a3b",
        task="pretrain",
        num_gpus=8,
        gpu="h100",
        compute_dtype="bf16",
        config_variant=None,
        domain="qwen3vl",
        dryrun=True,
        save_config_filepath="/tmp/config.yaml",
        dump_env=True,
    )
    recipe = SimpleNamespace()
    final_recipe = SimpleNamespace()
    forward_step = object()
    runner = SimpleNamespace(
        load_perf_recipe_by_name=Mock(return_value=recipe),
        load_forward_step=Mock(return_value=forward_step),
        run_config=Mock(),
    )
    apply_overrides = Mock(return_value=final_recipe)

    monkeypatch.setattr(run_script, "_load_recipe_runner", lambda: runner)
    monkeypatch.setattr(run_script, "_apply_perf_recipe_overrides", apply_overrides)

    run_script._run_training(args, ["model.num_layers=1"])

    runner.load_perf_recipe_by_name.assert_called_once_with(
        model_recipe_name="qwen3_vl_30b_a3b",
        task="pretrain",
        num_gpus=8,
        gpu="h100",
        precision="bf16",
        config_variant=None,
    )
    apply_overrides.assert_called_once_with(recipe, ["model.num_layers=1"], args)
    runner.load_forward_step.assert_called_once_with("qwen3_vl_step", mode="pretrain")
    runner.run_config.assert_called_once_with(
        config=final_recipe,
        mode="pretrain",
        step_func=forward_step,
        dryrun=True,
        save_config_filepath="/tmp/config.yaml",
        barrier_before_destroy=True,
        dryrun_num_gpus=8,
        dump_environment=True,
    )


def test_run_script_maps_perf_domains_to_shared_step_names():
    assert run_script._step_function_name("llm") == "llm_step"
    assert run_script._step_function_name("vlm") == "vlm_step"
    assert run_script._step_function_name("qwen3vl") == "qwen3_vl_step"
    assert run_script._step_function_name("diffusion") == "wan_step"


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
