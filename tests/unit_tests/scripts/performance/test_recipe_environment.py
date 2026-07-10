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

"""Tests for launcher-side library recipe environment resolution."""

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))


def _install_lightweight_environment_utils() -> None:
    """Load environment_utils without importing the full Bridge dependency tree."""
    module_name = "megatron.bridge.recipes.utils.environment_utils"
    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    for package_name in ("megatron", "megatron.bridge", "megatron.bridge.recipes", "megatron.bridge.recipes.utils"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
    training_package = types.ModuleType("megatron.bridge.training")
    training_package.__path__ = []
    sys.modules[training_package.__name__] = training_package
    config_module = types.ModuleType("megatron.bridge.training.config")
    config_module.ConfigContainer = object
    sys.modules[config_module.__name__] = config_module

    source = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "megatron"
        / "bridge"
        / "recipes"
        / "utils"
        / "environment_utils.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


_install_lightweight_environment_utils()

import run_recipe
import run_script
from utils import utils


def _add_environment(config, custom_env_vars):
    utils.add_library_recipe_environment_variables(custom_env_vars=custom_env_vars, config=config)


def test_library_recipe_environment_copies_explicit_values_and_preserves_launcher_values():
    config = SimpleNamespace(
        env_vars={
            "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            "TORCHINDUCTOR_WORKER_START": "fork",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        },
    )
    custom_env_vars = {"NVTE_FWD_LAYERNORM_SM_MARGIN": "48"}
    original_mapping = custom_env_vars

    _add_environment(config, custom_env_vars)

    assert custom_env_vars is original_mapping
    assert custom_env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == "48"
    assert custom_env_vars["TORCHINDUCTOR_WORKER_START"] == "fork"
    assert custom_env_vars["TORCH_NCCL_AVOID_RECORD_STREAMS"] == "1"


@pytest.mark.parametrize(
    ("gpu", "ep_size", "expected_ranks", "expected_domain", "expected_mnnvl"),
    [("h100", 32, 8, 8, 0), ("gb200", 32, 32, 72, 1), ("vr200", 64, 64, 72, 1)],
)
def test_library_target_topology_adapts_only_hybridep(gpu, ep_size, expected_ranks, expected_domain, expected_mnnvl):
    config = SimpleNamespace(
        env_vars={
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
            "NVLINK_DOMAIN_SIZE": 8,
            "USE_MNNVL": 0,
            "MODEL_SPECIFIC": 1,
        },
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep", expert_model_parallel_size=ep_size),
    )

    utils.apply_library_target_topology_environment(config, gpu=gpu)

    assert config.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == expected_ranks
    assert config.env_vars["NVLINK_DOMAIN_SIZE"] == expected_domain
    assert config.env_vars["USE_MNNVL"] == expected_mnnvl
    assert config.env_vars["MODEL_SPECIFIC"] == 1


def test_library_target_topology_removes_disabled_hybridep_values():
    config = SimpleNamespace(
        env_vars={
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
            "NVLINK_DOMAIN_SIZE": 8,
            "USE_MNNVL": 0,
        },
        model=SimpleNamespace(moe_flex_dispatcher_backend=None, expert_model_parallel_size=8),
    )

    utils.apply_library_target_topology_environment(config, gpu="h100")

    assert not config.env_vars


def test_library_target_topology_rejects_nonpositive_ep_size():
    config = SimpleNamespace(
        env_vars={},
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep", expert_model_parallel_size=0),
    )

    with pytest.raises(ValueError, match="must be positive"):
        utils.apply_library_target_topology_environment(config, gpu="h100")


def test_explicit_environment_map_protects_target_topology_values():
    base_env = {"NVLINK_DOMAIN_SIZE": 8, "USE_MNNVL": 0}
    protected = utils.explicit_environment_override_names(
        ["++env_vars={NVLINK_DOMAIN_SIZE:8,USE_MNNVL:0}"],
        base_env,
        base_env.copy(),
    )
    config = SimpleNamespace(
        env_vars=base_env.copy(),
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep", expert_model_parallel_size=32),
    )

    utils.apply_library_target_topology_environment(config, gpu="gb200", protected_env_names=protected)

    assert config.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert config.env_vars["USE_MNNVL"] == 0
    assert config.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 32


def test_workload_base_config_copies_recipe_environment():
    """The pre-exec perf bootstrap should receive an isolated copy of recipe env defaults."""
    recipe_env = {"TORCHINDUCTOR_WORKER_START": "fork", "QUANTIZATION_TYPE_DEBUG": 1}
    config = SimpleNamespace(
        env_vars=recipe_env,
        model=SimpleNamespace(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
        ),
        train=SimpleNamespace(global_batch_size=8, micro_batch_size=1),
        ddp=SimpleNamespace(),
        comm_overlap=None,
        mixed_precision=None,
    )

    workload_config = utils._workload_base_config_from_recipe(config, num_gpus=8)

    assert workload_config.env_vars == recipe_env
    assert workload_config.env_vars is not recipe_env


def test_perf_runner_environment_preserves_process_values(monkeypatch):
    recipe = SimpleNamespace(env_vars={"RECIPE_DEFAULT": 1, "LAUNCHER_VALUE": "recipe"})
    monkeypatch.delenv("RECIPE_DEFAULT", raising=False)
    monkeypatch.setenv("LAUNCHER_VALUE", "launcher")

    run_script._apply_recipe_environment(recipe)

    assert run_script.os.environ["RECIPE_DEFAULT"] == "1"
    assert run_script.os.environ["LAUNCHER_VALUE"] == "launcher"


@pytest.mark.parametrize("env_vars", [{"": "1"}, {1: "bad-name"}, {"VALID_NAME": ["not", "scalar"]}])
def test_perf_runner_environment_rejects_invalid_values(env_vars):
    with pytest.raises((TypeError, ValueError)):
        run_script._apply_recipe_environment(SimpleNamespace(env_vars=env_vars))


def test_perf_runner_applies_cli_environment_overrides_before_export(monkeypatch):
    """Hydra env overrides must be reflected in the pre-exec process environment."""
    args = SimpleNamespace(
        model_family_name=None,
        model_recipe_name="deepseek_v3",
        task="pretrain",
        num_gpus=8,
        gpu="gb200",
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
    cli_overrides = ["++env_vars={TORCHINDUCTOR_WORKER_START:spawn}"]
    base_recipe = SimpleNamespace(env_vars={"TORCHINDUCTOR_WORKER_START": "fork"})
    effective_recipe = SimpleNamespace(env_vars={"TORCHINDUCTOR_WORKER_START": "spawn"})
    calls = []

    monkeypatch.setattr(run_script, "get_perf_recipe_for_environment", lambda **kwargs: base_recipe)

    def apply_overrides(recipe, overrides, parsed_args):
        calls.append(("overrides", recipe, overrides, parsed_args))
        return effective_recipe

    monkeypatch.setattr(run_script, "_apply_perf_recipe_overrides", apply_overrides)
    monkeypatch.setattr(run_script, "_apply_recipe_environment", lambda recipe: calls.append(("environment", recipe)))
    monkeypatch.setattr(run_script.os, "execvpe", lambda *args: None)

    run_script._bootstrap_recipe_environment(args, cli_overrides)

    assert calls == [
        ("overrides", base_recipe, cli_overrides, args),
        ("environment", effective_recipe),
    ]


def test_library_runner_applies_known_ep_override_to_effective_recipe():
    """The library pre-exec config should mirror run_recipe's argparse EP update."""
    recipe = SimpleNamespace(
        model=SimpleNamespace(expert_model_parallel_size=8),
        ddp=SimpleNamespace(nccl_ub=False, fsdp_manual_registration=False),
    )
    args = SimpleNamespace(expert_model_parallel_size=32)

    effective_recipe = run_recipe._apply_library_recipe_overrides(recipe, [], args)

    assert effective_recipe.model.expert_model_parallel_size == 32


def test_library_runner_applies_env_relevant_argparse_overrides():
    """NCCL-UB and dispatcher settings must be visible before the runner execs."""
    recipe = SimpleNamespace(
        model=SimpleNamespace(
            expert_model_parallel_size=8,
            num_moe_experts=128,
            moe_token_dispatcher_type="alltoall",
            moe_flex_dispatcher_backend="deepep",
            moe_shared_expert_overlap=True,
        ),
        ddp=SimpleNamespace(
            nccl_ub=False,
            average_in_collective=True,
            use_megatron_fsdp=True,
            fsdp_manual_registration=False,
        ),
    )
    args = SimpleNamespace(
        expert_model_parallel_size=None,
        nccl_ub=True,
        moe_flex_dispatcher_backend="hybridep",
    )

    effective_recipe = run_recipe._apply_library_recipe_overrides(recipe, [], args)

    assert effective_recipe.ddp.nccl_ub is True
    assert effective_recipe.ddp.average_in_collective is False
    assert effective_recipe.ddp.fsdp_manual_registration is True
    assert effective_recipe.model.moe_token_dispatcher_type == "flex"
    assert effective_recipe.model.moe_flex_dispatcher_backend == "hybridep"
    assert effective_recipe.model.moe_shared_expert_overlap is False


def test_library_runner_disables_flex_dispatcher_consistently():
    """The explicit None backend must clear both flex dispatcher fields."""
    recipe = SimpleNamespace(
        model=SimpleNamespace(
            expert_model_parallel_size=8,
            num_moe_experts=128,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_shared_expert_overlap=False,
        ),
        ddp=SimpleNamespace(nccl_ub=False),
    )
    args = SimpleNamespace(
        expert_model_parallel_size=None,
        nccl_ub=False,
        moe_flex_dispatcher_backend=None,
    )

    effective_recipe = run_recipe._apply_library_recipe_overrides(recipe, [], args)

    assert effective_recipe.model.moe_token_dispatcher_type == "alltoall"
    assert effective_recipe.model.moe_flex_dispatcher_backend is None


@pytest.mark.parametrize("dispatcher_override", [-1, "hybridep"])
def test_library_runner_preserves_dense_dispatcher_fields(dispatcher_override):
    """Finalization must not rewrite dense-model dispatcher defaults."""
    recipe = SimpleNamespace(
        model=SimpleNamespace(
            expert_model_parallel_size=1,
            num_moe_experts=None,
            moe_token_dispatcher_type=None,
            moe_flex_dispatcher_backend=None,
        ),
        ddp=SimpleNamespace(nccl_ub=False, fsdp_manual_registration=False),
    )
    args = SimpleNamespace(
        expert_model_parallel_size=None,
        nccl_ub=False,
        moe_flex_dispatcher_backend=dispatcher_override,
    )

    effective_recipe = run_recipe._apply_library_recipe_overrides(recipe, [], args)

    assert effective_recipe.model.moe_token_dispatcher_type is None
    assert effective_recipe.model.moe_flex_dispatcher_backend is None


def test_library_runner_uses_shared_environment_override_helpers():
    """Both self-exec passes must use the shared argparse and topology logic."""
    repo_root = Path(__file__).resolve().parents[4]
    expected_calls = {
        repo_root / "scripts" / "performance" / "run_recipe.py": {
            "_apply_library_recipe_overrides": {
                "apply_library_argparse_overrides",
                "finalize_library_config_overrides",
            },
            "set_user_overrides": {"apply_library_argparse_overrides"},
            "_apply_target_environment": {
                "explicit_environment_override_names",
                "apply_library_target_topology_environment",
            },
        },
    }

    for path, functions in expected_calls.items():
        tree = ast.parse(path.read_text())
        for function_name, expected_helpers in functions.items():
            function = next(
                node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            called_helpers = {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            assert called_helpers >= expected_helpers


def test_library_hydra_ep_override_wins_after_known_ep_override(monkeypatch):
    """Hydra EP must win after argparse EP, matching the final library runner."""
    recipe = SimpleNamespace(
        model=SimpleNamespace(expert_model_parallel_size=8),
        ddp=SimpleNamespace(nccl_ub=False, fsdp_manual_registration=False),
    )
    args = SimpleNamespace(expert_model_parallel_size=32)

    def apply_hydra(config, overrides):
        assert config.model.expert_model_parallel_size == 32
        assert overrides == ["model.expert_model_parallel_size=64"]
        config.model.expert_model_parallel_size = 64
        return config

    monkeypatch.setattr(run_recipe, "_process_library_hydra_overrides", apply_hydra)

    effective_recipe = run_recipe._apply_library_recipe_overrides(
        recipe,
        ["model.expert_model_parallel_size=64"],
        args,
    )

    assert effective_recipe.model.expert_model_parallel_size == 64


def test_library_hydra_disable_clears_argparse_dependent_state(monkeypatch):
    """Hydra primary-field precedence must not leave stale coupled settings."""
    recipe = SimpleNamespace(
        model=SimpleNamespace(
            expert_model_parallel_size=8,
            num_moe_experts=128,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_shared_expert_overlap=False,
        ),
        ddp=SimpleNamespace(
            nccl_ub=False,
            average_in_collective=True,
            use_megatron_fsdp=True,
            fsdp_manual_registration=False,
        ),
    )
    args = SimpleNamespace(
        expert_model_parallel_size=None,
        nccl_ub=True,
        moe_flex_dispatcher_backend="hybridep",
    )

    def apply_hydra(config, _overrides):
        config.ddp.nccl_ub = False
        config.model.moe_flex_dispatcher_backend = None
        return config

    monkeypatch.setattr(run_recipe, "_process_library_hydra_overrides", apply_hydra)

    effective_recipe = run_recipe._apply_library_recipe_overrides(
        recipe,
        ["ddp.nccl_ub=false", "model.moe_flex_dispatcher_backend=null"],
        args,
    )

    assert effective_recipe.ddp.nccl_ub is False
    assert effective_recipe.ddp.fsdp_manual_registration is False
    assert effective_recipe.model.moe_token_dispatcher_type == "alltoall"
    assert effective_recipe.model.moe_flex_dispatcher_backend is None


def test_library_runner_exports_resolved_recipe_environment_before_exec(monkeypatch):
    """The original runner exports its explicit env map before self-exec."""
    args = SimpleNamespace(
        use_recipes=True,
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3_32nodes",
        task="pretrain",
        wandb_experiment_name="test-experiment",
        gpu="gb200",
        expert_model_parallel_size=32,
    )
    base_recipe = SimpleNamespace(
        env_vars={"TORCHINDUCTOR_WORKER_START": "fork"},
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep", expert_model_parallel_size=8),
    )
    effective_recipe = SimpleNamespace(
        env_vars={"TORCHINDUCTOR_WORKER_START": "fork"},
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep", expert_model_parallel_size=64),
    )
    calls = []
    cli_overrides = ["model.expert_model_parallel_size=64"]
    monkeypatch.setattr(run_recipe, "_get_library_recipe", lambda parsed_args: base_recipe)

    def apply_overrides(recipe, overrides, parsed_args):
        calls.append(("overrides", recipe, overrides, parsed_args))
        return effective_recipe

    def apply_target(recipe, base_env_vars, overrides, parsed_args):
        calls.append(("target", recipe, base_env_vars, overrides, parsed_args))

    def apply_environment(recipe):
        calls.append(("environment", recipe))

    def exec_runner(executable, argv, env):
        calls.append(("exec", executable, argv, env))

    monkeypatch.setattr(run_recipe, "_apply_library_recipe_overrides", apply_overrides)
    monkeypatch.setattr(run_recipe, "_apply_target_environment", apply_target)
    monkeypatch.setattr(run_recipe, "_apply_recipe_environment", apply_environment)
    monkeypatch.setattr(run_recipe.os, "execvpe", exec_runner)

    run_recipe._bootstrap_recipe_environment(args, cli_overrides)

    assert calls[0] == ("overrides", base_recipe, cli_overrides, args)
    assert calls[1] == (
        "target",
        effective_recipe,
        {"TORCHINDUCTOR_WORKER_START": "fork"},
        cli_overrides,
        args,
    )
    assert calls[2] == ("environment", effective_recipe)
    assert calls[3][0] == "exec"
    assert calls[3][2][1].endswith("run_recipe.py")
    assert calls[3][3][run_recipe.ENV_BOOTSTRAP_MARKER] == str(run_recipe.os.getpid())


def test_library_recipe_environment_rejects_non_scalar_values():
    config = SimpleNamespace(env_vars={"INVALID": ["value"]})

    with pytest.raises(TypeError, match="must have a scalar value"):
        _add_environment(config, {})
