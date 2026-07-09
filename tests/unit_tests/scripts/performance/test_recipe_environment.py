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

import run_script_with_env
from utils import utils


def _add_environment(config, custom_env_vars, **overrides):
    args = {
        "custom_env_vars": custom_env_vars,
        "config": config,
        "gpu": "h100",
    }
    args.update(overrides)
    utils.add_library_recipe_environment_variables(**args)


def test_library_recipe_environment_mutates_existing_mapping_and_preserves_explicit_values():
    config = SimpleNamespace(
        env_vars={
            "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            "TORCHINDUCTOR_WORKER_START": "fork",
        },
        model=SimpleNamespace(moe_flex_dispatcher_backend=None),
        ddp=SimpleNamespace(nccl_ub=False),
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
    [("h100", 4, "4", "8", "0"), ("gb200", 32, "32", "72", "1")],
)
def test_library_hybridep_topology_environment_tracks_final_ep_size(
    gpu, ep_size, expected_ranks, expected_domain, expected_mnnvl
):
    config = SimpleNamespace(
        env_vars={
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
            "USE_MNNVL": 1,
        },
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep", expert_model_parallel_size=ep_size),
        ddp=SimpleNamespace(nccl_ub=False),
    )
    custom_env_vars = {}

    _add_environment(config, custom_env_vars, gpu=gpu)

    assert custom_env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == expected_ranks
    assert custom_env_vars["NVLINK_DOMAIN_SIZE"] == expected_domain
    assert custom_env_vars["USE_MNNVL"] == expected_mnnvl


@pytest.mark.parametrize(
    ("gpu", "expected_ranks", "expected_domain", "expected_mnnvl"),
    [("h100", "8", "8", "0"), ("gb200", "32", "72", "1")],
)
def test_library_hybridep_topology_uses_recipe_ep_without_override(
    gpu, expected_ranks, expected_domain, expected_mnnvl
):
    """The GPU topology must be normalized even when the launcher has no EP override."""
    config = SimpleNamespace(
        env_vars={},
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep", expert_model_parallel_size=32),
        ddp=SimpleNamespace(nccl_ub=False),
    )
    custom_env_vars = {}

    _add_environment(config, custom_env_vars, gpu=gpu)

    assert custom_env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == expected_ranks
    assert custom_env_vars["NVLINK_DOMAIN_SIZE"] == expected_domain
    assert custom_env_vars["USE_MNNVL"] == expected_mnnvl
    assert custom_env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_library_hybridep_topology_preserves_explicit_values():
    """Explicit launcher values retain precedence over normalized recipe defaults."""
    config = SimpleNamespace(
        env_vars={},
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep", expert_model_parallel_size=32),
        ddp=SimpleNamespace(nccl_ub=False),
    )
    custom_env_vars = {
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": "custom-ranks",
        "NVLINK_DOMAIN_SIZE": "custom-domain",
        "USE_MNNVL": "custom-mnnvl",
    }

    _add_environment(config, custom_env_vars, gpu="gb200")

    assert custom_env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == "custom-ranks"
    assert custom_env_vars["NVLINK_DOMAIN_SIZE"] == "custom-domain"
    assert custom_env_vars["USE_MNNVL"] == "custom-mnnvl"


def test_library_recipe_owns_legacy_process_defaults_and_nccl_ub_overrides():
    config = SimpleNamespace(
        env_vars={},
        model=SimpleNamespace(moe_flex_dispatcher_backend=None),
        ddp=SimpleNamespace(nccl_ub=True),
    )
    custom_env_vars = {}

    _add_environment(config, custom_env_vars, gpu="h100")

    assert config.env_vars["NCCL_NVLS_ENABLE"] == 1
    assert config.env_vars["NCCL_CTA_POLICY"] == 1
    assert config.env_vars["NVTE_NORM_FWD_USE_CUDNN"] == 1
    assert config.env_vars["NVTE_NORM_BWD_USE_CUDNN"] == 1
    assert config.env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert custom_env_vars["NCCL_NVLS_ENABLE"] == "1"


def test_workload_base_config_copies_recipe_environment():
    """The pre-exec perf wrapper should receive an isolated copy of recipe env defaults."""
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


def test_explicit_environment_map_protects_values_equal_to_recipe_defaults():
    """Explicit same-valued topology settings must still beat derived topology."""
    base_env = {"NVLINK_DOMAIN_SIZE": 8, "USE_MNNVL": 0}

    protected = utils.explicit_environment_override_names(
        ["++env_vars={NVLINK_DOMAIN_SIZE:8,USE_MNNVL:0}"],
        base_env,
        base_env.copy(),
    )

    assert protected == {"NVLINK_DOMAIN_SIZE", "USE_MNNVL"}


def test_perf_wrapper_applies_cli_overrides_before_deriving_environment(monkeypatch):
    """Hydra env overrides must be reflected in the pre-exec workload config."""
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
    workload_config = SimpleNamespace()
    calls = []

    parser = SimpleNamespace(parse_known_args=lambda: (args, cli_overrides))
    monkeypatch.setattr(run_script_with_env, "parse_cli_args", lambda: parser)
    monkeypatch.setattr(run_script_with_env, "get_perf_recipe_by_name", lambda **kwargs: base_recipe)

    def apply_overrides(recipe, overrides, parsed_args):
        calls.append(("overrides", recipe, overrides, parsed_args))
        return effective_recipe

    def build_workload(recipe, *, num_gpus):
        calls.append(("workload", recipe, num_gpus))
        return workload_config

    def finalize_environment(recipe, parsed_args, protected_env_names):
        calls.append(("finalize", recipe, parsed_args, protected_env_names))
        return recipe

    class FakePerfEnvPlugin:
        def __init__(self, **kwargs):
            pass

        def setup_recipe_environment(self, task, executor, config, protected_recipe_env_names=None):
            calls.append(("environment", config, protected_recipe_env_names))

    monkeypatch.setattr(run_script_with_env, "_apply_perf_recipe_overrides", apply_overrides)
    monkeypatch.setattr(run_script_with_env, "_finalize_perf_recipe_environment", finalize_environment)
    monkeypatch.setattr(run_script_with_env, "_workload_base_config_from_recipe", build_workload)
    monkeypatch.setattr(run_script_with_env, "PerfEnvPlugin", FakePerfEnvPlugin)
    monkeypatch.setattr(run_script_with_env.os, "execvpe", lambda *args: None)

    run_script_with_env.main()

    assert calls[:3] == [
        ("overrides", base_recipe, cli_overrides, args),
        ("finalize", effective_recipe, args, {"TORCHINDUCTOR_WORKER_START"}),
        ("workload", effective_recipe, 8),
    ]
    assert calls[3] == ("environment", workload_config, {"TORCHINDUCTOR_WORKER_START"})


def test_library_wrapper_applies_known_ep_override_to_effective_recipe():
    """The library pre-exec config should mirror run_recipe's argparse EP update."""
    recipe = SimpleNamespace(
        model=SimpleNamespace(expert_model_parallel_size=8),
        ddp=SimpleNamespace(nccl_ub=False, fsdp_manual_registration=False),
    )
    args = SimpleNamespace(expert_model_parallel_size=32)

    effective_recipe = run_script_with_env._apply_library_recipe_overrides(recipe, [], args)

    assert effective_recipe.model.expert_model_parallel_size == 32


def test_library_wrapper_applies_env_relevant_argparse_overrides():
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

    effective_recipe = run_script_with_env._apply_library_recipe_overrides(recipe, [], args)

    assert effective_recipe.ddp.nccl_ub is True
    assert effective_recipe.ddp.average_in_collective is False
    assert effective_recipe.ddp.fsdp_manual_registration is True
    assert effective_recipe.model.moe_token_dispatcher_type == "flex"
    assert effective_recipe.model.moe_flex_dispatcher_backend == "hybridep"
    assert effective_recipe.model.moe_shared_expert_overlap is False


def test_library_wrapper_disables_flex_dispatcher_consistently():
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

    effective_recipe = run_script_with_env._apply_library_recipe_overrides(recipe, [], args)

    assert effective_recipe.model.moe_token_dispatcher_type == "alltoall"
    assert effective_recipe.model.moe_flex_dispatcher_backend is None


@pytest.mark.parametrize("dispatcher_override", [-1, "hybridep"])
def test_library_wrapper_preserves_dense_dispatcher_fields(dispatcher_override):
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

    effective_recipe = run_script_with_env._apply_library_recipe_overrides(recipe, [], args)

    assert effective_recipe.model.moe_token_dispatcher_type is None
    assert effective_recipe.model.moe_flex_dispatcher_backend is None


def test_library_wrapper_and_final_runner_share_environment_override_helper():
    """Pre-exec and final-runner config must use the same argparse logic."""
    repo_root = Path(__file__).resolve().parents[4]
    expected_calls = {
        repo_root / "scripts" / "performance" / "run_script_with_env.py": {
            "_apply_library_recipe_overrides": {
                "apply_library_environment_overrides",
                "finalize_library_environment_overrides",
            }
        },
        repo_root / "scripts" / "performance" / "run_recipe.py": {
            "set_user_overrides": {"apply_library_environment_overrides"},
            "main": {"finalize_library_environment_overrides"},
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

    monkeypatch.setattr(run_script_with_env, "_process_library_hydra_overrides", apply_hydra)

    effective_recipe = run_script_with_env._apply_library_recipe_overrides(
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

    monkeypatch.setattr(run_script_with_env, "_process_library_hydra_overrides", apply_hydra)

    effective_recipe = run_script_with_env._apply_library_recipe_overrides(
        recipe,
        ["ddp.nccl_ub=false", "model.moe_flex_dispatcher_backend=null"],
        args,
    )

    assert effective_recipe.ddp.nccl_ub is False
    assert effective_recipe.ddp.fsdp_manual_registration is False
    assert effective_recipe.model.moe_token_dispatcher_type == "alltoall"
    assert effective_recipe.model.moe_flex_dispatcher_backend is None


def test_library_wrapper_resolves_environment_inside_worker_before_exec(monkeypatch):
    """Library env derives from the final user-then-Hydra effective recipe."""
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
    parser = SimpleNamespace(parse_known_args=lambda: (args, cli_overrides))

    monkeypatch.setattr(run_script_with_env, "parse_cli_args", lambda: parser)
    monkeypatch.setattr(run_script_with_env, "get_library_recipe", lambda **kwargs: base_recipe)

    def apply_overrides(recipe, overrides, parsed_args):
        calls.append(("overrides", recipe, overrides, parsed_args))
        return effective_recipe

    def add_environment(**kwargs):
        calls.append(("environment", kwargs))

    def exec_runner(executable, argv, env):
        calls.append(("exec", executable, argv, env))

    monkeypatch.setattr(run_script_with_env, "_apply_library_recipe_overrides", apply_overrides)
    monkeypatch.setattr(run_script_with_env, "add_library_recipe_environment_variables", add_environment)
    monkeypatch.setattr(run_script_with_env.os, "execvpe", exec_runner)

    run_script_with_env.main()

    assert calls[0] == ("overrides", base_recipe, cli_overrides, args)
    assert calls[1][0] == "environment"
    assert calls[1][1]["config"] is effective_recipe
    assert calls[1][1]["gpu"] == "gb200"
    assert "expert_model_parallel_size" not in calls[1][1]
    assert calls[2][0] == "exec"
    assert calls[2][2][1].endswith("run_recipe.py")


def test_library_recipe_environment_rejects_non_scalar_values():
    config = SimpleNamespace(
        env_vars={"INVALID": ["value"]},
        model=SimpleNamespace(moe_flex_dispatcher_backend=None),
        ddp=SimpleNamespace(nccl_ub=False),
    )

    with pytest.raises(TypeError, match="must have a scalar value"):
        _add_environment(config, {})
