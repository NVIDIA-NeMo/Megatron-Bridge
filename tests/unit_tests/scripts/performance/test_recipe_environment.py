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

"""Tests for launcher-side recipe environment resolution."""

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

import bootstrap
import run_recipe
import run_script
from utils import utils


@pytest.mark.parametrize(
    ("gpu", "ep_size", "expected_ranks", "expected_domain", "expected_mnnvl"),
    [("h100", 32, 8, 8, 0), ("gb200", 32, 32, 72, 1), ("vr200", 64, 64, 72, 1)],
)
def test_target_topology_adapts_only_hybridep(gpu, ep_size, expected_ranks, expected_domain, expected_mnnvl):
    config = SimpleNamespace(
        env_vars={
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
            "NVLINK_DOMAIN_SIZE": 8,
            "USE_MNNVL": 0,
            "MODEL_SPECIFIC": 1,
        },
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep", expert_model_parallel_size=ep_size),
    )

    utils.apply_target_topology_environment(config, gpu=gpu)

    assert config.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == expected_ranks
    assert config.env_vars["NVLINK_DOMAIN_SIZE"] == expected_domain
    assert config.env_vars["USE_MNNVL"] == expected_mnnvl
    assert config.env_vars["MODEL_SPECIFIC"] == 1


def test_target_topology_removes_disabled_hybridep_values():
    config = SimpleNamespace(
        env_vars={
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
            "NVLINK_DOMAIN_SIZE": 8,
            "USE_MNNVL": 0,
        },
        model=SimpleNamespace(moe_flex_dispatcher_backend=None, expert_model_parallel_size=8),
    )

    utils.apply_target_topology_environment(config, gpu="h100")

    assert not config.env_vars


def test_target_topology_rejects_nonpositive_ep_size():
    config = SimpleNamespace(
        env_vars={},
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep", expert_model_parallel_size=0),
    )

    with pytest.raises(ValueError, match="must be positive"):
        utils.apply_target_topology_environment(config, gpu="h100")


def test_nccl_ub_environment_matches_legacy_launcher_behavior():
    config = SimpleNamespace(
        env_vars={
            "NCCL_GRAPH_REGISTER": 0,
            "NCCL_NVLS_ENABLE": 0,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        ddp=SimpleNamespace(nccl_ub=True),
    )

    utils.apply_feature_environment(config, nccl_ub_override=True)

    assert config.env_vars["NCCL_NVLS_ENABLE"] == 1
    assert config.env_vars["NCCL_CTA_POLICY"] == 1
    assert config.env_vars["NCCL_GRAPH_REGISTER"] == 0
    assert config.env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_recipe_nccl_ub_does_not_rewrite_inline_env_without_cli_override():
    config = SimpleNamespace(
        env_vars={"NCCL_NVLS_ENABLE": 0, "RECIPE_SPECIFIC": 7},
        ddp=SimpleNamespace(nccl_ub=True),
    )
    original_env = config.env_vars.copy()

    utils.apply_feature_environment(config, nccl_ub_override=None)

    assert config.env_vars == original_env


def test_nccl_ub_respects_explicit_environment_overrides():
    config = SimpleNamespace(
        env_vars={
            "NCCL_GRAPH_REGISTER": 7,
            "NCCL_NVLS_ENABLE": 7,
            "PYTORCH_CUDA_ALLOC_CONF": "explicit",
        },
        ddp=SimpleNamespace(nccl_ub=True),
    )

    utils.apply_feature_environment(
        config,
        nccl_ub_override=True,
        protected_env_names={"NCCL_GRAPH_REGISTER", "NCCL_NVLS_ENABLE", "PYTORCH_CUDA_ALLOC_CONF"},
    )

    assert config.env_vars["NCCL_GRAPH_REGISTER"] == 7
    assert config.env_vars["NCCL_NVLS_ENABLE"] == 7
    assert config.env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "explicit"
    assert config.env_vars["NCCL_CTA_POLICY"] == 1


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

    utils.apply_target_topology_environment(config, gpu="gb200", protected_env_names=protected)

    assert config.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert config.env_vars["USE_MNNVL"] == 0
    assert config.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 32


def test_workload_base_config_copies_recipe_environment():
    """The pre-exec perf bootstrap should receive an isolated copy of recipe env defaults."""
    recipe_env = {"CUDA_DEVICE_MAX_CONNECTIONS": 32, "NCCL_NVLS_ENABLE": 0}
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

    bootstrap._apply_recipe_environment(recipe)

    assert run_script.os.environ["RECIPE_DEFAULT"] == "1"
    assert run_script.os.environ["LAUNCHER_VALUE"] == "launcher"


@pytest.mark.parametrize("env_vars", [{"": "1"}, {1: "bad-name"}, {"VALID_NAME": ["not", "scalar"]}])
def test_perf_runner_environment_rejects_invalid_values(env_vars):
    with pytest.raises((TypeError, ValueError)):
        bootstrap._apply_recipe_environment(SimpleNamespace(env_vars=env_vars))


def test_flat_environment_preparation_applies_cli_overrides(monkeypatch):
    """Hydra env overrides must be reflected in the bootstrap recipe."""
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
    cli_overrides = ["++env_vars={CUDA_DEVICE_MAX_CONNECTIONS:1}"]
    base_recipe = SimpleNamespace(env_vars={"CUDA_DEVICE_MAX_CONNECTIONS": 32})
    effective_recipe = SimpleNamespace(env_vars={"CUDA_DEVICE_MAX_CONNECTIONS": 1})
    calls = []

    monkeypatch.setattr(run_script, "get_perf_recipe_by_name", lambda **kwargs: base_recipe)

    def apply_overrides(recipe, overrides, parsed_args):
        calls.append(("overrides", recipe, overrides, parsed_args))
        return effective_recipe

    monkeypatch.setattr(run_script, "_apply_perf_recipe_overrides", apply_overrides)

    result = run_script._prepare_perf_recipe(args, cli_overrides)

    assert result is effective_recipe
    assert calls == [("overrides", base_recipe, cli_overrides, args)]


def test_flat_deterministic_environment_reaches_training_exec(monkeypatch):
    """The flat training interpreter must start with deterministic process values."""
    from megatron.bridge.recipes.utils.determinism_utils import apply_determinism_overrides

    deterministic_env = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    }
    for name in deterministic_env:
        monkeypatch.delenv(name, raising=False)

    recipe = SimpleNamespace(
        env_vars={},
        model=SimpleNamespace(deterministic_mode=False, cross_entropy_loss_fusion=True),
        comm_overlap=None,
    )
    args = SimpleNamespace(
        model_recipe_name="test_model",
        task="pretrain",
        num_gpus=1,
        gpu="h100",
        compute_dtype="bf16",
        config_variant=None,
        use_recipes=False,
    )
    parser = SimpleNamespace(parse_known_args=lambda: (args, []))
    exec_environments = []

    monkeypatch.setattr(bootstrap, "parse_cli_args", lambda: parser)
    monkeypatch.setattr(run_script, "get_perf_recipe_by_name", lambda **_kwargs: recipe)

    def apply_overrides(config, _overrides, _args):
        apply_determinism_overrides(config)
        return config

    monkeypatch.setattr(run_script, "_apply_perf_recipe_overrides", apply_overrides)
    monkeypatch.setattr(bootstrap.os, "execvpe", lambda _executable, _argv, env: exec_environments.append(env))

    bootstrap.main()

    assert len(exec_environments) == 1
    exec_environment = exec_environments[0]
    assert {name: exec_environment[name] for name in deterministic_env} == deterministic_env


def test_flat_deterministic_preserves_explicit_hydra_environment(monkeypatch):
    """Hydra env additions, changes, and removals remain final with deterministic argparse."""
    from utils import overrides as override_utils

    from megatron.bridge.recipes.utils.determinism_utils import apply_determinism_overrides

    recipe = SimpleNamespace(
        env_vars={"NCCL_ALGO": "Recipe", "REMOVE_ME": "recipe"},
        model=SimpleNamespace(
            deterministic_mode=False,
            cross_entropy_loss_fusion=True,
            moe_flex_dispatcher_backend=None,
        ),
        comm_overlap=SimpleNamespace(tp_comm_overlap=True),
    )
    cli_overrides = [
        "env_vars.NCCL_ALGO=Tree",
        "+env_vars.USER_SELECTED=custom",
        "~env_vars.REMOVE_ME",
    ]

    def apply_hydra(config, overrides):
        assert overrides == cli_overrides
        config.env_vars["NCCL_ALGO"] = "Tree"
        config.env_vars["USER_SELECTED"] = "custom"
        config.env_vars.pop("REMOVE_ME")
        return config

    def apply_argparse(config, args):
        assert args.deterministic is True
        apply_determinism_overrides(config)
        return config

    monkeypatch.setattr(override_utils, "set_cli_overrides", apply_hydra)
    monkeypatch.setattr(override_utils, "set_user_overrides", apply_argparse)
    monkeypatch.setattr(
        override_utils,
        "_apply_flat_cli_environment_compatibility",
        lambda config, _args, **_kwargs: config,
    )

    effective_recipe = run_script._apply_perf_recipe_overrides(
        recipe,
        cli_overrides,
        SimpleNamespace(deterministic=True),
    )

    assert effective_recipe.env_vars["NCCL_ALGO"] == "Tree"
    assert effective_recipe.env_vars["USER_SELECTED"] == "custom"
    assert "REMOVE_ME" not in effective_recipe.env_vars
    assert effective_recipe.env_vars["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert effective_recipe.env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == 0


def test_flat_environment_compatibility_preserves_explicit_hydra_values():
    from utils.overrides import _apply_flat_cli_environment_compatibility

    base_env = {
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    effective_env = {**base_env, "CUDA_DEVICE_MAX_CONNECTIONS": 7, "NVLINK_DOMAIN_SIZE": 7}
    cli_overrides = ["++env_vars={CUDA_DEVICE_MAX_CONNECTIONS:7,NVLINK_DOMAIN_SIZE:7}"]
    protected = utils.explicit_environment_override_names(cli_overrides, base_env, effective_env)
    recipe = SimpleNamespace(
        env_vars=effective_env,
        model=SimpleNamespace(
            moe_flex_dispatcher_backend="hybridep",
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=32,
            cuda_graph_impl="none",
            cuda_graph_scope=[],
            use_transformer_engine_op_fuser=False,
            fine_grained_activation_offloading=False,
        ),
        comm_overlap=None,
        ddp=SimpleNamespace(nccl_ub=False, use_megatron_fsdp=False),
    )
    args = SimpleNamespace(
        gpu="gb200",
        moe_a2a_overlap=None,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=None,
        context_parallel_size=None,
        expert_model_parallel_size=32,
        nccl_ub=None,
        model_family_name="qwen",
        model_recipe_name="qwen3_30b_a3b",
        task="pretrain",
    )

    _apply_flat_cli_environment_compatibility(
        recipe,
        args,
        base_dispatcher_backend="hybridep",
        base_moe_a2a_overlap=False,
        protected_env_names=protected,
    )

    assert recipe.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 7
    assert recipe.env_vars["NVLINK_DOMAIN_SIZE"] == 7
    assert recipe.env_vars["USE_MNNVL"] == 0
    assert recipe.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 32


def test_flat_default_args_leave_inline_recipe_environment_unchanged():
    from utils.overrides import _apply_flat_cli_environment_compatibility

    recipe = SimpleNamespace(
        env_vars={"FUTURE_RECIPE_SPECIFIC_SETTING": 7},
        model=SimpleNamespace(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=1,
            expert_model_parallel_size=16,
        ),
    )
    args = SimpleNamespace(
        gpu="h100",
        moe_a2a_overlap=None,
        tensor_model_parallel_size=None,
        pipeline_model_parallel_size=None,
        context_parallel_size=None,
        expert_model_parallel_size=None,
        nccl_ub=None,
        model_family_name="future",
        model_recipe_name="future_model",
        task="pretrain",
    )
    original_env = recipe.env_vars.copy()

    _apply_flat_cli_environment_compatibility(
        recipe,
        args,
        base_dispatcher_backend="hybridep",
        base_moe_a2a_overlap=False,
    )

    assert recipe.env_vars == original_env


def test_flat_explicit_argparse_compatibility_changes_only_legacy_coupled_values():
    from utils.overrides import _apply_flat_cli_environment_compatibility

    recipe = SimpleNamespace(
        env_vars={
            "CUDA_DEVICE_MAX_CONNECTIONS": 1,
            "NCCL_GRAPH_REGISTER": 0,
            "NCCL_NVLS_ENABLE": 0,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "RECIPE_SPECIFIC": 7,
        },
        model=SimpleNamespace(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=1,
            expert_model_parallel_size=32,
        ),
    )
    args = SimpleNamespace(
        gpu="gb200",
        moe_a2a_overlap=True,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=None,
        expert_model_parallel_size=32,
        nccl_ub=True,
        model_family_name="llama",
        model_recipe_name="llama3_70b",
        task="pretrain",
    )

    _apply_flat_cli_environment_compatibility(
        recipe,
        args,
        base_dispatcher_backend="hybridep",
        base_moe_a2a_overlap=False,
    )

    assert recipe.env_vars == {
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "NCCL_CTA_POLICY": 1,
        "NCCL_NVLS_ENABLE": 1,
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "RECIPE_SPECIFIC": 7,
    }


def test_flat_explicit_false_a2a_preserves_legacy_config_and_env_behavior():
    from utils.overrides import _apply_flat_cli_environment_compatibility, _set_moe_a2a_overlap_overrides

    recipe = SimpleNamespace(
        env_vars={"CUDA_DEVICE_MAX_CONNECTIONS": 32},
        model=SimpleNamespace(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=8,
        ),
        comm_overlap=SimpleNamespace(
            overlap_moe_expert_parallel_comm=True,
            delay_wgrad_compute=True,
        ),
    )
    args = SimpleNamespace(
        gpu="h100",
        moe_a2a_overlap=False,
        tensor_model_parallel_size=None,
        pipeline_model_parallel_size=None,
        context_parallel_size=None,
        expert_model_parallel_size=None,
        nccl_ub=None,
        model_family_name="qwen",
        model_recipe_name="qwen3_vl_235b_a22b",
        task="pretrain",
    )

    _set_moe_a2a_overlap_overrides(recipe, moe_a2a_overlap=args.moe_a2a_overlap)
    _apply_flat_cli_environment_compatibility(
        recipe,
        args,
        base_dispatcher_backend="hybridep",
        base_moe_a2a_overlap=True,
    )

    assert recipe.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert recipe.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 1


def test_runner_applies_parallelism_overrides_to_effective_recipe():
    """The pre-exec config should receive every argparse parallelism override."""
    recipe = SimpleNamespace(
        model=SimpleNamespace(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            virtual_pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            expert_tensor_parallel_size=1,
        ),
        ddp=SimpleNamespace(nccl_ub=False, fsdp_manual_registration=False),
    )
    args = SimpleNamespace(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        context_parallel_size=8,
        virtual_pipeline_model_parallel_size=None,
        expert_model_parallel_size=32,
        expert_tensor_parallel_size=16,
    )

    effective_recipe = run_recipe._apply_recipe_overrides(recipe, args, [], environment_only=True)

    assert effective_recipe.model.tensor_model_parallel_size == 2
    assert effective_recipe.model.pipeline_model_parallel_size == 4
    assert effective_recipe.model.context_parallel_size == 8
    assert effective_recipe.model.virtual_pipeline_model_parallel_size is None
    assert effective_recipe.model.expert_model_parallel_size == 32
    assert effective_recipe.model.expert_tensor_parallel_size == 16


def test_runner_preserves_recipe_parallelism_when_argparse_values_are_omitted():
    recipe = SimpleNamespace(
        model=SimpleNamespace(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=4,
            context_parallel_size=8,
            virtual_pipeline_model_parallel_size=None,
            expert_model_parallel_size=16,
            expert_tensor_parallel_size=1,
        ),
        ddp=SimpleNamespace(nccl_ub=False),
    )
    original_parallelism = vars(recipe.model).copy()
    args = SimpleNamespace(virtual_pipeline_model_parallel_size=-1)

    effective_recipe = run_recipe._apply_recipe_overrides(recipe, args, [], environment_only=True)

    assert vars(effective_recipe.model) == original_parallelism


def test_runner_applies_env_relevant_argparse_overrides():
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

    effective_recipe = run_recipe._apply_recipe_overrides(recipe, args, [], environment_only=True)
    effective_recipe = utils.finalize_config_overrides(effective_recipe)

    assert effective_recipe.ddp.nccl_ub is True
    assert effective_recipe.ddp.average_in_collective is False
    assert effective_recipe.ddp.fsdp_manual_registration is True
    assert effective_recipe.model.moe_token_dispatcher_type == "flex"
    assert effective_recipe.model.moe_flex_dispatcher_backend == "hybridep"
    assert effective_recipe.model.moe_shared_expert_overlap is False


def test_runner_applies_determinism_before_environment_export():
    """Deterministic env and config must be resolved in the pre-exec pass."""
    recipe = SimpleNamespace(
        env_vars={"RECIPE_ENV": 1},
        model=SimpleNamespace(
            deterministic_mode=False,
            cross_entropy_loss_fusion=True,
            moe_flex_dispatcher_backend=None,
            moe_token_dispatcher_type=None,
            num_moe_experts=None,
        ),
        comm_overlap=SimpleNamespace(tp_comm_overlap=True),
        ddp=SimpleNamespace(nccl_ub=False, fsdp_manual_registration=False),
    )
    args = SimpleNamespace(
        deterministic=True,
        expert_model_parallel_size=None,
        nccl_ub=False,
        moe_flex_dispatcher_backend=-1,
    )

    effective_recipe = run_recipe._apply_recipe_overrides(recipe, args, [], environment_only=True)

    assert effective_recipe.model.deterministic_mode is True
    assert effective_recipe.model.cross_entropy_loss_fusion is False
    assert effective_recipe.comm_overlap.tp_comm_overlap is False
    assert effective_recipe.env_vars == {
        "RECIPE_ENV": 1,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }


def test_deterministic_environment_reaches_training_exec(monkeypatch):
    """The training interpreter must start with deterministic process values."""
    deterministic_env = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    }
    for name in deterministic_env:
        monkeypatch.delenv(name, raising=False)

    recipe = SimpleNamespace(
        env_vars={},
        model=SimpleNamespace(
            deterministic_mode=False,
            cross_entropy_loss_fusion=True,
            moe_flex_dispatcher_backend=None,
            moe_token_dispatcher_type=None,
            num_moe_experts=None,
        ),
        comm_overlap=None,
        ddp=SimpleNamespace(nccl_ub=False, fsdp_manual_registration=False),
    )
    args = SimpleNamespace(
        model_family_name="qwen",
        model_recipe_name="qwen3_30b_a3b",
        task="pretrain",
        wandb_experiment_name="deterministic-test",
        deterministic=True,
        expert_model_parallel_size=None,
        nccl_ub=False,
        moe_flex_dispatcher_backend=-1,
        use_recipes=True,
    )
    exec_environments = []
    parser = SimpleNamespace(parse_known_args=lambda: (args, []))

    monkeypatch.setattr(bootstrap, "parse_cli_args", lambda: parser)
    monkeypatch.setattr(utils, "build_recipe_config", lambda **_kwargs: recipe)
    monkeypatch.setattr(run_recipe, "_finalize_recipe", lambda prepared, *_args: prepared)
    monkeypatch.setattr(bootstrap.os, "execvpe", lambda _executable, _argv, env: exec_environments.append(env))

    bootstrap.main()

    assert len(exec_environments) == 1
    exec_environment = exec_environments[0]
    assert {name: exec_environment[name] for name in deterministic_env} == deterministic_env


def test_hydra_environment_override_wins_after_deterministic_argparse(monkeypatch):
    """Explicit Hydra env values remain final in the deterministic pre-exec pass."""
    recipe = SimpleNamespace(
        env_vars={},
        model=SimpleNamespace(
            deterministic_mode=False,
            cross_entropy_loss_fusion=True,
            moe_flex_dispatcher_backend=None,
            moe_token_dispatcher_type=None,
            num_moe_experts=None,
        ),
        comm_overlap=None,
        ddp=SimpleNamespace(nccl_ub=False, fsdp_manual_registration=False),
    )
    args = SimpleNamespace(
        deterministic=True,
        expert_model_parallel_size=None,
        nccl_ub=False,
        moe_flex_dispatcher_backend=-1,
    )

    def apply_hydra(config, overrides):
        assert config.env_vars["NCCL_ALGO"] == "Ring"
        assert overrides == ["env_vars.NCCL_ALGO=Tree"]
        config.env_vars["NCCL_ALGO"] = "Tree"
        return config

    monkeypatch.setattr(run_recipe, "_apply_hydra_overrides", apply_hydra)

    effective_recipe = run_recipe._apply_recipe_overrides(
        recipe,
        args,
        ["env_vars.NCCL_ALGO=Tree"],
        environment_only=True,
    )

    assert effective_recipe.env_vars["NCCL_ALGO"] == "Tree"


def test_runner_disables_flex_dispatcher_consistently():
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

    effective_recipe = run_recipe._apply_recipe_overrides(recipe, args, [], environment_only=True)
    effective_recipe = utils.finalize_config_overrides(effective_recipe)

    assert effective_recipe.model.moe_token_dispatcher_type == "alltoall"
    assert effective_recipe.model.moe_flex_dispatcher_backend is None


@pytest.mark.parametrize("dispatcher_override", [-1, "hybridep"])
def test_runner_preserves_dense_dispatcher_fields(dispatcher_override):
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

    effective_recipe = run_recipe._apply_recipe_overrides(recipe, args, [], environment_only=True)
    effective_recipe = utils.finalize_config_overrides(effective_recipe)

    assert effective_recipe.model.moe_token_dispatcher_type is None
    assert effective_recipe.model.moe_flex_dispatcher_backend is None


def test_runner_uses_three_stage_recipe_preparation():
    """Bootstrap and training share base, override, and finalize stages."""
    repo_root = Path(__file__).resolve().parents[4]
    expected_calls = {
        repo_root / "scripts" / "performance" / "run_recipe.py": {
            "_prepare_recipe": {"build_recipe_config", "_apply_recipe_overrides", "_finalize_recipe"},
            "_apply_recipe_overrides": {
                "apply_argparse_overrides",
                "apply_determinism_overrides",
                "_apply_training_argparse_overrides",
                "_apply_hydra_overrides",
            },
            "_apply_training_argparse_overrides": {"apply_argparse_overrides"},
            "_finalize_recipe": {
                "finalize_config_overrides",
                "explicit_environment_override_names",
                "apply_target_topology_environment",
                "apply_feature_environment",
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


def test_hydra_ep_override_wins_after_known_ep_override(monkeypatch):
    """Hydra EP must win after argparse EP in the final runner config."""
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

    monkeypatch.setattr(run_recipe, "_apply_hydra_overrides", apply_hydra)

    effective_recipe = run_recipe._apply_recipe_overrides(
        recipe,
        args,
        ["model.expert_model_parallel_size=64"],
        environment_only=True,
    )

    assert effective_recipe.model.expert_model_parallel_size == 64


def test_hydra_disable_clears_argparse_dependent_state(monkeypatch):
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

    monkeypatch.setattr(run_recipe, "_apply_hydra_overrides", apply_hydra)

    effective_recipe = run_recipe._apply_recipe_overrides(
        recipe,
        args,
        ["ddp.nccl_ub=false", "model.moe_flex_dispatcher_backend=null"],
        environment_only=True,
    )
    effective_recipe = utils.finalize_config_overrides(effective_recipe)

    assert effective_recipe.ddp.nccl_ub is False
    assert effective_recipe.ddp.fsdp_manual_registration is False
    assert effective_recipe.model.moe_token_dispatcher_type == "alltoall"
    assert effective_recipe.model.moe_flex_dispatcher_backend is None


def test_prepare_recipe_runs_base_override_and_finalize_stages(monkeypatch):
    """Recipe preparation has one visible path with exactly three config stages."""
    args = SimpleNamespace(
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
    final_recipe = SimpleNamespace(env_vars={"FINAL": 1})
    calls = []
    cli_overrides = ["model.expert_model_parallel_size=64"]

    def build_config(**kwargs):
        calls.append(("base", kwargs))
        return base_recipe

    def apply_overrides(recipe, parsed_args, overrides, *, environment_only):
        calls.append(("overrides", recipe, parsed_args, overrides, environment_only))
        return effective_recipe

    def finalize(recipe, parsed_args, overrides, base_env_vars):
        calls.append(("finalize", recipe, parsed_args, overrides, base_env_vars))
        return final_recipe

    monkeypatch.setattr(utils, "build_recipe_config", build_config)
    monkeypatch.setattr(run_recipe, "_apply_recipe_overrides", apply_overrides)
    monkeypatch.setattr(run_recipe, "_finalize_recipe", finalize)

    result = run_recipe._prepare_recipe(args, cli_overrides, environment_only=True)

    assert result is final_recipe
    assert calls == [
        (
            "base",
            {
                "model_family_name": "deepseek",
                "model_recipe_name": "deepseek_v3_32nodes",
                "train_task": "pretrain",
                "wandb_experiment_name": "test-experiment",
            },
        ),
        ("overrides", base_recipe, args, cli_overrides, True),
        ("finalize", effective_recipe, args, cli_overrides, {"TORCHINDUCTOR_WORKER_START": "fork"}),
    ]


def test_bootstrap_exports_model_recipe_environment_before_exec(monkeypatch):
    """The bootstrap exports a prepared model recipe before training."""
    args = SimpleNamespace(use_recipes=True)
    parser = SimpleNamespace(parse_known_args=lambda: (args, []))
    effective_recipe = SimpleNamespace(env_vars={"TORCHINDUCTOR_WORKER_START": "fork"})
    calls = []

    def apply_environment(recipe):
        calls.append(("environment", recipe))

    def exec_runner(executable, argv, env):
        calls.append(("exec", executable, argv, env))

    monkeypatch.setattr(bootstrap, "parse_cli_args", lambda: parser)
    monkeypatch.setattr(run_recipe, "_prepare_recipe", lambda *_args, **_kwargs: effective_recipe)
    monkeypatch.setattr(bootstrap, "_apply_recipe_environment", apply_environment)
    monkeypatch.setattr(bootstrap.os, "execvpe", exec_runner)

    bootstrap.main()

    assert calls[0] == ("environment", effective_recipe)
    assert calls[1][0] == "exec"
    assert calls[1][2][1].endswith("run_recipe.py")
