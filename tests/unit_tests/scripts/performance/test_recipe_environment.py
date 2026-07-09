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

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

import run_script_with_env
from utils import utils


def _add_environment(config, custom_env_vars, **overrides):
    args = {
        "custom_env_vars": custom_env_vars,
        "config": config,
        "gpu": "h100",
        "expert_model_parallel_size": None,
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
    )
    custom_env_vars = {"NVTE_FWD_LAYERNORM_SM_MARGIN": "48"}
    original_mapping = custom_env_vars

    _add_environment(config, custom_env_vars)

    assert custom_env_vars is original_mapping
    assert custom_env_vars == {
        "NVTE_FWD_LAYERNORM_SM_MARGIN": "48",
        "TORCHINDUCTOR_WORKER_START": "fork",
    }


@pytest.mark.parametrize(
    ("gpu", "ep_size", "expected_ranks", "expected_domain", "expected_mnnvl"),
    [("h100", 4, "4", "8", "0"), ("gb200", 32, "32", "72", "1")],
)
def test_library_hybridep_topology_environment_tracks_ep_override(
    gpu, ep_size, expected_ranks, expected_domain, expected_mnnvl
):
    config = SimpleNamespace(
        env_vars={
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
            "USE_MNNVL": 1,
        },
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep", expert_model_parallel_size=64),
    )
    custom_env_vars = {}

    _add_environment(config, custom_env_vars, gpu=gpu, expert_model_parallel_size=ep_size)

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
    )
    custom_env_vars = {}

    _add_environment(config, custom_env_vars, gpu=gpu)

    assert custom_env_vars == {
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": expected_ranks,
        "NVLINK_DOMAIN_SIZE": expected_domain,
        "USE_MNNVL": expected_mnnvl,
    }


def test_library_hybridep_topology_preserves_explicit_values():
    """Explicit launcher values retain precedence over normalized recipe defaults."""
    config = SimpleNamespace(
        env_vars={},
        model=SimpleNamespace(moe_flex_dispatcher_backend="hybridep", expert_model_parallel_size=32),
    )
    custom_env_vars = {
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": "custom-ranks",
        "NVLINK_DOMAIN_SIZE": "custom-domain",
        "USE_MNNVL": "custom-mnnvl",
    }

    _add_environment(config, custom_env_vars, gpu="gb200")

    assert custom_env_vars == {
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": "custom-ranks",
        "NVLINK_DOMAIN_SIZE": "custom-domain",
        "USE_MNNVL": "custom-mnnvl",
    }


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

    protected = run_script_with_env._explicit_environment_override_names(
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

    class FakePerfEnvPlugin:
        def __init__(self, **kwargs):
            pass

        def setup_recipe_environment(self, task, executor, config, protected_recipe_env_names=None):
            calls.append(("environment", config, protected_recipe_env_names))

    monkeypatch.setattr(run_script_with_env, "_apply_perf_recipe_overrides", apply_overrides)
    monkeypatch.setattr(run_script_with_env, "_workload_base_config_from_recipe", build_workload)
    monkeypatch.setattr(run_script_with_env, "PerfEnvPlugin", FakePerfEnvPlugin)
    monkeypatch.setattr(run_script_with_env.os, "execvpe", lambda *args: None)

    run_script_with_env.main()

    assert calls[:2] == [
        ("overrides", base_recipe, cli_overrides, args),
        ("workload", effective_recipe, 8),
    ]
    assert calls[2] == ("environment", workload_config, {"TORCHINDUCTOR_WORKER_START"})


def test_library_wrapper_applies_known_ep_override_to_effective_recipe():
    """The library pre-exec config should mirror run_recipe's argparse EP update."""
    recipe = SimpleNamespace(model=SimpleNamespace(expert_model_parallel_size=8))
    args = SimpleNamespace(expert_model_parallel_size=32)

    effective_recipe = run_script_with_env._apply_library_recipe_overrides(recipe, [], args)

    assert effective_recipe.model.expert_model_parallel_size == 32


def test_library_hydra_ep_override_wins_after_known_ep_override(monkeypatch):
    """Hydra EP must win after argparse EP, matching the final library runner."""
    recipe = SimpleNamespace(model=SimpleNamespace(expert_model_parallel_size=8))
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
    )

    with pytest.raises(TypeError, match="must have a scalar value"):
        _add_environment(config, {})
