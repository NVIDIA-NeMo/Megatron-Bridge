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

"""Tests for scripts/performance/utils/executors.py — container_env on SlurmExecutor."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


# scripts/performance is not an installed package; add it to sys.path so we
# can import ``utils.executors`` the same way the scripts themselves do.
_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

try:
    import nemo_run  # noqa: F401

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

if HAS_NEMO_RUN:
    from setup_experiment import _build_nemorun_script
    from utils import executors as executors_module
    from utils.executors import (
        KUBEFLOW_NUMA_BINDING_ENV,
        OFFLINE_BENCHMARK_ENV_VARS,
        _kubeflow_numa_binding_enabled,
        _kubeflow_numa_binding_script,
        kubeflow_executor,
        slurm_executor,
    )


RECIPE_PROCESS_ENV_NAMES = {
    "NCCL_GRAPH_REGISTER",
    "NCCL_NVLS_ENABLE",
    "NVTE_NORM_BWD_USE_CUDNN",
    "NVTE_NORM_FWD_USE_CUDNN",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_NCCL_AVOID_RECORD_STREAMS",
    "TORCH_NCCL_HIGH_PRIORITY",
}


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_container_env_includes_offline_benchmark_vars(tmp_path):
    """Offline benchmark defaults must override matching container values."""
    executor = slurm_executor(
        gpu="h100",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=1,
        num_gpus_per_node=8,
    )
    assert executor.container_env is not None, "container_env is None — was the field removed from the executor?"
    missing = set(OFFLINE_BENCHMARK_ENV_VARS) - set(executor.container_env)
    assert not missing, f"Offline benchmark vars missing from container_env: {missing}"


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_custom_env_vars_in_container_env(tmp_path):
    """Vars passed via custom_env_vars must also appear in container_env."""
    executor = slurm_executor(
        gpu="h100",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=1,
        num_gpus_per_node=8,
        custom_env_vars={"MY_CUSTOM_VAR": "1"},
    )
    assert "MY_CUSTOM_VAR" in executor.container_env


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_kubeflow_numa_binding_wraps_each_torchrun_worker():
    """The opt-in Kubeflow launcher must resolve and bind each worker's local GPU."""
    assert _kubeflow_numa_binding_enabled({KUBEFLOW_NUMA_BINDING_ENV: "1"})
    task = nemo_run.Script(
        path="train.py",
        entrypoint="python",
        args=["--steps", "10"],
        env={"PYTHONPATH": "/workspace:$PYTHONPATH"},
        metadata={"test": "value"},
    )
    wrapper = _kubeflow_numa_binding_script(task)
    assert 'nvidia-smi -i "$LOCAL_RANK"' in wrapper.inline
    assert "head -n1" not in wrapper.inline
    assert 'NUMA_FILE="/sys/bus/pci/devices/$PCI_BUS/numa_node"' in wrapper.inline
    assert (
        'exec numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" python train.py --steps 10' in wrapper.inline
    )
    assert wrapper.env == task.env
    assert wrapper.env is not task.env
    assert wrapper.metadata == task.metadata
    assert wrapper.metadata is not task.metadata


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_build_nemorun_script_wraps_only_enabled_kubeflow_tasks():
    """The setup helper must preserve task env while gating the Kubeflow wrapper."""
    kwargs = {
        "script_path": "/opt/Megatron-Bridge/scripts/performance/run_recipe.py",
        "repo_root": "/opt/Megatron-Bridge",
        "args": ["--steps", "10"],
    }
    enabled = _build_nemorun_script(
        **kwargs,
        kubeflow_namespace="nemo-ci",
        custom_env_vars={KUBEFLOW_NUMA_BINDING_ENV: "1"},
    )
    disabled = _build_nemorun_script(
        **kwargs,
        kubeflow_namespace="nemo-ci",
        custom_env_vars={},
    )
    non_kubeflow = _build_nemorun_script(
        **kwargs,
        kubeflow_namespace=None,
        custom_env_vars={KUBEFLOW_NUMA_BINDING_ENV: "1"},
    )

    expected_env = {
        "PYTHONPATH": (
            "/opt/Megatron-Bridge/scripts/performance:"
            "/opt/Megatron-Bridge/src:"
            "/opt/Megatron-Bridge/3rdparty/Megatron-LM:"
            "$PYTHONPATH"
        )
    }
    assert enabled.inline
    assert enabled.env == expected_env
    assert not disabled.inline
    assert disabled.env == expected_env
    assert not non_kubeflow.inline
    assert non_kubeflow.env == expected_env


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_kubeflow_numa_binding_is_disabled_by_default():
    """Normal Kubeflow jobs must retain the unmodified Torchrun launcher."""
    assert not _kubeflow_numa_binding_enabled({})
    assert not _kubeflow_numa_binding_enabled({KUBEFLOW_NUMA_BINDING_ENV: "0"})


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_executor_never_supplies_recipe_process_defaults(tmp_path):
    """Flat performance and model recipes supply process settings without executor shadowing."""
    executor = slurm_executor(
        gpu="h100",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=1,
        num_gpus_per_node=8,
    )

    assert RECIPE_PROCESS_ENV_NAMES.isdisjoint(executor.env_vars)
    assert "TRANSFORMERS_OFFLINE" in executor.env_vars


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_recipe_env_vars_are_exported_and_forced_into_container(tmp_path):
    """Launcher-resolved recipe vars must reach Slurm tasks before Python starts."""
    recipe_env_vars = {
        "TORCHINDUCTOR_WORKER_START": "fork",
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": "64",
    }
    executor = slurm_executor(
        gpu="gb200",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=2,
        num_gpus_per_node=4,
        custom_env_vars=recipe_env_vars,
    )

    assert executor.env_vars.items() >= recipe_env_vars.items()
    assert set(recipe_env_vars) <= set(executor.container_env or [])


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_recipe_env_vars_are_added_to_kubeflow_trainer_environment(monkeypatch):
    """Kubeflow workers should inherit launcher-resolved recipe variables."""
    monkeypatch.setattr(
        executors_module.run,
        "KubeflowExecutor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    recipe_env_vars = {
        "TORCHINDUCTOR_WORKER_START": "fork",
        "QUANTIZATION_TYPE_DEBUG": "1",
    }
    executor = kubeflow_executor(
        namespace="test",
        nodes=2,
        num_gpus_per_node=4,
        custom_env_vars=recipe_env_vars,
    )

    assert executor.env_vars.items() >= recipe_env_vars.items()
