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
    from argument_parser import parse_cli_args
    from setup_experiment import _build_nemorun_script
    from utils.executors import (
        INLINE_TEMPLATE,
        KUBEFLOW_NUMA_BINDING_ENV,
        PERF_ENV_VARS,
        _kubeflow_numa_binding_enabled,
        _kubeflow_numa_binding_script,
        slurm_executor,
    )


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_container_env_includes_perf_vars(tmp_path):
    """PERF_ENV_VARS keys must appear in container_env so they override container defaults."""
    executor = slurm_executor(
        gpu="h100",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=1,
        num_gpus_per_node=8,
    )
    assert executor.container_env is not None, "container_env is None — was the field removed from the executor?"
    missing = set(PERF_ENV_VARS) - set(executor.container_env)
    assert not missing, f"PERF_ENV_VARS keys missing from container_env: {missing}"


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
        "script_dir": "/opt/Megatron-Bridge/scripts/performance",
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

    expected_env = {"PYTHONPATH": "/opt/Megatron-Bridge/scripts/performance:$PYTHONPATH"}
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
def test_parser_accepts_repeatable_post_hooks():
    args = parse_cli_args().parse_args(
        [
            "--model_family_name",
            "llama",
            "--model_recipe_name",
            "llama3_8b",
            "--num_gpus",
            "8",
            "--gpu",
            "h100",
            "--custom_post_bash_cmds",
            "echo",
            "post-one",
            "--custom_post_bash_cmds",
            "echo",
            "post-two",
        ]
    )

    assert args.custom_post_bash_cmds == [["echo", "post-one"], ["echo", "post-two"]]


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_slurm_executor_preserves_training_status_and_defaults_to_noop_post_hook(tmp_path):
    executor = slurm_executor(
        gpu="h100",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=1,
        num_gpus_per_node=8,
    )

    assert executor.launcher.template_vars["post_cmds"] == ":"
    assert "NEMO_RUN_TRAINING_EXIT_CODE" in INLINE_TEMPLATE
    assert 'exit "${TRAIN_RC}"' in INLINE_TEMPLATE


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_slurm_executor_renders_custom_post_hooks(tmp_path):
    executor = slurm_executor(
        gpu="h100",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=1,
        num_gpus_per_node=8,
        custom_post_bash_cmds=[["echo", "first"], ["echo", "second"]],
    )

    assert executor.launcher.template_vars["post_cmds"] == "echo first ; echo second"


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_kubeflow_script_wraps_pre_and_post_hooks():
    task = _build_nemorun_script(
        script_path="/opt/Megatron-Bridge/scripts/performance/run_script.py",
        script_dir="/opt/Megatron-Bridge/scripts/performance",
        args=["--max_steps", "10"],
        kubeflow_namespace="nemo-ci",
        custom_env_vars={},
        custom_bash_cmds=[["bash", "/hooks/pre.sh"]],
        custom_post_bash_cmds=[["bash", "/hooks/post.sh"]],
    )

    wrapped = task.args[0]
    assert task.entrypoint == "bash"
    assert "bash /hooks/pre.sh" in wrapped
    assert "bash /hooks/post.sh" in wrapped
    assert "NEMO_RUN_TRAINING_EXIT_CODE" in wrapped
    assert "KUBEFLOW_FAILURE_SLEEP_SECONDS" in wrapped
    assert "NEMO_CLUSTERDIAG_FAILURE_SLEEP_SECONDS" in wrapped
    assert "Failure detected train_rc=" in wrapped
    assert 'exit "${TRAIN_RC}"' in wrapped
