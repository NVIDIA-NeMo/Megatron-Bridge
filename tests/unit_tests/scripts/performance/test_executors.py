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
    from setup_experiment import _build_nemorun_script, _filter_run_script_args
    from utils.executors import (
        INLINE_TEMPLATE,
        KUBEFLOW_NUMA_BINDING_ENV,
        KUBEFLOW_TORCHRUN_RDZV_READ_TIMEOUT_ENV,
        PERF_ENV_VARS,
        _host_hook_setup_lines,
        _kubeflow_numa_binding_enabled,
        _kubeflow_numa_binding_script,
        _patch_kubeflow_torchrun_launch_script,
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
    wrapped_command = wrapper.args[1]
    assert wrapper.path == "bash"
    assert wrapper.entrypoint == "/usr/bin/env"
    assert wrapper.args[0] == "-lc"
    assert 'nvidia-smi -i "$LOCAL_RANK"' in wrapped_command
    assert "head -n1" not in wrapped_command
    assert 'NUMA_FILE="/sys/bus/pci/devices/$PCI_BUS/numa_node"' in wrapped_command
    assert (
        'exec numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" python train.py --steps 10' in wrapped_command
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
    assert enabled.path == "bash"
    assert enabled.entrypoint == "/usr/bin/env"
    assert enabled.args[0] == "-lc"
    assert 'nvidia-smi -i "$LOCAL_RANK"' in enabled.args[1]
    assert enabled.env == expected_env
    assert disabled.path == kwargs["script_path"]
    assert disabled.entrypoint == "python"
    assert disabled.env == expected_env
    assert non_kubeflow.path == kwargs["script_path"]
    assert non_kubeflow.entrypoint == "python"
    assert non_kubeflow.env == expected_env


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_kubeflow_numa_binding_is_disabled_by_default():
    """Normal Kubeflow jobs must retain the unmodified Torchrun launcher."""
    assert not _kubeflow_numa_binding_enabled({})
    assert not _kubeflow_numa_binding_enabled({KUBEFLOW_NUMA_BINDING_ENV: "0"})


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_kubeflow_launch_script_patches_torchrun_timeout_once():
    """Patch the generated launch.sh with one configurable rendezvous timeout."""
    script = (
        "#!/usr/bin/env bash\n"
        "torchrun --rdzv-backend c10d --rdzv-endpoint $PET_MASTER_ADDR:29500 "
        "--rdzv-id 123 --nnodes 2 --nproc-per-node 4 --node-rank $PET_NODE_RANK "
        "--tee 3 --no-python bash -lc 'python train.py'\n"
    )

    patched = _patch_kubeflow_torchrun_launch_script(script)

    expected = f"--rdzv-conf read_timeout=${{{KUBEFLOW_TORCHRUN_RDZV_READ_TIMEOUT_ENV}:-300}}"
    assert expected in patched
    assert _patch_kubeflow_torchrun_launch_script(patched) == patched


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_filter_run_script_args_removes_launcher_only_hook_and_kubeflow_values():
    argv = [
        "--max_steps",
        "10",
        "--custom_bash_cmds",
        "bash",
        "/hooks/pre.sh",
        "--kubeflow_volumes_json",
        '[{"name":"workdir"}]',
        "--csp=aws",
        "model.num_layers=2",
    ]

    assert _filter_run_script_args(argv) == ["--max_steps", "10", "model.num_layers=2"]


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

    wrapped = task.args[1]
    assert task.path == "bash"
    assert task.args[0] == "-lc"
    assert task.args[2] == "nemo-kubeflow-hook-wrapper"
    assert "bash /hooks/pre.sh" in wrapped
    assert "bash /hooks/post.sh" in wrapped
    assert "NEMO_RUN_TRAINING_EXIT_CODE" in wrapped
    assert 'ARTIFACT_DIR="${NEMO_CLUSTERDIAG_ARTIFACT_DIR:-${NEMO_CLUSTERDIAG_RUN_DIR:-.}}"' in wrapped
    assert 'TRAIN_LOG="${ARTIFACT_DIR}/ranks/train-rank-${RANK_ID}.log"' in wrapped
    assert '2>&1 | tee "${TRAIN_LOG}"' in wrapped
    assert 'TRAIN_RC="${PIPESTATUS[0]}"' in wrapped
    assert '"$@"' in wrapped
    assert 'exit "${TRAIN_RC}"' in wrapped


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_parser_accepts_slurm_host_hooks():
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
            "--host_pre_hook",
            "/shared/hooks/pre.sh",
            "--host_post_hook",
            "/shared/hooks/post.sh",
        ]
    )

    assert args.host_pre_hook == "/shared/hooks/pre.sh"
    assert args.host_post_hook == "/shared/hooks/post.sh"


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
def test_slurm_executor_renders_host_hooks_outside_the_container(tmp_path):
    executor = slurm_executor(
        gpu="h100",
        account="test",
        partition="test",
        log_dir=str(tmp_path),
        nodes=2,
        num_gpus_per_node=8,
        host_pre_hook="/shared/hooks/pre.sh",
        host_post_hook="/shared/hooks/post.sh",
    )

    assert executor.env_vars["NEMO_CLUSTERDIAG_HOST_PRE_HOOK"] == "/shared/hooks/pre.sh"
    assert executor.env_vars["NEMO_CLUSTERDIAG_HOST_POST_HOOK"] == "/shared/hooks/post.sh"
    assert executor.setup_lines == _host_hook_setup_lines(
        host_pre_hook="/shared/hooks/pre.sh",
        host_post_hook="/shared/hooks/post.sh",
    )
    assert '--ntasks="${SLURM_NNODES}"' in executor.setup_lines
    assert "--ntasks-per-node=1" in executor.setup_lines
    assert 'bash "${NEMO_CLUSTERDIAG_HOST_PRE_HOOK}"' in executor.setup_lines
    assert 'bash "${NEMO_CLUSTERDIAG_HOST_POST_HOOK}"' in executor.setup_lines
    assert "trap _nemo_run_host_post_hook EXIT" in executor.setup_lines
    assert "--container" not in executor.setup_lines
