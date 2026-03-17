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

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _package_module(name: str) -> types.ModuleType:
    """Create a minimal package-like module."""
    module = types.ModuleType(name)
    module.__path__ = []
    return module


@pytest.fixture
def import_performance_module():
    """Import performance modules with only the nemo_run surface these tests need."""
    nemo_run = _package_module("nemo_run")
    nemo_run_config = types.ModuleType("nemo_run.config")
    nemo_run_core = _package_module("nemo_run.core")
    nemo_run_core_execution = _package_module("nemo_run.core.execution")
    nemo_run_core_execution_launcher = types.ModuleType("nemo_run.core.execution.launcher")

    nemo_run.LocalTunnel = lambda **kwargs: SimpleNamespace(**kwargs)
    nemo_run.GitArchivePackager = lambda **kwargs: SimpleNamespace(**kwargs)
    nemo_run.SlurmExecutor = lambda **kwargs: SimpleNamespace(**kwargs)
    nemo_run.DGXCloudExecutor = lambda **kwargs: SimpleNamespace(**kwargs)

    nemo_run_config.get_nemorun_home = lambda: "/tmp/nemorun"
    nemo_run_config.set_nemorun_home = lambda _path: None

    nemo_run_core_execution_launcher.SlurmTemplate = lambda **kwargs: SimpleNamespace(**kwargs)

    nemo_run.config = nemo_run_config
    nemo_run.core = nemo_run_core
    nemo_run_core.execution = nemo_run_core_execution
    nemo_run_core_execution.launcher = nemo_run_core_execution_launcher

    with patch.dict(
        sys.modules,
        {
            "nemo_run": nemo_run,
            "nemo_run.config": nemo_run_config,
            "nemo_run.core": nemo_run_core,
            "nemo_run.core.execution": nemo_run_core_execution,
            "nemo_run.core.execution.launcher": nemo_run_core_execution_launcher,
        },
    ):

        def _import(module_name: str):
            sys.modules.pop(module_name, None)
            return importlib.import_module(module_name)

        yield _import


def test_parse_cli_args_accepts_offline_flag(import_performance_module):
    """The performance CLI should keep exposing the offline switch."""
    argument_parser = import_performance_module("scripts.performance.argument_parser")

    parser = argument_parser.parse_cli_args()
    args = parser.parse_args(
        [
            "--model_family_name",
            "llama",
            "--model_recipe_name",
            "llama3_8b",
            "--num_gpus",
            "8",
            "--gpu",
            "h100",
            "--offline",
        ]
    )

    assert args.offline is True


def test_argparse_rejects_hf_token_with_offline(import_performance_module):
    """argparse should reject --hf_token and --offline together at parse time."""
    argument_parser = import_performance_module("scripts.performance.argument_parser")

    parser = argument_parser.parse_cli_args()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model_family_name",
                "llama",
                "--model_recipe_name",
                "llama3_8b",
                "--num_gpus",
                "8",
                "--gpu",
                "h100",
                "--hf_token",
                "hf_test_token",
                "--offline",
            ]
        )


def test_slurm_executor_sets_offline_env_and_container_writable(import_performance_module):
    """Offline mode should set HF_HUB_OFFLINE and preserve the offline Transformers default."""
    executors = import_performance_module("scripts.performance.utils.executors")

    executor = executors.slurm_executor(
        gpu="h100",
        account="test_account",
        partition="test_partition",
        log_dir="/tmp/log_dir",
        nodes=1,
        num_gpus_per_node=8,
        offline=True,
    )

    assert "--container-writable" in executor.srun_args
    assert executor.env_vars["HF_HUB_OFFLINE"] == "1"
    assert executor.env_vars["TRANSFORMERS_OFFLINE"] == "1"


def test_slurm_executor_default_has_container_writable_and_hub_online(import_performance_module):
    """By default, --container-writable is always set and HF Hub access stays online."""
    executors = import_performance_module("scripts.performance.utils.executors")

    executor = executors.slurm_executor(
        gpu="h100",
        account="test_account",
        partition="test_partition",
        log_dir="/tmp/log_dir",
        nodes=1,
        num_gpus_per_node=8,
    )

    assert "--container-writable" in executor.srun_args
    assert executor.env_vars["HF_HUB_OFFLINE"] == "0"
    assert executor.env_vars["TRANSFORMERS_OFFLINE"] == "1"


def test_slurm_executor_hf_token_enables_online_transformers(import_performance_module):
    """Providing an HF token should enable the online Transformers path."""
    executors = import_performance_module("scripts.performance.utils.executors")

    executor = executors.slurm_executor(
        gpu="h100",
        account="test_account",
        partition="test_partition",
        log_dir="/tmp/log_dir",
        nodes=1,
        num_gpus_per_node=8,
        hf_token="hf_test_token",
    )

    assert executor.env_vars["HF_TOKEN"] == "hf_test_token"
    assert executor.env_vars["TRANSFORMERS_OFFLINE"] == "0"
    assert executor.env_vars["HF_HUB_OFFLINE"] == "0"
