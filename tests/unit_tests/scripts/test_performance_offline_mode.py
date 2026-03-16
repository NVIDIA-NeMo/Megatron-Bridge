# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import importlib.util
import sys
import types
from pathlib import Path

import pytest


SCRIPTS_PERF_PATH = Path(__file__).parents[3] / "scripts" / "performance"
ARGUMENT_PARSER_PATH = SCRIPTS_PERF_PATH / "argument_parser.py"
SETUP_EXPERIMENT_PATH = SCRIPTS_PERF_PATH / "setup_experiment.py"
EXECUTORS_PATH = SCRIPTS_PERF_PATH / "utils" / "executors.py"


def _package_module(name: str) -> types.ModuleType:
    """Create a minimal package-like module."""
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_module(module_name: str, path: Path):
    """Load a module from disk under a specific name."""
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_fake_nemo_run(monkeypatch):
    """Install lightweight nemo_run stubs so performance scripts can be imported in unit tests."""
    nemo_run = types.ModuleType("nemo_run")
    nemo_run_config = types.ModuleType("nemo_run.config")
    nemo_run_core = _package_module("nemo_run.core")
    nemo_run_core_execution = _package_module("nemo_run.core.execution")
    nemo_run_core_execution_launcher = types.ModuleType("nemo_run.core.execution.launcher")

    nemo_run.LocalTunnel = lambda **kwargs: types.SimpleNamespace(**kwargs)
    nemo_run.GitArchivePackager = lambda **kwargs: types.SimpleNamespace(**kwargs)
    nemo_run.SlurmExecutor = lambda **kwargs: types.SimpleNamespace(**kwargs)
    nemo_run.DGXCloudExecutor = lambda **kwargs: types.SimpleNamespace(**kwargs)

    nemo_run_config.get_nemorun_home = lambda: "/tmp/nemorun"
    nemo_run_config.set_nemorun_home = lambda _path: None

    nemo_run_core_execution_launcher.SlurmTemplate = lambda **kwargs: types.SimpleNamespace(**kwargs)

    nemo_run.config = nemo_run_config
    nemo_run.core = nemo_run_core
    nemo_run_core.execution = nemo_run_core_execution
    nemo_run_core_execution.launcher = nemo_run_core_execution_launcher

    monkeypatch.setitem(sys.modules, "nemo_run", nemo_run)
    monkeypatch.setitem(sys.modules, "nemo_run.config", nemo_run_config)
    monkeypatch.setitem(sys.modules, "nemo_run.core", nemo_run_core)
    monkeypatch.setitem(sys.modules, "nemo_run.core.execution", nemo_run_core_execution)
    monkeypatch.setitem(sys.modules, "nemo_run.core.execution.launcher", nemo_run_core_execution_launcher)


def _install_setup_experiment_stubs(monkeypatch):
    """Install the minimal dependency graph needed to import setup_experiment."""
    _install_fake_nemo_run(monkeypatch)
    _load_module("argument_parser", ARGUMENT_PARSER_PATH)

    utils_pkg = _package_module("utils")
    utils_evaluate = types.ModuleType("utils.evaluate")
    utils_executors = types.ModuleType("utils.executors")
    utils_utils = types.ModuleType("utils.utils")
    perf_plugins = types.ModuleType("perf_plugins")
    resiliency_plugins = types.ModuleType("resiliency_plugins")

    utils_evaluate.calc_convergence_and_performance = lambda *args, **kwargs: None
    utils_executors.slurm_executor = lambda **kwargs: types.SimpleNamespace(**kwargs)
    utils_executors.dgxc_executor = lambda **kwargs: types.SimpleNamespace(**kwargs)
    utils_utils.get_exp_name_config = lambda *args, **kwargs: "test_config"
    utils_utils.select_config_variant_interactive = lambda **kwargs: "v1"

    class _DummyPlugin:
        def __init__(self, *args, **kwargs):
            pass

    perf_plugins.NsysPlugin = _DummyPlugin
    perf_plugins.PerfEnvPlugin = _DummyPlugin
    perf_plugins.PyTorchProfilerPlugin = _DummyPlugin
    resiliency_plugins.FaultTolerancePlugin = _DummyPlugin

    utils_pkg.evaluate = utils_evaluate
    utils_pkg.executors = utils_executors
    utils_pkg.utils = utils_utils

    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.evaluate", utils_evaluate)
    monkeypatch.setitem(sys.modules, "utils.executors", utils_executors)
    monkeypatch.setitem(sys.modules, "utils.utils", utils_utils)
    monkeypatch.setitem(sys.modules, "perf_plugins", perf_plugins)
    monkeypatch.setitem(sys.modules, "resiliency_plugins", resiliency_plugins)


def _minimal_main_kwargs():
    """Return a minimal valid kwargs payload for setup_experiment.main()."""
    return {
        "use_recipes": True,
        "model_family_name": "llama",
        "model_recipe_name": "llama3_8b",
        "task": "pretrain",
        "compute_dtype": "bf16",
        "gpu": "h100",
        "hf_token": None,
        "offline": False,
        "detach": True,
        "dryrun": True,
        "enable_vboost": False,
        "enable_nsys": False,
        "pytorch_profiler": False,
        "moe_a2a_overlap": False,
        "tp_size": None,
        "pp_size": None,
        "cp_size": None,
        "vp_size": None,
        "ep_size": None,
        "etp_size": None,
        "micro_batch_size": None,
        "global_batch_size": None,
        "wandb_key": None,
        "wandb_project_name": None,
        "wandb_experiment_name": None,
        "wandb_entity_name": None,
        "profiling_start_step": 10,
        "profiling_stop_step": 11,
        "record_memory_history": False,
        "profiling_gpu_metrics": False,
        "profiling_ranks": None,
        "nsys_trace": None,
        "nsys_extra_args": None,
        "nemo_home": "/tmp/nemo",
        "account": "test_account",
        "partition": "test_partition",
        "log_dir": "/tmp/log_dir",
        "gpus_per_node": 8,
        "time_limit": "00:30:00",
        "container_image": "nvcr.io/nvidia/nemo:dev",
        "custom_mounts": [],
        "custom_env_vars": {},
        "custom_srun_args": [],
        "custom_bash_cmds": [],
        "nccl_ub": False,
        "pretrained_checkpoint": None,
        "num_gpus": 8,
        "is_long_convergence_run": False,
        "additional_slurm_params": None,
        "golden_values_path": "",
        "convergence_params": {},
        "performance_params": {},
        "memory_params": {},
        "max_retries": 1,
        "dgxc_base_url": "",
        "dgxc_cluster": "",
        "dgxc_kube_apiserver_url": "",
        "dgxc_app_id": "",
        "dgxc_app_secret": "",
        "dgxc_project_name": "",
        "dgxc_pvc_claim_name": "",
        "dgxc_pvc_mount_path": "",
        "config_variant": "v1",
        "gres": None,
    }


def test_parse_cli_args_accepts_offline_flag(monkeypatch):
    """The performance CLI should keep exposing the offline switch."""
    _install_fake_nemo_run(monkeypatch)
    argument_parser = _load_module("test_perf_argument_parser", ARGUMENT_PARSER_PATH)

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
            "--account",
            "test_account",
            "--partition",
            "test_partition",
            "--offline",
        ]
    )

    assert args.offline is True


def test_setup_experiment_rejects_hf_token_with_offline(monkeypatch):
    """The launcher should fail fast if online token mode and offline mode are both requested."""
    _install_setup_experiment_stubs(monkeypatch)
    setup_experiment = _load_module("test_perf_setup_experiment", SETUP_EXPERIMENT_PATH)

    kwargs = _minimal_main_kwargs()
    kwargs.update({"hf_token": "hf_test_token", "offline": True})

    with pytest.raises(ValueError, match="--hf_token and --offline cannot be used together"):
        setup_experiment.main(**kwargs)


def test_slurm_executor_sets_offline_env_and_container_writable(monkeypatch):
    """Offline mode should disable HF Hub access and keep the writable-container workaround."""
    _install_fake_nemo_run(monkeypatch)
    executors = _load_module("test_perf_executors", EXECUTORS_PATH)

    # PERF_ENV_VARS is intentionally mutated in-place by the launcher. Copy it so this test stays isolated.
    monkeypatch.setattr(executors, "PERF_ENV_VARS", executors.PERF_ENV_VARS.copy())

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
