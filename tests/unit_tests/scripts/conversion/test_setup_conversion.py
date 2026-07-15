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

import importlib.util
import sys
import types
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "scripts" / "conversion"


def _load_setup_conversion_module():
    nemo_run = types.ModuleType("nemo_run")
    nemo_run_config = types.ModuleType("nemo_run.config")
    nemo_run_config.get_nemorun_home = lambda: str(Path.home() / ".nemo_run")
    nemo_run.config = nemo_run_config

    previous_modules = {name: sys.modules.get(name) for name in ("nemo_run", "nemo_run.config", "arguments")}
    sys.modules["nemo_run"] = nemo_run
    sys.modules["nemo_run.config"] = nemo_run_config
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "test_conversion_setup_conversion", SCRIPT_DIR / "setup_conversion.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_DIR))
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def _parse(module, *options):
    return module.build_parser(include_execution=True).parse_args(
        ["import", "--hf-model", "hf/model", "--megatron-path", "/checkpoint", *options]
    )


def test_setup_import_is_lightweight(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "megatron.bridge", raising=False)

    _load_setup_conversion_module()

    assert "torch" not in sys.modules
    assert "megatron.bridge" not in sys.modules


def test_cpu_backend_rejects_distributed_parallelism():
    module = _load_setup_conversion_module()
    args = _parse(module, "--tp", "2")

    with pytest.raises(ValueError, match="TP=PP=EP=ETP=1"):
        module._validate_args(args)


def test_gpu_backend_requires_gpu_count():
    module = _load_setup_conversion_module()
    args = _parse(module, "--device", "gpu")

    with pytest.raises(ValueError, match="requires --gpus-per-node"):
        module._validate_args(args)


def test_gpu_backend_validates_parallelism_against_world_size():
    module = _load_setup_conversion_module()
    args = _parse(module, "--device", "gpu", "--gpus-per-node", "4", "--tp", "3")

    with pytest.raises(ValueError, match=r"nodes\*gpus-per-node"):
        module._validate_args(args)


def test_gpu_backend_validates_expert_parallelism_against_world_size():
    module = _load_setup_conversion_module()
    args = _parse(module, "--device", "gpu", "--gpus-per-node", "4", "--etp", "2", "--ep", "4")

    with pytest.raises(ValueError, match=r"ETP\*EP\*PP"):
        module._validate_args(args)


def test_gpu_backend_allows_etp_topology_independent_from_tp():
    module = _load_setup_conversion_module()
    args = _parse(module, "--device", "gpu", "--gpus-per-node", "8", "--tp", "4", "--ep", "4")

    module._validate_args(args)


def test_local_cpu_executor_uses_one_process_without_launcher():
    module = _load_setup_conversion_module()
    captured = {}
    module.run.Packager = lambda: "packager"
    module.run.LocalExecutor = lambda **kwargs: captured.update(kwargs) or types.SimpleNamespace(**kwargs)
    args = _parse(module)

    executor = module._build_executor(args, [], [])

    assert executor.ntasks_per_node == 1
    assert executor.nodes == 1
    assert executor.launcher is None


def test_local_gpu_executor_uses_nemo_run_torchrun():
    module = _load_setup_conversion_module()
    launcher = object()
    module.run.Packager = lambda: "packager"
    module.run.Torchrun = lambda: launcher
    module.run.LocalExecutor = lambda **kwargs: types.SimpleNamespace(**kwargs)
    args = _parse(module, "--device", "gpu", "--gpus-per-node", "4", "--ep", "4")
    module._validate_args(args)

    executor = module._build_executor(args, [], [])

    assert executor.ntasks_per_node == 4
    assert executor.launcher is launcher


def test_slurm_cpu_executor_does_not_request_gpus(tmp_path, monkeypatch):
    module = _load_setup_conversion_module()

    class _SlurmExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.run.Packager = lambda: "packager"
    module.run.LocalTunnel = lambda **kwargs: types.SimpleNamespace(**kwargs)
    module.run.SlurmExecutor = _SlurmExecutor
    monkeypatch.setattr(module, "get_nemorun_home", lambda: str(tmp_path))
    args = _parse(
        module,
        "--executor",
        "slurm",
        "--account",
        "account",
        "--partition",
        "partition",
        "--container-image",
        "image.sqsh",
    )
    module._validate_args(args)

    executor = module._build_executor(args, ["HF_TOKEN"], ["/host:/container"])

    assert executor.kwargs["ntasks_per_node"] == 1
    assert "gpus_per_node" not in executor.kwargs
    assert executor.kwargs["container_env"] == ["HF_TOKEN", "PYTHONPATH"]
    assert executor.kwargs["additional_parameters"] == {"export": "HF_TOKEN,PYTHONPATH"}
    assert executor.env_vars == {}


def test_slurm_gpu_executor_uses_srun_native_tasks(tmp_path, monkeypatch):
    module = _load_setup_conversion_module()

    class _SlurmExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.run.Packager = lambda: "packager"
    module.run.LocalTunnel = lambda **kwargs: types.SimpleNamespace(**kwargs)
    module.run.SlurmExecutor = _SlurmExecutor
    monkeypatch.setattr(module, "get_nemorun_home", lambda: str(tmp_path))
    args = _parse(
        module,
        "--executor",
        "slurm",
        "--device",
        "gpu",
        "--gpus-per-node",
        "4",
        "--account",
        "account",
        "--partition",
        "partition",
        "--container-image",
        "image.sqsh",
        "--ep",
        "4",
    )
    module._validate_args(args)

    executor = module._build_executor(args, [], [])

    assert executor.kwargs["ntasks_per_node"] == 4
    assert executor.kwargs["launcher"] is None


def test_main_waits_and_builds_local_worker_task(monkeypatch):
    module = _load_setup_conversion_module()
    scripts = []
    run_calls = []
    executor = object()

    class _Script:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            scripts.append(self)

    class _Experiment:
        def __init__(self, _name):
            self.jobs = [types.SimpleNamespace(id="import-cpu", state=module.AppState.SUCCEEDED)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def add(self, _task, *, executor: object, name: str):
            assert executor is sentinel_executor
            assert name == "import-cpu"

        def run(self, **kwargs):
            run_calls.append(kwargs)

    sentinel_executor = executor
    module.run.Script = _Script
    module.run.Experiment = _Experiment
    monkeypatch.setattr(module, "_build_executor", lambda *_args: sentinel_executor)

    module.main(["import", "--hf-model", "hf/model", "--megatron-path", "/checkpoint"])

    assert run_calls == [{"detach": False, "tail_logs": True}]
    assert scripts[0].path == str(REPO_ROOT / "scripts/conversion/run_conversion.py")
    assert scripts[0].args[:3] == ["import", "--device", "cpu"]


def test_main_propagates_worker_failure(monkeypatch):
    module = _load_setup_conversion_module()

    class _Experiment:
        def __init__(self, _name):
            self.jobs = [types.SimpleNamespace(id="import-cpu", state=module.AppState.FAILED)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def add(self, *_args, **_kwargs):
            pass

        def run(self, **_kwargs):
            pass

    module.run.Packager = lambda: "packager"
    module.run.LocalExecutor = lambda **kwargs: types.SimpleNamespace(**kwargs)
    module.run.Script = lambda **kwargs: types.SimpleNamespace(**kwargs)
    module.run.Experiment = _Experiment

    with pytest.raises(RuntimeError, match="import-cpu=FAILED"):
        module.main(["import", "--hf-model", "hf/model", "--megatron-path", "/checkpoint"])


def test_parse_env_rejects_inline_secret_values(monkeypatch):
    module = _load_setup_conversion_module()
    monkeypatch.setenv("HF_TOKEN", "secret")

    with pytest.raises(ValueError, match="accepts NAME only"):
        module._parse_env(["HF_TOKEN=secret"])
