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
    torchx = types.ModuleType("torchx")
    torchx_specs = types.ModuleType("torchx.specs")
    torchx_specs_api = types.ModuleType("torchx.specs.api")
    torchx_specs_api.AppState = types.SimpleNamespace(SUCCEEDED="SUCCEEDED", FAILED="FAILED")
    torchx.specs = torchx_specs
    torchx_specs.api = torchx_specs_api

    fake_modules = {
        "nemo_run": nemo_run,
        "nemo_run.config": nemo_run_config,
        "torchx": torchx,
        "torchx.specs": torchx_specs,
        "torchx.specs.api": torchx_specs_api,
    }
    previous_modules = {name: sys.modules.get(name) for name in (*fake_modules, "arguments")}
    sys.modules.update(fake_modules)
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


def _parse_export(module, *options):
    return module.build_parser(include_execution=True).parse_args(
        [
            "export",
            "--hf-model",
            "hf/model",
            "--megatron-path",
            "/checkpoint",
            "--hf-path",
            "/hf-export",
            *options,
        ]
    )


def _parse_roundtrip(module, *options):
    return module.build_parser(include_execution=True).parse_args(
        ["roundtrip", "--hf-model-id", "hf/model", "--gpus-per-node", "2", *options]
    )


def _parse_hf_comparison(module, *options):
    return module.build_parser(include_execution=True).parse_args(
        ["compare-hf", "--hf-model", "hf/model", "--hf-path", "/candidate", *options]
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


def test_local_executor_rejects_detach():
    module = _load_setup_conversion_module()
    args = _parse(module, "--detach")

    with pytest.raises(ValueError, match="only supported by the Slurm executor"):
        module._validate_args(args)


def test_local_executor_rejects_srun_args():
    module = _load_setup_conversion_module()
    args = _parse(module, "--srun-arg=--mpi=pmix")

    with pytest.raises(ValueError, match="only supported by the Slurm executor"):
        module._validate_args(args)


def test_srun_args_reject_empty_values():
    module = _load_setup_conversion_module()
    args = _parse(module, "--srun-arg= ")

    with pytest.raises(ValueError, match="must not be empty"):
        module._validate_args(args)


def test_cpu_export_rejects_export_weight_dtype():
    module = _load_setup_conversion_module()
    args = _parse_export(module, "--export-weight-dtype", "float32")

    with pytest.raises(ValueError, match="only supported by the GPU backend"):
        module._validate_args(args)


def test_cpu_export_validates_effective_distributed_save():
    module = _load_setup_conversion_module()
    args = _parse_export(module, "--save-every-n-ranks", "2")

    with pytest.raises(ValueError, match="requires --distributed-save"):
        module._validate_args(args)


def test_gpu_backend_requires_gpu_count():
    module = _load_setup_conversion_module()
    args = _parse(module, "--device", "gpu")

    with pytest.raises(ValueError, match="requires --gpus-per-node"):
        module._validate_args(args)


def test_gpu_backend_validates_parallelism_against_world_size():
    module = _load_setup_conversion_module()
    args = _parse(module, "--device", "gpu", "--gpus-per-node", "4", "--tp", "3")

    with pytest.raises(ValueError, match=r"nodes\*gpus-per-node must equal TP\*PP\*EP"):
        module._validate_args(args)


def test_gpu_backend_rejects_replicated_data_parallel_conversion():
    module = _load_setup_conversion_module()
    args = _parse(module, "--device", "gpu", "--gpus-per-node", "8", "--tp", "2", "--ep", "2")

    with pytest.raises(ValueError, match=r"nodes\*gpus-per-node must equal TP\*PP\*EP"):
        module._validate_args(args)


def test_roundtrip_accepts_exact_multinode_product_topology():
    module = _load_setup_conversion_module()
    args = _parse_roundtrip(
        module,
        "--executor",
        "slurm",
        "--nodes",
        "12",
        "--gpus-per-node",
        "8",
        "--account",
        "account",
        "--partition",
        "partition",
        "--container-image",
        "image.sqsh",
        "--tp",
        "2",
        "--ep",
        "48",
    )

    module._validate_args(args)


def test_gpu_backend_validates_expert_parallelism_against_world_size():
    module = _load_setup_conversion_module()
    args = _parse(module, "--device", "gpu", "--gpus-per-node", "4", "--etp", "2", "--ep", "4")

    with pytest.raises(ValueError, match=r"ETP\*EP\*PP"):
        module._validate_args(args)


def test_gpu_backend_allows_etp_topology_independent_from_tp():
    module = _load_setup_conversion_module()
    args = _parse(module, "--device", "gpu", "--gpus-per-node", "8", "--tp", "2", "--ep", "4")

    module._validate_args(args)


def test_roundtrip_rejects_cpu_backend():
    module = _load_setup_conversion_module()
    args = _parse_roundtrip(module, "--device", "cpu")

    with pytest.raises(ValueError, match="requires the GPU backend"):
        module._validate_args(args)


def test_hf_comparison_accepts_cpu_without_gpu_resource():
    module = _load_setup_conversion_module()
    args = _parse_hf_comparison(
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


def test_hf_comparison_rejects_gpu_backend():
    module = _load_setup_conversion_module()
    args = _parse_hf_comparison(module, "--device", "gpu", "--gpus-per-node", "1")

    with pytest.raises(ValueError, match="requires the CPU backend"):
        module._validate_args(args)


@pytest.mark.parametrize(
    ("command", "options"),
    [
        ("roundtrip", ("--hf-model-id", "/model path", "--tp", "2")),
        (
            "import",
            ("--device", "gpu", "--gpus-per-node", "2", "--tp", "2", "--megatron-path", "/checkpoint path"),
        ),
        (
            "export",
            ("--device", "gpu", "--gpus-per-node", "2", "--tp", "2", "--hf-path", "/export path"),
        ),
    ],
)
def test_local_gpu_rejects_worker_value_requiring_shell_quoting(command, options):
    module = _load_setup_conversion_module()
    parse = {"roundtrip": _parse_roundtrip, "import": _parse, "export": _parse_export}[command]
    args = parse(module, *options)

    with pytest.raises(ValueError, match="cannot pass model IDs or paths containing whitespace"):
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


def test_roundtrip_task_uses_conversion_worker():
    module = _load_setup_conversion_module()
    module.run.Script = lambda **kwargs: types.SimpleNamespace(**kwargs)
    args = _parse_roundtrip(
        module,
        "--tp",
        "2",
    )
    module._validate_args(args)

    task, display_args = module._build_task(args)

    assert task.path == str(REPO_ROOT / "scripts/conversion/run_conversion.py")
    assert task.entrypoint == "python"
    assert task.env["PYTHON_EXEC"] == sys.executable
    assert task.args == display_args
    assert task.args == [
        "roundtrip",
        "--device",
        "gpu",
        "--hf-model",
        "hf/model",
        "--tp",
        "2",
        "--pp",
        "1",
        "--ep",
        "1",
        "--etp",
        "1",
    ]


def test_slurm_roundtrip_task_uses_container_conversion_worker():
    module = _load_setup_conversion_module()
    module.run.Script = lambda **kwargs: types.SimpleNamespace(**kwargs)
    args = _parse_roundtrip(
        module,
        "--executor",
        "slurm",
        "--account",
        "account",
        "--partition",
        "partition",
        "--container-image",
        "image.sqsh",
        "--hf-model-id",
        "/model path",
        "--tp",
        "2",
    )
    module._validate_args(args)

    task, display_args = module._build_task(args)

    assert task.path == "/opt/Megatron-Bridge/scripts/conversion/run_conversion.py"
    assert task.entrypoint == "python"
    assert "PYTHON_EXEC" not in task.env
    assert task.args != display_args
    assert display_args[4] == "/model path"
    assert task.args == [*display_args[:4], "'/model path'", *display_args[5:]]


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
        "--experiment-name",
        "mb4909-nano4b-conversion",
    )
    module._validate_args(args)

    executor = module._build_executor(args, ["HF_TOKEN"], ["/host:/container"])

    assert executor.kwargs["ntasks_per_node"] == 1
    assert executor.kwargs["exclusive"] is None
    assert executor.kwargs["job_name_prefix"] == "mb4909-nano4b-conversion"
    assert "cpus_per_task" not in executor.kwargs
    assert "gpus_per_node" not in executor.kwargs
    assert executor.kwargs["container_env"] == ["HF_TOKEN", "PYTHONPATH"]
    assert executor.kwargs["additional_parameters"] == {"export": "HF_TOKEN,PYTHONPATH"}
    assert executor.kwargs["srun_args"] == []
    assert executor.env_vars == {}


def test_slurm_cpu_executor_can_request_gpu_runtime(tmp_path, monkeypatch):
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
        "--gpus-per-node",
        "1",
        "--account",
        "account",
        "--partition",
        "partition",
        "--container-image",
        "image.sqsh",
    )
    module._validate_args(args)

    executor = module._build_executor(args, [], [])

    assert executor.kwargs["ntasks_per_node"] == 1
    assert executor.kwargs["gpus_per_node"] == 1
    assert executor.kwargs["exclusive"] is None


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
        "--srun-arg=--mpi=pmix",
        "--srun-arg=--cpus-per-task=8",
        "--ep",
        "4",
    )
    module._validate_args(args)

    executor = module._build_executor(args, [], [])

    assert executor.kwargs["ntasks_per_node"] == 4
    assert executor.kwargs["exclusive"] is None
    assert "cpus_per_task" not in executor.kwargs
    assert executor.kwargs["launcher"] is None
    assert executor.kwargs["srun_args"] == ["--mpi=pmix", "--cpus-per-task=8"]


def test_slurm_executor_requests_exclusive_node_only_when_explicit(tmp_path, monkeypatch):
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
        "--exclusive",
        "--account",
        "account",
        "--partition",
        "partition",
        "--container-image",
        "image.sqsh",
    )
    module._validate_args(args)

    executor = module._build_executor(args, [], [])

    assert executor.kwargs["exclusive"] is True


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
