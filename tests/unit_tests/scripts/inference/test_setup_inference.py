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
SCRIPT_PATH = REPO_ROOT / "scripts" / "inference" / "setup_inference.py"


def _load_setup_inference_module():
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
    previous_modules = {name: sys.modules.get(name) for name in fake_modules}
    sys.modules.update(fake_modules)
    try:
        spec = importlib.util.spec_from_file_location("test_inference_setup_inference", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def _launcher_args(*extra_options: str) -> list[str]:
    return [
        "--gpus-per-node",
        "1",
        "--account",
        "account",
        "--partition",
        "partition",
        "--container-image",
        "image.sqsh",
        *extra_options,
    ]


def test_setup_import_does_not_load_inference_stack(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "megatron.bridge", raising=False)

    _load_setup_inference_module()

    assert "torch" not in sys.modules
    assert "megatron.bridge" not in sys.modules


def test_parser_forwards_model_checkpoint_prompt_and_engine_args():
    module = _load_setup_inference_module()

    args, inference_args = module.parse_args(
        [
            "--gpus-per-node",
            "2",
            "--hf-model-path",
            "hf/model",
            "--megatron-model-path",
            "/checkpoints/model",
            "--prompt",
            "A prompt with spaces",
            "--tp",
            "2",
            "--max_new_tokens",
            "8",
        ]
    )

    assert args.gpus_per_node == 2
    assert inference_args == [
        "--hf-model-path",
        "hf/model",
        "--megatron-model-path",
        "/checkpoints/model",
        "--prompt",
        "A prompt with spaces",
        "--tp",
        "2",
        "--max_new_tokens",
        "8",
    ]


def test_parser_consumes_repeatable_srun_args():
    module = _load_setup_inference_module()

    args, inference_args = module.parse_args(
        [
            "--srun-arg=--mpi=pmix",
            "--srun-arg=--container-writable",
            "--prompt",
            "hello",
        ]
    )

    assert args.srun_args == ["--mpi=pmix", "--container-writable"]
    assert inference_args == ["--prompt", "hello"]


@pytest.mark.parametrize("submission_option", ["--submission-dry-run", "--dry-run"])
def test_submission_dry_run_aliases_are_consumed(submission_option):
    module = _load_setup_inference_module()

    args, inference_args = module.parse_args([submission_option, "--prompt", "hello"])

    assert args.submission_dry_run is True
    assert inference_args == ["--prompt", "hello"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (("--nodes", "0"), "--nodes"),
        (("--gpus-per-node", "0"), "--gpus-per-node"),
        (("--cpus-per-task", "0"), "--cpus-per-task"),
        (("--srun-arg=",), "--srun-arg"),
    ],
)
def test_resource_validation_rejects_invalid_values(options, message):
    module = _load_setup_inference_module()
    args, _ = module.parse_args(_launcher_args(*options))

    with pytest.raises(ValueError, match=message):
        module._validate_args(args)


def test_parse_env_deduplicates_names_and_rejects_values(monkeypatch):
    module = _load_setup_inference_module()
    monkeypatch.setenv("HF_TOKEN", "secret")

    assert module._parse_env(["HF_TOKEN", "HF_TOKEN"]) == ["HF_TOKEN"]
    with pytest.raises(ValueError, match="accepts NAME only"):
        module._parse_env(["HF_TOKEN=secret"])


def test_parse_env_rejects_missing_names(monkeypatch):
    module = _load_setup_inference_module()
    monkeypatch.delenv("MISSING_VALUE", raising=False)

    with pytest.raises(ValueError, match="is not set"):
        module._parse_env(["MISSING_VALUE"])


def test_parse_mounts_normalizes_and_deduplicates():
    module = _load_setup_inference_module()

    mounts = module._parse_mounts(["/shared", "/host:/container", "/shared"])

    assert mounts == ["/shared:/shared", "/host:/container"]


@pytest.mark.parametrize("value", ["", ":/container", "/host:"])
def test_parse_mounts_rejects_empty_paths(value):
    module = _load_setup_inference_module()

    with pytest.raises(ValueError, match="expected HOST or HOST:CONTAINER"):
        module._parse_mounts([value])


def test_slurm_executor_uses_srun_native_tasks_and_keeps_secrets_out(tmp_path, monkeypatch):
    module = _load_setup_inference_module()

    class _SlurmExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.run.Packager = lambda: "packager"
    module.run.LocalTunnel = lambda **kwargs: types.SimpleNamespace(**kwargs)
    module.run.SlurmExecutor = _SlurmExecutor
    monkeypatch.setattr(module, "get_nemorun_home", lambda: str(tmp_path))
    args, _ = module.parse_args(
        _launcher_args(
            "--gpus-per-node",
            "4",
            "--srun-arg=--mpi=pmix",
            "--srun-arg=--container-writable",
        )
    )

    executor = module._build_executor(args, ["HF_TOKEN"], ["/host:/container"])

    assert executor.kwargs["ntasks_per_node"] == 4
    assert executor.kwargs["gpus_per_node"] == 4
    assert executor.kwargs["exclusive"] is None
    assert "launcher" not in executor.kwargs
    assert executor.kwargs["tunnel"].job_dir == str(tmp_path / "experiments")
    assert executor.kwargs["container_env"] == ["HF_TOKEN"]
    assert executor.kwargs["additional_parameters"] == {"export": "PATH,HF_TOKEN"}
    assert executor.kwargs["container_mounts"] == ["/host:/container"]
    assert executor.kwargs["srun_args"] == ["--mpi=pmix", "--container-writable"]
    assert executor.env_vars == {}


def test_slurm_executor_can_request_exclusive_nodes(tmp_path, monkeypatch):
    module = _load_setup_inference_module()

    class _SlurmExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.run.Packager = lambda: "packager"
    module.run.LocalTunnel = lambda **kwargs: types.SimpleNamespace(**kwargs)
    module.run.SlurmExecutor = _SlurmExecutor
    monkeypatch.setattr(module, "get_nemorun_home", lambda: str(tmp_path))
    args, _ = module.parse_args(_launcher_args("--exclusive"))

    executor = module._build_executor(args, [], [])

    assert executor.kwargs["exclusive"] is True


def test_slurm_executor_can_skip_explicit_gpu_request(tmp_path, monkeypatch):
    module = _load_setup_inference_module()

    class _SlurmExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.run.Packager = lambda: "packager"
    module.run.LocalTunnel = lambda **kwargs: types.SimpleNamespace(**kwargs)
    module.run.SlurmExecutor = _SlurmExecutor
    monkeypatch.setattr(module, "get_nemorun_home", lambda: str(tmp_path))
    args, _ = module.parse_args(_launcher_args("--no-gpu-resource-request"))

    executor = module._build_executor(args, [], [])

    assert executor.kwargs["ntasks_per_node"] == 1
    assert "gpus_per_node" not in executor.kwargs
    assert executor.kwargs["additional_parameters"] == {"export": "PATH"}


def test_build_task_quotes_prompts_and_uses_existing_entrypoint():
    module = _load_setup_inference_module()
    scripts = []
    module.run.Script = lambda **kwargs: scripts.append(types.SimpleNamespace(**kwargs)) or scripts[-1]
    sentinel = "benign; echo should-not-run"

    module._build_task(["--prompt", sentinel])

    assert scripts[0].path == "/opt/Megatron-Bridge/scripts/inference/text_generation.py"
    assert scripts[0].entrypoint == "python"
    assert scripts[0].env == {
        "PYTHONPATH": "/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM:$PYTHONPATH"
    }
    assert scripts[0].args == ["--prompt", "'benign; echo should-not-run'"]


@pytest.mark.parametrize(
    ("extra_options", "expected_run", "expected_dryrun"),
    [
        ([], [{"detach": False, "tail_logs": True}], 0),
        (["--detach"], [{"detach": True, "tail_logs": False}], 0),
        (["--submission-dry-run"], [], 1),
        (["--dry-run"], [], 1),
    ],
)
def test_main_submission_wait_detach_and_dry_run_modes(
    monkeypatch,
    extra_options,
    expected_run,
    expected_dryrun,
):
    module = _load_setup_inference_module()
    run_calls = []
    dryrun_calls = []

    class _Experiment:
        def __init__(self, name):
            assert name == "inference"
            self.jobs = [types.SimpleNamespace(id="text-generation", state=module.AppState.SUCCEEDED)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def add(self, task, *, executor, name):
            assert task is sentinel_task
            assert executor is sentinel_executor
            assert name == "text-generation"

        def run(self, **kwargs):
            run_calls.append(kwargs)

        def dryrun(self):
            dryrun_calls.append(True)

    sentinel_task = object()
    sentinel_executor = object()
    module.run.Experiment = _Experiment
    monkeypatch.setattr(module, "_build_executor", lambda *_args: sentinel_executor)
    monkeypatch.setattr(module, "_build_task", lambda *_args: sentinel_task)

    module.main(_launcher_args("--prompt", "hello", *extra_options))

    assert run_calls == expected_run
    assert len(dryrun_calls) == expected_dryrun


def test_main_propagates_synchronous_inference_failure(monkeypatch):
    module = _load_setup_inference_module()

    class _Experiment:
        def __init__(self, _name):
            self.jobs = [types.SimpleNamespace(id="text-generation", state=module.AppState.FAILED)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def add(self, *_args, **_kwargs):
            pass

        def run(self, **kwargs):
            assert kwargs == {"detach": False, "tail_logs": True}

    module.run.Experiment = _Experiment
    monkeypatch.setattr(module, "_build_executor", lambda *_args: object())
    monkeypatch.setattr(module, "_build_task", lambda *_args: object())

    with pytest.raises(RuntimeError, match="text-generation=FAILED"):
        module.main(_launcher_args("--prompt", "hello"))
