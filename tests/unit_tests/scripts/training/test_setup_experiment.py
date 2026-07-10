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


def _load_setup_experiment_module():
    script = Path(__file__).resolve().parents[4] / "scripts" / "training" / "setup_experiment.py"
    nemo_run = types.ModuleType("nemo_run")
    nemo_run_config = types.ModuleType("nemo_run.config")
    nemo_run_config.get_nemorun_home = lambda: str(Path.home() / ".nemo_run")
    nemo_run.config = nemo_run_config
    previous = sys.modules.get("nemo_run")
    previous_config = sys.modules.get("nemo_run.config")
    sys.modules["nemo_run"] = nemo_run
    sys.modules["nemo_run.config"] = nemo_run_config
    try:
        spec = importlib.util.spec_from_file_location("test_training_setup_experiment", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("nemo_run", None)
        else:
            sys.modules["nemo_run"] = previous
        if previous_config is None:
            sys.modules.pop("nemo_run.config", None)
        else:
            sys.modules["nemo_run.config"] = previous_config
    return module


def test_parser_forwards_training_selection_and_overrides():
    module = _load_setup_experiment_module()

    args, training_args = module.parse_args(
        [
            "--local",
            "--gpus-per-node",
            "1",
            "--recipe",
            "gpt_oss_20b_sft_config",
            "--mode",
            "sft",
            "--dataset",
            "openmathinstruct2",
            "optimizer.lr=0.0001",
        ]
    )

    assert args.local is True
    assert training_args == [
        "--recipe",
        "gpt_oss_20b_sft_config",
        "--mode",
        "sft",
        "--dataset",
        "openmathinstruct2",
        "optimizer.lr=0.0001",
    ]


def test_selector_injects_topology_but_full_recipe_does_not():
    module = _load_setup_experiment_module()

    selector = module._inject_selector_topology(
        ["--family", "gpt_oss", "--model", "gpt_oss_20b", "--mode", "pretrain"],
        nodes=2,
        gpus_per_node=8,
        gpu_type="h100",
    )
    full_recipe = module._inject_selector_topology(
        ["--recipe", "gpt_oss_20b_pretrain_config", "--mode", "pretrain"],
        nodes=2,
        gpus_per_node=8,
        gpu_type=None,
    )

    assert selector[-4:] == ["--gpus", "16", "--gpu", "h100"]
    assert full_recipe == ["--recipe", "gpt_oss_20b_pretrain_config", "--mode", "pretrain"]


def test_selector_requires_gpu_type():
    module = _load_setup_experiment_module()

    with pytest.raises(ValueError, match="requires --gpu-type"):
        module._inject_selector_topology(
            ["--family", "gpt_oss", "--model", "gpt_oss_20b", "--mode", "pretrain"],
            nodes=1,
            gpus_per_node=8,
            gpu_type=None,
        )


def test_recipe_and_model_are_mutually_exclusive():
    module = _load_setup_experiment_module()
    args, training_args = module.parse_args(
        [
            "--local",
            "--gpus-per-node",
            "1",
            "--recipe",
            "gpt_oss_20b_pretrain_config",
            "--model",
            "gpt_oss_20b",
        ]
    )

    with pytest.raises(ValueError, match="already identifies the model"):
        module._validate_args(args, training_args)


def test_dataset_prefix_and_cache_are_mounted_automatically(tmp_path, monkeypatch):
    module = _load_setup_experiment_module()
    prefix = tmp_path / "dclm_text_document"
    prefix.with_suffix(".bin").touch()
    prefix.with_suffix(".idx").touch()
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("DCLM_CACHE", str(cache))

    mounts = module._discover_mounts(["--dataset-path", str(prefix)], [])

    assert f"{tmp_path}:{tmp_path}" in mounts
    assert f"{cache}:{cache}" in mounts


def test_nonexistent_output_mounts_nearest_existing_parent(tmp_path):
    module = _load_setup_experiment_module()
    output = tmp_path / "new" / "checkpoints"

    mounts = module._discover_mounts(["--save-dir", str(output)], [])

    assert f"{tmp_path}:{tmp_path}" in mounts


def test_invalid_env_syntax_is_rejected():
    module = _load_setup_experiment_module()

    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        module._parse_env(["MISSING_VALUE"])


def test_slurm_executor_configures_local_tunnel_job_dir(tmp_path, monkeypatch):
    module = _load_setup_experiment_module()

    class _SlurmExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.run.LocalTunnel = lambda **kwargs: types.SimpleNamespace(**kwargs)
    module.run.Packager = object
    module.run.SlurmExecutor = _SlurmExecutor
    monkeypatch.setattr(module, "get_nemorun_home", lambda: str(tmp_path))
    args, _ = module.parse_args(
        [
            "--gpus-per-node",
            "1",
            "--account",
            "account",
            "--partition",
            "partition",
            "--container-image",
            "image.sqsh",
            "--recipe",
            "gpt_oss_20b_pretrain_config",
        ]
    )

    executor = module._build_executor(args, {}, [])

    assert executor.kwargs["tunnel"].job_dir == str(tmp_path / "experiments")


@pytest.mark.parametrize("status", ["FAILED", "CANCELLED", "UNKNOWN"])
def test_synchronous_experiment_failure_is_propagated(status):
    module = _load_setup_experiment_module()

    class _Experiment:
        @staticmethod
        def from_title(_experiment_name):
            return types.SimpleNamespace(status=lambda **_kwargs: {"training": {"status": status}})

    module.run.Experiment = _Experiment

    with pytest.raises(RuntimeError, match=rf"training={status}"):
        module._ensure_experiment_succeeded("test-experiment")


def test_synchronous_experiment_success_is_accepted():
    module = _load_setup_experiment_module()

    class _Experiment:
        @staticmethod
        def from_title(_experiment_name):
            return types.SimpleNamespace(status=lambda **_kwargs: {"training": {"status": "SUCCEEDED"}})

    module.run.Experiment = _Experiment

    module._ensure_experiment_succeeded("test-experiment")
