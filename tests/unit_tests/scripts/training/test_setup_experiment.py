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


def _load_performance_recipe_module():
    script = Path(__file__).resolve().parents[4] / "scripts" / "training" / "performance_recipe.py"
    module_name = "test_training_performance_recipe"
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return module


def test_parser_forwards_training_selection_and_overrides():
    module = _load_setup_experiment_module()

    args, training_args = module.parse_args(
        [
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

    assert args.gpus_per_node == 1
    assert args.srun_args == []
    assert training_args == [
        "--recipe",
        "gpt_oss_20b_sft_config",
        "--mode",
        "sft",
        "--dataset",
        "openmathinstruct2",
        "optimizer.lr=0.0001",
    ]


def test_parser_consumes_repeatable_srun_args():
    module = _load_setup_experiment_module()

    args, training_args = module.parse_args(
        [
            "--srun-arg=--mpi=pmix",
            "--srun-arg=--container-writable",
            "--recipe",
            "gpt_oss_20b_pretrain_config",
        ]
    )

    assert args.srun_args == ["--mpi=pmix", "--container-writable"]
    assert training_args == ["--recipe", "gpt_oss_20b_pretrain_config"]


def test_srun_args_reject_empty_values():
    module = _load_setup_experiment_module()
    args, _ = module.parse_args(["--srun-arg="])

    with pytest.raises(ValueError, match="must not be empty"):
        module._validate_args(args)


@pytest.mark.parametrize("submission_option", ["--submission-dry-run", "--dry-run"])
def test_submission_dry_run_aliases_are_consumed(submission_option):
    module = _load_setup_experiment_module()

    args, training_args = module.parse_args([submission_option, "--dry_run"])

    assert args.submission_dry_run is True
    assert training_args == ["--dry_run"]


def test_setup_import_does_not_load_training_stack(monkeypatch):
    monkeypatch.delitem(sys.modules, "recipe_runner", raising=False)
    monkeypatch.delitem(sys.modules, "megatron.bridge", raising=False)

    _load_setup_experiment_module()

    assert "recipe_runner" not in sys.modules
    assert "megatron.bridge" not in sys.modules


def test_parse_env_deduplicates_inherited_names(monkeypatch):
    module = _load_setup_experiment_module()
    monkeypatch.setenv("INHERITED_VALUE", "from-launcher")

    env_names = module._parse_env(["INHERITED_VALUE", "INHERITED_VALUE"])

    assert env_names == ["INHERITED_VALUE"]


def test_parse_env_rejects_inline_values(monkeypatch):
    module = _load_setup_experiment_module()
    monkeypatch.setenv("SECRET_VALUE", "from-launcher")

    with pytest.raises(ValueError, match="accepts NAME only"):
        module._parse_env(["SECRET_VALUE=inline"])


def test_parse_env_rejects_missing_inherited_name(monkeypatch):
    module = _load_setup_experiment_module()
    monkeypatch.delenv("MISSING_VALUE", raising=False)

    with pytest.raises(ValueError, match="is not set"):
        module._parse_env(["MISSING_VALUE"])


def test_parse_mounts_supports_same_path_and_explicit_destination():
    module = _load_setup_experiment_module()

    mounts = module._parse_mounts(["/shared/data", "/host/cache:/container/cache", "/shared/data"])

    assert mounts == ["/shared/data:/shared/data", "/host/cache:/container/cache"]


@pytest.mark.parametrize("value", ["", ":/container", "/host:"])
def test_parse_mounts_rejects_empty_paths(value):
    module = _load_setup_experiment_module()

    with pytest.raises(ValueError, match="expected HOST or HOST:CONTAINER"):
        module._parse_mounts([value])


def test_container_image_defaults_only_from_public_environment(monkeypatch):
    module = _load_setup_experiment_module()
    monkeypatch.setenv("CONTAINER_IMAGE", "public.sqsh")
    monkeypatch.setenv("MB_CONTAINER_IMAGE", "private.sqsh")

    args, _ = module.parse_args([])

    assert args.container_image == "public.sqsh"


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
            "--mode",
            "pretrain",
        ]
    )

    executor = module._build_executor(args, ["HF_TOKEN"], ["/host:/container"])

    assert executor.kwargs["tunnel"].job_dir == str(tmp_path / "experiments")
    assert executor.kwargs["ntasks_per_node"] == 1
    assert executor.kwargs["gpus_per_node"] == 1
    assert executor.env_vars == {}
    assert set(executor.container_env) == {"HF_TOKEN", "PYTHONPATH", *module.TRAINING_LAUNCH_ENV}
    assert executor.additional_parameters == {"export": "HF_TOKEN"}
    assert executor.container_mounts == ["/host:/container"]
    assert executor.srun_args == []


def test_slurm_executor_can_skip_gpu_request_for_implicit_whole_node_clusters(tmp_path, monkeypatch):
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
            "8",
            "--no-gpu-resource-request",
            "--account",
            "account",
            "--partition",
            "partition",
            "--container-image",
            "image.sqsh",
            "--srun-arg=--mpi=pmix",
            "--srun-arg=--container-writable",
        ]
    )

    executor = module._build_executor(args, [], [])

    assert executor.kwargs["ntasks_per_node"] == 8
    assert "gpus_per_node" not in executor.kwargs
    assert executor.additional_parameters == {"export": "NIL"}
    assert executor.srun_args == ["--mpi=pmix", "--container-writable"]


@pytest.mark.parametrize(
    ("extra_options", "expected_run", "expected_dryrun"),
    [
        ([], [{"detach": True, "tail_logs": False}], 0),
        (["--wait"], [{"detach": False, "tail_logs": True}], 0),
        (["--dry_run"], [{"detach": True, "tail_logs": False}], 0),
        (["--wait", "--dry_run"], [{"detach": False, "tail_logs": True}], 0),
        (["--submission-dry-run"], [], 1),
        (["--dry-run"], [], 1),
    ],
)
def test_main_keeps_submission_and_training_dry_runs_separate(
    monkeypatch,
    extra_options,
    expected_run,
    expected_dryrun,
):
    module = _load_setup_experiment_module()
    run_kwargs = []
    dryrun_calls = []
    scripts = []

    class _Script:
        def __init__(self, *, path, entrypoint, env, args):
            self.path = path
            self.entrypoint = entrypoint
            self.env = env
            self.args = args
            scripts.append(self)

        def to_command(self):
            return [self.entrypoint, self.path, *self.args]

    class _Experiment:
        def __init__(self, _name):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def add(self, _task, *, executor, name):
            assert executor is sentinel_executor
            assert name == "training"

        def run(self, **kwargs):
            run_kwargs.append(kwargs)

        def dryrun(self):
            dryrun_calls.append(True)

    sentinel_executor = object()
    module.run.Script = _Script
    module.run.Experiment = _Experiment
    monkeypatch.setattr(module, "_build_executor", lambda *_args, **_kwargs: sentinel_executor)

    training_args = ["--recipe", "gpt_oss_20b_pretrain_config", "--mode", "pretrain"]
    module.main(
        [
            "--gpus-per-node",
            "1",
            "--account",
            "account",
            "--partition",
            "partition",
            "--container-image",
            "image.sqsh",
            *training_args,
            *extra_options,
        ]
    )

    assert run_kwargs == expected_run
    assert len(dryrun_calls) == expected_dryrun
    assert len(scripts) == 1
    assert scripts[0].path == "/opt/Megatron-Bridge/scripts/training/run_recipe.py"
    assert scripts[0].entrypoint == "python"
    assert scripts[0].env == {
        **module.TRAINING_LAUNCH_ENV,
        "PYTHONPATH": "/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM:$PYTHONPATH",
    }
    submission_options = {"--submission-dry-run", "--dry-run", "--wait"}
    expected_training_options = [option for option in extra_options if option not in submission_options]
    assert scripts[0].args == [*training_args, *expected_training_options]


def test_train_launcher_routes_exact_performance_recipe(monkeypatch):
    module = _load_setup_experiment_module()
    scripts = []
    dryrun_calls = []

    class _Script:
        def __init__(self, **kwargs):
            scripts.append(types.SimpleNamespace(**kwargs))

    class _Experiment:
        def __init__(self, _name):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def add(self, *_args, **_kwargs):
            pass

        def dryrun(self):
            dryrun_calls.append(True)

    module.run.Script = _Script
    module.run.Experiment = _Experiment
    monkeypatch.setattr(module, "_build_executor", lambda *_args: object())

    module.main(
        [
            "--nodes",
            "4",
            "--gpus-per-node",
            "4",
            "--account",
            "account",
            "--partition",
            "partition",
            "--container-image",
            "image.sqsh",
            "--submission-dry-run",
            "--recipe",
            "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
            "train.train_iters=10",
        ]
    )

    assert len(dryrun_calls) == 1
    assert scripts[0].path == "/opt/Megatron-Bridge/scripts/performance/run_script.py"
    assert scripts[0].args == [
        "--model_family_name",
        "qwen",
        "--model_recipe_name",
        "qwen3_30b_a3b",
        "--task",
        "pretrain",
        "--num_gpus",
        "16",
        "--gpu",
        "h100",
        "--compute_dtype",
        "bf16",
        "--domain",
        "llm",
        "train.train_iters=10",
    ]
    assert scripts[0].env["TORCH_NCCL_AVOID_RECORD_STREAMS"] == "1"
    assert scripts[0].env["PYTHONPATH"].startswith("/opt/Megatron-Bridge/scripts/performance:")

    mismatched_args, _ = module.parse_args(
        [
            "--nodes",
            "3",
            "--gpus-per-node",
            "4",
            "--account",
            "account",
            "--partition",
            "partition",
            "--container-image",
            "image.sqsh",
        ]
    )
    metadata, _ = module.resolve_performance_recipe_args(["--recipe", "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config"])
    assert metadata is not None
    with pytest.raises(ValueError, match="request 12"):
        module._validate_args(mismatched_args, metadata)


def test_all_exact_performance_exports_round_trip_through_legacy_selectors():
    module = _load_performance_recipe_module()
    precision_names = {value: key for key, value in module.PRECISION_ARGUMENTS.items()}
    exports = module.performance_recipe_exports()

    assert exports
    for recipe_name, family in exports.items():
        metadata, runner_args = module.resolve_performance_recipe_args(["--recipe", recipe_name])
        if recipe_name in module.LIBRARY_RECIPE_PRECEDENCE_COLLISIONS:
            assert metadata is None
            assert runner_args == ["--recipe", recipe_name]
            continue

        assert metadata is not None
        assert metadata.model_family_name == family
        assert "--recipe" not in runner_args
        variant = f"_{metadata.config_variant}" if metadata.config_variant is not None else ""
        reconstructed = (
            f"{metadata.model_recipe_name}_{metadata.task}_{metadata.num_gpus}gpu_{metadata.hardware}_"
            f"{precision_names[metadata.compute_dtype]}{variant}_config"
        )
        assert reconstructed == recipe_name


@pytest.mark.parametrize(
    ("recipe_name", "family", "compute_dtype", "config_variant", "domain"),
    [
        ("qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_large_scale_config", "qwen", "fp8_cs", "large_scale", "llm"),
        ("qwen3_vl_30b_a3b_pretrain_16gpu_h100_bf16_config", "qwen_vl", "bf16", None, "qwen3vl"),
        ("wan_14b_pretrain_32gpu_h100_bf16_config", "wan", "bf16", None, "diffusion"),
        ("llama3_8b_sft_8gpu_h100_fp8cs_config", "llama", "fp8_cs", None, "llm"),
        ("llama3_70b_peft_8gpu_h100_fp8cs_config", "llama", "fp8_cs", None, "llm"),
    ],
)
def test_performance_recipe_metadata(recipe_name, family, compute_dtype, config_variant, domain):
    module = _load_performance_recipe_module()

    metadata = module.performance_recipe_metadata(recipe_name)

    assert metadata.model_family_name == family
    assert metadata.compute_dtype == compute_dtype
    assert metadata.config_variant == config_variant
    assert metadata.domain == domain


def test_exact_performance_recipe_rejects_conflicting_selection():
    module = _load_performance_recipe_module()
    recipe_name = "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config"

    with pytest.raises(ValueError, match="already selects the workload.*--task"):
        module.resolve_performance_recipe_args(["--recipe", recipe_name, "--task", "sft"])
    with pytest.raises(ValueError, match="exactly one --recipe"):
        module.resolve_performance_recipe_args(["--recipe", recipe_name, "--recipe", recipe_name])


def test_ambiguous_finetuning_names_keep_library_route():
    module = _load_performance_recipe_module()
    arguments = ["--recipe", "llama3_70b_sft_32gpu_h100_bf16_config"]

    metadata, runner_args = module.resolve_performance_recipe_args(arguments)

    assert metadata is None
    assert runner_args == arguments


def test_main_shell_quotes_forwarded_training_arguments(monkeypatch):
    module = _load_setup_experiment_module()
    scripts = []

    class _Script:
        def __init__(self, **kwargs):
            scripts.append(types.SimpleNamespace(**kwargs))

    class _Experiment:
        def __init__(self, _name):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def add(self, *_args, **_kwargs):
            pass

        def dryrun(self):
            pass

    module.run.Script = _Script
    module.run.Experiment = _Experiment
    monkeypatch.setattr(module, "_build_executor", lambda *_args, **_kwargs: object())
    sentinel = "logger.wandb_exp_name=benign; echo should-not-run"

    module.main(
        [
            "--gpus-per-node",
            "1",
            "--account",
            "account",
            "--partition",
            "partition",
            "--container-image",
            "image.sqsh",
            "--submission-dry-run",
            sentinel,
        ]
    )

    assert scripts[0].args == ["'logger.wandb_exp_name=benign; echo should-not-run'"]
