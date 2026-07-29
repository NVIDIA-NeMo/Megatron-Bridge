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

"""Unit tests for the public recipe training command."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytestmark = pytest.mark.unit


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _load_module():
    script_path = Path(__file__).resolve().parents[4] / "scripts" / "training" / "run_recipe.py"
    module_name = "test_public_run_recipe_module"

    recipe_runner = types.ModuleType("recipe_runner")
    for name in (
        "apply_cli_overrides",
        "apply_determinism",
        "apply_runtime_environment",
        "bootstrap_recipe_environment",
        "load_forward_step",
        "load_recipe",
        "run_config",
        "sync_finetuning_cp_invariants",
        "sync_model_pipeline_layout",
        "sync_offline_packing_alignment",
        "sync_model_dataset_sequence_length",
    ):
        setattr(recipe_runner, name, Mock(name=name))
    recipe_runner.apply_cli_overrides.side_effect = lambda config, _: config
    recipe_runner.apply_determinism.side_effect = lambda config, **_: config
    recipe_runner.apply_runtime_environment.side_effect = lambda config: config
    recipe_runner.bootstrap_recipe_environment.side_effect = lambda config, **_: config
    recipe_runner.sync_finetuning_cp_invariants.side_effect = lambda config, **_: config
    recipe_runner.sync_model_pipeline_layout.side_effect = lambda config, **_: config
    recipe_runner.sync_offline_packing_alignment.side_effect = lambda config: config
    recipe_runner.sync_model_dataset_sequence_length.side_effect = lambda config: config
    recipe_runner.load_forward_step.return_value = object()

    dataset_utils = types.ModuleType("megatron.bridge.recipes.utils.dataset_utils")
    pretraining_datasets = {"mock", "megatron-indexed"}
    dataset_names = {
        *pretraining_datasets,
        "squad",
        "tulu3",
        "openmathinstruct2",
        "openmathinstruct2-thinking",
        "gsm8k",
        "local-jsonl",
        "local-vlm",
        "cord-v2",
        "llava-video-178k",
        "medpix",
        "raven",
        "rdr",
    }
    dataset_utils.DATASET_PRESETS = dict.fromkeys(dataset_names, object())
    dataset_utils.build_dataset_config = Mock(
        side_effect=lambda _config, dataset_name: SimpleNamespace(
            dataset_name=dataset_name,
            num_workers=None,
            pin_memory=None,
            persistent_workers=None,
            split=None,
        )
    )
    dataset_utils.dataset_train_mode = Mock(
        side_effect=lambda dataset: "pretrain" if dataset.dataset_name in pretraining_datasets else "finetune"
    )
    config_module = types.ModuleType("megatron.bridge.training.config")
    config_module.ConfigContainer = object
    stub_modules = {
        "recipe_runner": recipe_runner,
        "megatron": _package("megatron"),
        "megatron.bridge": _package("megatron.bridge"),
        "megatron.bridge.recipes": _package("megatron.bridge.recipes"),
        "megatron.bridge.recipes.utils": _package("megatron.bridge.recipes.utils"),
        "megatron.bridge.recipes.utils.dataset_utils": dataset_utils,
        "megatron.bridge.training": _package("megatron.bridge.training"),
        "megatron.bridge.training.config": config_module,
    }
    previous = {name: sys.modules.get(name) for name in (*stub_modules, module_name)}
    sys.modules.update(stub_modules)

    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module

    return module, SimpleNamespace(
        build_dataset_config=dataset_utils.build_dataset_config,
        recipe_runner=recipe_runner,
    )


@pytest.mark.parametrize(
    "option",
    [
        "--task",
        "--source",
        "--family",
        "--domain",
        "--data",
        "--use-recipes",
        "--step",
        "--set",
        "--hf-path",
        "--hf_path",
        "--recipe-source",
    ],
)
def test_removed_selection_options_are_rejected(option):
    module, _ = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--model", "gpt_oss_20b", "--mode", "pretrain", option, "value"])


@pytest.mark.parametrize("mode", ["finetune", "peft"])
def test_removed_modes_are_rejected(mode):
    module, _ = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--model", "gpt_oss_20b", "--mode", mode])


def test_exactly_one_model_or_recipe_is_required():
    module, _ = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--mode", "pretrain"])
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--model",
                "gpt_oss_20b",
                "--recipe",
                "gpt_oss_20b_pretrain_config",
                "--mode",
                "pretrain",
            ]
        )


def test_model_selection_requires_mode_when_it_cannot_be_inferred():
    module, _ = _load_module()

    with pytest.raises(ValueError, match="Unable to infer training mode"):
        module.parse_args(["--model", "gpt_oss_20b"])


@pytest.mark.parametrize(
    ("recipe_name", "mode"),
    [
        ("gpt_oss_20b_pretrain_config", "pretrain"),
        ("llama3_8b_sft_8gpu_gb200_bf16_config", "sft"),
        ("llama3_70b_peft_8gpu_gb200_bf16_config", "lora"),
    ],
)
def test_conventional_recipe_name_infers_mode(recipe_name, mode):
    module, _ = _load_module()

    args, _ = module.parse_args(["--recipe", recipe_name])

    assert args.mode == mode


@pytest.mark.parametrize("mode", ["lora", "dora"])
def test_model_adapter_modes_select_peft_recipe_and_scheme(mode):
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    module.main(["--model", "gpt_oss_20b", "--mode", mode])

    handles.recipe_runner.load_recipe.assert_called_once_with(
        "gpt_oss_20b_peft_config",
        peft_scheme=mode,
    )
    assert handles.recipe_runner.run_config.call_args.kwargs["mode"] == "finetune"


def test_full_recipe_uses_library_recipe_and_default_llm_step():
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()

    module.main(["--recipe", "gpt_oss_20b_pretrain_config", "--mode", "pretrain"])

    handles.recipe_runner.load_recipe.assert_called_once_with(
        "gpt_oss_20b_pretrain_config",
        peft_scheme=None,
    )
    handles.recipe_runner.load_forward_step.assert_called_once_with("llm_step", mode="pretrain")


def test_kimi_supported_pp_vp_override_refreshes_pipeline_layout():
    module, handles = _load_module()
    default_layout = [[f"stage-{index}"] for index in range(16)]
    overridden_layout = [[f"stage-{index}"] for index in range(4)]
    config = SimpleNamespace(
        model=SimpleNamespace(
            pipeline_model_parallel_size=16,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_layout=default_layout,
            _pipeline_model_parallel_layout_builder=lambda pp, vp: overridden_layout
            if (pp, vp) == (4, 1)
            else default_layout,
        )
    )
    handles.recipe_runner.load_recipe.return_value = config

    def apply_override(current_config, overrides):
        assert overrides == [
            "model.pipeline_model_parallel_size=4",
            "model.virtual_pipeline_model_parallel_size=1",
        ]
        current_config.model.pipeline_model_parallel_size = 4
        current_config.model.virtual_pipeline_model_parallel_size = 1
        return current_config

    def refresh_layout(current_config, *, cli_overrides):
        assert cli_overrides == [
            "model.pipeline_model_parallel_size=4",
            "model.virtual_pipeline_model_parallel_size=1",
        ]
        model = current_config.model
        model.pipeline_model_parallel_layout = model._pipeline_model_parallel_layout_builder(
            model.pipeline_model_parallel_size,
            model.virtual_pipeline_model_parallel_size,
        )
        return current_config

    def validate_layout(*, config, **_):
        model = config.model
        detected_vp = len(model.pipeline_model_parallel_layout) // model.pipeline_model_parallel_size
        assert detected_vp == model.virtual_pipeline_model_parallel_size, (
            "virtual_pipeline_model_parallel_size conflicts with the pipeline layout"
        )

    handles.recipe_runner.apply_cli_overrides.side_effect = apply_override
    handles.recipe_runner.sync_model_pipeline_layout.side_effect = refresh_layout
    handles.recipe_runner.run_config.side_effect = validate_layout

    module.main(["--recipe", "kimi_k2_pretrain_config", "-pp", "4", "-vp", "1"])

    assert config.model.pipeline_model_parallel_layout == overridden_layout


@pytest.mark.parametrize(
    ("recipe_name", "mode", "step_name"),
    [
        ("qwen25_vl_7b_sft_config", "sft", "vlm_step"),
        ("qwen3_vl_8b_sft_config", "sft", "qwen3_vl_step"),
        ("qwen2_audio_7b_sft_config", "sft", "audio_lm_step"),
        ("flux_12b_pretrain_config", "pretrain", "flux_step"),
    ],
)
def test_library_recipe_infers_modality_step(recipe_name, mode, step_name):
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()

    module.main(["--recipe", recipe_name, "--mode", mode])

    handles.recipe_runner.load_forward_step.assert_called_once_with(
        step_name,
        mode="pretrain" if mode == "pretrain" else "finetune",
    )


def test_full_recipe_auto_detects_benchmark_recipe(monkeypatch):
    module, handles = _load_module()
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.delenv("SLURM_NTASKS_PER_NODE", raising=False)
    config = SimpleNamespace(optimizer=SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False))
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--recipe",
            "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
            "--mode",
            "pretrain",
        ]
    )

    handles.recipe_runner.load_recipe.assert_called_once_with(
        "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
        peft_scheme=None,
    )
    assert config.optimizer.use_precision_aware_optimizer is True
    handles.recipe_runner.bootstrap_recipe_environment.assert_called_once()
    bootstrap_call = handles.recipe_runner.bootstrap_recipe_environment.call_args
    assert bootstrap_call.args == (config,)
    assert bootstrap_call.kwargs["script_path"].endswith("scripts/training/run_recipe.py")
    handles.recipe_runner.load_forward_step.assert_called_once_with("llm_step", mode="pretrain")
    handles.recipe_runner.run_config.assert_called_once_with(
        config=config,
        mode="pretrain",
        step_func=handles.recipe_runner.load_forward_step.return_value,
        dryrun=False,
        dryrun_world_size=16,
        dump_environment=False,
    )


@pytest.mark.parametrize(
    "override",
    [
        "optimizer.use_precision_aware_optimizer=false",
        "++optimizer.use_precision_aware_optimizer=false",
        "~optimizer.use_precision_aware_optimizer=false",
        "optimizer={optimizer:adam,use_precision_aware_optimizer:false}",
    ],
)
def test_benchmark_cli_override_wins_over_runtime_default(override):
    module, handles = _load_module()
    config = SimpleNamespace(optimizer=SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False))
    handles.recipe_runner.load_recipe.return_value = config

    def apply_override(current_config, overrides):
        assert current_config.optimizer.use_precision_aware_optimizer is False
        assert overrides == [override]
        current_config.optimizer.use_precision_aware_optimizer = False
        return current_config

    handles.recipe_runner.apply_cli_overrides.side_effect = apply_override

    module.main(
        [
            "--recipe",
            "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
            "--mode",
            "pretrain",
            "--dry-run",
            override,
        ]
    )

    assert config.optimizer.use_precision_aware_optimizer is False


def test_benchmark_optimizer_type_override_recomputes_runtime_default():
    module, handles = _load_module()
    config = SimpleNamespace(optimizer=SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False))
    handles.recipe_runner.load_recipe.return_value = config

    def apply_override(current_config, overrides):
        assert overrides == ["optimizer.optimizer=sgd"]
        current_config.optimizer.optimizer = "sgd"
        return current_config

    handles.recipe_runner.apply_cli_overrides.side_effect = apply_override

    module.main(
        [
            "--recipe",
            "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
            "--mode",
            "pretrain",
            "--dry-run",
            "optimizer.optimizer=sgd",
        ]
    )

    assert config.optimizer.optimizer == "sgd"
    assert config.optimizer.use_precision_aware_optimizer is False


@pytest.mark.parametrize(
    ("mode", "task", "world_size", "recipe_name"),
    [
        ("sft", "sft", 8, "llama3_8b_sft_8gpu_gb200_bf16_config"),
        ("lora", "peft", 8, "llama3_70b_peft_8gpu_gb200_bf16_config"),
        ("sft", "sft", 32, "llama3_70b_sft_32gpu_h100_bf16_config"),
        ("lora", "peft", 8, "llama3_70b_peft_8gpu_h100_bf16_config"),
    ],
)
def test_benchmark_finetuning_recipes_use_unified_runner(monkeypatch, mode, task, world_size, recipe_name):
    module, handles = _load_module()
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    config = SimpleNamespace(
        dataset=SimpleNamespace(num_workers=3, pin_memory=True, persistent_workers=True),
        optimizer=SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False),
    )
    handles.recipe_runner.load_recipe.return_value = config

    module.main(["--recipe", recipe_name, "--mode", mode])

    handles.recipe_runner.load_recipe.assert_called_once_with(
        recipe_name,
        peft_scheme=mode if mode == "lora" else None,
    )
    handles.recipe_runner.bootstrap_recipe_environment.assert_called_once()
    handles.build_dataset_config.assert_called_once_with(config, "mock")
    assert config.dataset.dataset_name == "mock"
    assert config.dataset.num_workers == 3
    assert config.dataset.pin_memory is True
    assert config.dataset.persistent_workers is True
    assert config.dataset.split == "99990,8,2"
    handles.recipe_runner.load_forward_step.assert_called_once_with("llm_step", mode=task)
    assert handles.recipe_runner.run_config.call_args.kwargs["mode"] == "pretrain"


def test_library_only_canonical_name_does_not_enable_benchmark_runtime():
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()

    module.main(
        [
            "--recipe",
            "qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config",
            "--mode",
            "pretrain",
        ]
    )

    handles.recipe_runner.bootstrap_recipe_environment.assert_not_called()
    handles.recipe_runner.load_forward_step.assert_called_once_with("llm_step", mode="pretrain")


def test_benchmark_dry_run_accepts_config_overrides(monkeypatch):
    module, handles = _load_module()
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "heterogeneous-dry-run-allocation")
    config = SimpleNamespace(optimizer=SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False))
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--recipe",
            "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
            "--mode",
            "pretrain",
            "--dry-run",
            "--max_steps",
            "75",
            "logger.log_interval=5",
            "profiling.use_pytorch_profiler=true",
            "env_vars.NCCL_DEBUG=INFO",
            "model.expert_model_parallel_size=8",
            "dataset.seq_length=8192",
            "train.micro_batch_size=2",
        ]
    )

    handles.recipe_runner.apply_cli_overrides.assert_called_once_with(
        config,
        [
            "train.train_iters=75",
            "logger.log_interval=5",
            "profiling.use_pytorch_profiler=true",
            "env_vars.NCCL_DEBUG=INFO",
            "model.expert_model_parallel_size=8",
            "dataset.seq_length=8192",
            "train.micro_batch_size=2",
        ],
    )
    handles.recipe_runner.run_config.assert_called_once_with(
        config=config,
        mode="pretrain",
        step_func=handles.recipe_runner.load_forward_step.return_value,
        dryrun=True,
        dryrun_world_size=16,
        dump_environment=False,
    )


def test_benchmark_recipe_rejects_noncanonical_world_size(monkeypatch):
    module, handles = _load_module()
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace(
        optimizer=SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False)
    )

    with pytest.raises(ValueError, match="requires exactly 16 GPUs"):
        module.main(
            [
                "--recipe",
                "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
                "--mode",
                "pretrain",
            ]
        )

    handles.recipe_runner.bootstrap_recipe_environment.assert_not_called()


def test_benchmark_recipe_accepts_user_selected_per_node_topology(monkeypatch):
    module, handles = _load_module()
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    config = SimpleNamespace(optimizer=SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False))
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--recipe",
            "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
            "--mode",
            "pretrain",
        ]
    )

    handles.recipe_runner.bootstrap_recipe_environment.assert_called_once()


def test_benchmark_recipe_rejects_incompatible_forward_step():
    module, handles = _load_module()

    with pytest.raises(ValueError, match="canonical llm_step"):
        module.main(
            [
                "--recipe",
                "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
                "--mode",
                "pretrain",
                "--step-func",
                "vlm_step",
            ]
        )

    handles.recipe_runner.load_recipe.assert_not_called()


def test_benchmark_recipe_applies_deterministic_overrides(monkeypatch):
    module, handles = _load_module()
    monkeypatch.setenv("WORLD_SIZE", "16")
    config = SimpleNamespace(optimizer=SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False))
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--recipe",
            "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
            "--mode",
            "pretrain",
            "--deterministic",
        ]
    )

    handles.recipe_runner.apply_determinism.assert_called_once_with(config, deterministic=True)


def test_benchmark_recipe_requires_distributed_world_size(monkeypatch):
    module, handles = _load_module()
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("SLURM_NTASKS_PER_NODE", raising=False)
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace(
        optimizer=SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False)
    )

    with pytest.raises(ValueError, match="existing distributed environment"):
        module.main(
            [
                "--recipe",
                "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
                "--mode",
                "pretrain",
            ]
        )

    handles.recipe_runner.bootstrap_recipe_environment.assert_not_called()


def test_benchmark_recipe_rejects_public_dataset_replacement():
    module, handles = _load_module()

    with pytest.raises(ValueError, match="own their canonical dataset"):
        module.main(
            [
                "--recipe",
                "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
                "--mode",
                "pretrain",
                "--dataset",
                "mock",
            ]
        )

    handles.recipe_runner.load_recipe.assert_not_called()


@pytest.mark.parametrize(
    ("recipe_name", "world_size", "step_name"),
    [
        ("qwen3_vl_30b_a3b_pretrain_16gpu_h100_bf16_config", 16, "qwen3_vl_step"),
        ("wan_14b_pretrain_16gpu_gb200_bf16_config", 16, "wan_step"),
    ],
)
def test_non_text_benchmark_recipes_use_modality_step(monkeypatch, recipe_name, world_size, step_name):
    module, handles = _load_module()
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    config = SimpleNamespace(optimizer=SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False))
    handles.recipe_runner.load_recipe.return_value = config

    module.main(["--recipe", recipe_name, "--mode", "pretrain"])

    handles.recipe_runner.bootstrap_recipe_environment.assert_called_once()
    handles.build_dataset_config.assert_not_called()
    handles.recipe_runner.load_forward_step.assert_called_once_with(step_name, mode="pretrain")
    assert handles.recipe_runner.run_config.call_args.kwargs["mode"] == "pretrain"


def test_benchmark_recipe_metadata_accepts_named_variant():
    module, _ = _load_module()

    metadata = module.resolved_benchmark_recipe_metadata(
        "qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_large_scale_config"
    )

    assert metadata is not None
    assert metadata.num_gpus == 256
    assert metadata.family == "qwen"
    assert metadata.hardware == "h100"
    assert metadata.precision == "fp8cs"
    assert metadata.task == "pretrain"


@pytest.mark.parametrize(
    ("recipe_name", "family"),
    [
        ("deepseek_v3_pretrain_64gpu_h100_bf16_config", "deepseek"),
        ("glm51_sft_192gpu_gb200_bf16_config", "glm_moe_dsa"),
        ("nemotron_3_nano_pretrain_16gpu_h100_bf16_config", "nemotronh"),
        ("qwen3_vl_30b_a3b_pretrain_16gpu_h100_bf16_config", "qwen_vl"),
        ("qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config", "qwen_vl"),
        ("wan_14b_pretrain_32gpu_h100_bf16_config", "wan"),
    ],
)
def test_benchmark_recipe_metadata_selects_one_family(recipe_name, family):
    module, _ = _load_module()

    metadata = module.resolved_benchmark_recipe_metadata(recipe_name)

    assert metadata is not None
    assert metadata.family == family


def test_full_recipe_rejects_incompatible_mode():
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()

    with pytest.raises(ValueError, match="incompatible with recipe"):
        module.main(["--recipe", "gpt_oss_20b_sft_config", "--mode", "lora"])


@pytest.mark.parametrize("option", ["--peft-scheme", "--peft_scheme"])
def test_removed_peft_scheme_flags_are_rejected(option):
    module, _ = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--model", "gpt_oss_20b", "--mode", "lora", option, "dora"])


def test_named_finetuning_dataset_maps_to_internal_config():
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    module.main(["--recipe", "gpt_oss_20b_sft_config", "--mode", "sft", "--dataset", "squad"])

    handles.build_dataset_config.assert_called_once_with(config, "squad")
    handles.recipe_runner.apply_cli_overrides.assert_called_once_with(config, [])


def test_tulu3_dataset_is_listed_in_launcher_help():
    module, _ = _load_module()

    help_text = module._build_parser().format_help()

    assert "tulu3" in help_text


@pytest.mark.parametrize(
    ("public_name", "options"),
    [
        ("local-jsonl", ["dataset.dataset_root=/data/sft"]),
        (
            "local-vlm",
            [
                "dataset.source.load_kwargs.data_files.train=/data/vlm.jsonl",
                "dataset.hf_processor_path=Qwen/Qwen3-VL-8B-Instruct",
            ],
        ),
    ],
)
def test_local_dataset_names_use_presets_then_apply_config_overrides(public_name, options):
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    step_options = ["--step-func", "vlm_step"] if public_name == "local-vlm" else []
    module.main(
        ["--recipe", "gpt_oss_20b_sft_config", "--mode", "sft", "--dataset", public_name, *step_options, *options]
    )

    handles.build_dataset_config.assert_called_once_with(config, public_name)
    handles.recipe_runner.apply_cli_overrides.assert_called_once_with(config, options)


@pytest.mark.parametrize(
    ("mode", "dataset"),
    [("pretrain", "squad"), ("sft", "megatron-indexed")],
)
def test_dataset_mode_mismatch_is_rejected(mode, dataset):
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()

    with pytest.raises(ValueError, match="incompatible with dataset"):
        module.main(["--model", "gpt_oss_20b", "--mode", mode, "--dataset", dataset])


@pytest.mark.parametrize("option", ["--step-func", "--step_func"])
def test_step_function_option_accepts_hyphen_and_underscore_spellings(option):
    module, _ = _load_module()

    args, _ = module.parse_args(["--model", "gpt_oss_20b", "--mode", "pretrain", option, "value"])

    assert args.step_func == "value"


def test_help_and_module_docstring_document_common_recipe_overrides():
    module, _ = _load_module()
    help_text = module._build_parser().format_help()
    module_docstring = module.__doc__
    expected_mappings = (
        ("--max_steps", "train.train_iters=STEPS"),
        ("--global_batch_size", "train.global_batch_size=SIZE"),
        ("--micro_batch_size", "train.micro_batch_size=SIZE"),
        ("--tensor_model_parallel_size", "model.tensor_model_parallel_size=N"),
        ("--pipeline_model_parallel_size", "model.pipeline_model_parallel_size=N"),
        ("--context_parallel_size", "model.context_parallel_size=N"),
        ("--virtual_pipeline_model_parallel_size", "model.virtual_pipeline_model_parallel_size=N"),
        ("--expert_model_parallel_size", "model.expert_model_parallel_size=N"),
        ("--expert_tensor_parallel_size", "model.expert_tensor_parallel_size=N"),
        ("--lr", "optimizer.lr=VALUE"),
        ("--min_lr", "optimizer.min_lr=VALUE"),
        ("--warmup_iters", "scheduler.lr_warmup_iters=STEPS"),
        ("--pretrained_checkpoint", "checkpoint.pretrained_checkpoint=PATH"),
        ("--save_dir", "checkpoint.save=PATH"),
        ("--load_dir", "checkpoint.load=PATH"),
        ("--save_interval", "checkpoint.save_interval=STEPS"),
    )

    assert module_docstring is not None
    for recipe_option, config_override in expected_mappings:
        assert recipe_option in module_docstring
        assert config_override in module_docstring
        assert recipe_option in help_text
        assert config_override in help_text
    assert "--seq_length" in help_text
    assert "dataset.seq_length=LENGTH" in help_text
    assert "dataset.sequence_length" not in help_text
    assert "model.seq_length" in help_text


def test_common_convenience_arguments_become_config_overrides():
    module, _ = _load_module()

    _, overrides = module.parse_args(
        [
            "--model",
            "gpt_oss_20b",
            "--mode",
            "pretrain",
            "--max_steps",
            "2",
            "--global_batch_size",
            "16",
            "--micro_batch_size",
            "2",
            "--seq_length",
            "4096",
            "--tensor_model_parallel_size",
            "2",
            "--pipeline_model_parallel_size",
            "4",
            "--context_parallel_size",
            "8",
            "--virtual_pipeline_model_parallel_size",
            "2",
            "--expert_model_parallel_size",
            "4",
            "--expert_tensor_parallel_size",
            "2",
            "--lr",
            "0.0002",
            "--min_lr",
            "0.00002",
            "--warmup_iters",
            "10",
            "--pretrained_checkpoint",
            "/checkpoints/base model",
            "--save_dir",
            "/checkpoints/save",
            "--load_dir",
            "/checkpoints/load",
            "--save_interval",
            "100",
        ]
    )

    assert overrides == [
        "train.train_iters=2",
        "train.global_batch_size=16",
        "train.micro_batch_size=2",
        "dataset.seq_length=4096",
        "model.tensor_model_parallel_size=2",
        "model.pipeline_model_parallel_size=4",
        "model.context_parallel_size=8",
        "model.virtual_pipeline_model_parallel_size=2",
        "model.expert_model_parallel_size=4",
        "model.expert_tensor_parallel_size=2",
        "optimizer.lr=0.0002",
        "optimizer.min_lr=2e-05",
        "scheduler.lr_warmup_iters=10",
        'checkpoint.pretrained_checkpoint="/checkpoints/base model"',
        'checkpoint.save="/checkpoints/save"',
        'checkpoint.load="/checkpoints/load"',
        "checkpoint.save_interval=100",
    ]


@pytest.mark.parametrize(
    ("option", "value", "expected_override"),
    [
        ("-ms", "2", "train.train_iters=2"),
        ("-gb", "16", "train.global_batch_size=16"),
        ("-mb", "2", "train.micro_batch_size=2"),
        ("-sl", "4096", "dataset.seq_length=4096"),
        ("-tp", "2", "model.tensor_model_parallel_size=2"),
        ("-pp", "4", "model.pipeline_model_parallel_size=4"),
        ("-cp", "8", "model.context_parallel_size=8"),
        ("-vp", "2", "model.virtual_pipeline_model_parallel_size=2"),
        ("-ep", "4", "model.expert_model_parallel_size=4"),
        ("-etp", "2", "model.expert_tensor_parallel_size=2"),
    ],
)
def test_common_short_arguments_become_config_overrides(option, value, expected_override):
    module, _ = _load_module()

    _, overrides = module.parse_args(["--model", "gpt_oss_20b", "--mode", "pretrain", option, value])

    assert overrides == [expected_override]


def test_trailing_config_override_takes_precedence_over_convenience_argument():
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--model",
            "gpt_oss_20b",
            "--mode",
            "pretrain",
            "--max_steps",
            "2",
            "train.train_iters=3",
        ]
    )

    handles.recipe_runner.apply_cli_overrides.assert_called_once_with(
        config,
        ["train.train_iters=2", "train.train_iters=3"],
    )


@pytest.mark.parametrize(
    "option",
    [
        "--seq-length",
        "--max-steps",
        "--global-batch-size",
        "--micro-batch-size",
        "--dataset-path",
        "--dataset-root",
        "--offline-packing",
        "--train-data-path",
        "--hf-processor-path",
        "--tp",
        "--from",
        "--wandb-name",
        "--save-config",
        "-et",
    ],
)
def test_unsupported_config_shortcut_spellings_are_rejected(option):
    module, _ = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--model", "gpt_oss_20b", "--mode", "sft", option, "value"])


@pytest.mark.parametrize("option", ["--packed-sequence", "--packed_sequence"])
def test_ambiguous_packed_sequence_flags_are_rejected(option):
    module, _ = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--model", "gpt_oss_20b", "--mode", "sft", option])


def test_thinking_dataset_does_not_imply_offline_packing():
    module, handles = _load_module()
    config = SimpleNamespace(model=SimpleNamespace(seq_length=1024, context_parallel_size=1), dataset=object())
    handles.recipe_runner.load_recipe.return_value = config

    module.main(["--recipe", "gpt_oss_20b_sft_config", "--mode", "sft", "--dataset", "openmathinstruct2-thinking"])

    handles.build_dataset_config.assert_called_once_with(config, "openmathinstruct2-thinking")
    handles.recipe_runner.apply_cli_overrides.assert_called_once_with(config, [])


def test_offline_packing_alignment_is_finalized_after_all_overrides():
    module, handles = _load_module()
    config = SimpleNamespace(
        model=SimpleNamespace(
            seq_length=1024,
            context_parallel_size=1,
            tensor_model_parallel_size=1,
            sequence_parallel=True,
        ),
        dataset=object(),
    )
    handles.recipe_runner.load_recipe.return_value = config
    events = []

    def apply_trailing(current_config, _overrides):
        events.append("trailing")
        current_config.model.context_parallel_size = 2
        current_config.model.tensor_model_parallel_size = 3
        current_config.dataset = SimpleNamespace(seq_length=2048)
        return current_config

    def sync_cp_invariants(current_config, *, mode):
        assert mode == "finetune"
        events.append("cp-invariants")
        return current_config

    def finalize(current_config):
        events.append(
            (
                "packing",
                current_config.model.context_parallel_size,
                current_config.model.tensor_model_parallel_size,
                current_config.dataset.seq_length,
            )
        )
        return current_config

    handles.recipe_runner.apply_cli_overrides.side_effect = apply_trailing
    handles.recipe_runner.sync_finetuning_cp_invariants.side_effect = sync_cp_invariants
    handles.recipe_runner.sync_offline_packing_alignment.side_effect = finalize

    module.main(
        [
            "--recipe",
            "gpt_oss_20b_sft_config",
            "--mode",
            "sft",
            "--dataset",
            "openmathinstruct2-thinking",
            "dataset.enable_offline_packing=true",
            "model.context_parallel_size=2",
            "model.tensor_model_parallel_size=3",
            "dataset.seq_length=2048",
        ]
    )

    handles.recipe_runner.apply_cli_overrides.assert_called_once_with(
        config,
        [
            "dataset.enable_offline_packing=true",
            "model.context_parallel_size=2",
            "model.tensor_model_parallel_size=3",
            "dataset.seq_length=2048",
        ],
    )
    assert events == ["trailing", "cp-invariants", ("packing", 2, 3, 2048)]


def test_dataset_options_are_forwarded_without_launcher_specific_validation():
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--recipe",
            "qwen3_vl_8b_sft_config",
            "--mode",
            "sft",
            "--dataset",
            "medpix",
            "--step-func",
            "vlm_step",
            "dataset.enable_offline_packing=true",
        ]
    )

    handles.recipe_runner.apply_cli_overrides.assert_called_once_with(config, ["dataset.enable_offline_packing=true"])


def test_nested_dataset_config_overrides_are_forwarded_directly():
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--recipe",
            "qwen3_vl_8b_sft_config",
            "--mode",
            "sft",
            "--dataset",
            "local-vlm",
            "--step-func",
            "vlm_step",
            "dataset.source.load_kwargs.data_files.train=/data/vlm.jsonl",
            "dataset.hf_processor_path=Qwen/Qwen3-VL-8B-Instruct",
        ]
    )

    handles.recipe_runner.apply_cli_overrides.assert_called_once_with(
        config,
        [
            "dataset.source.load_kwargs.data_files.train=/data/vlm.jsonl",
            "dataset.hf_processor_path=Qwen/Qwen3-VL-8B-Instruct",
        ],
    )


@pytest.mark.parametrize(
    "dataset_name",
    [
        "llm-pretrain",
        "llm-pretrain-mock",
        "llm-finetune",
        "llm-finetune-preloaded",
        "vlm-energon",
        "vlm-hf",
        "vlm-preloaded",
        "squad-packed",
        "preloaded-vlm",
        "dclm",
        "rp2",
        "c4",
    ],
)
def test_legacy_dataset_names_are_rejected(dataset_name):
    module, _ = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--model", "gpt_oss_20b", "--mode", "sft", "--dataset", dataset_name])


def test_hf_vlm_dataset_name_is_forwarded():
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--recipe",
            "qwen3_vl_8b_sft_config",
            "--mode",
            "sft",
            "--dataset",
            "medpix",
            "--step-func",
            "vlm_step",
        ]
    )

    handles.build_dataset_config.assert_called_once_with(config, "medpix")


def test_forward_step_loading_remains_case_insensitive_with_dataset_presets():
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--recipe",
            "qwen3_vl_8b_sft_config",
            "--mode",
            "sft",
            "--dataset",
            "medpix",
            "--step-func",
            "VLM_STEP",
        ]
    )

    handles.recipe_runner.load_forward_step.assert_called_once_with("VLM_STEP", mode="finetune")


def test_indexed_dataset_path_is_applied_as_config_override(tmp_path):
    module, handles = _load_module()
    prefix = tmp_path / "text_document"
    prefix.with_suffix(".bin").touch()
    prefix.with_suffix(".idx").touch()
    config = SimpleNamespace(dataset=SimpleNamespace())
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--recipe",
            "gpt_oss_20b_pretrain_config",
            "--mode",
            "pretrain",
            "--dataset",
            "megatron-indexed",
            f"dataset.data_path={prefix}",
        ]
    )

    handles.recipe_runner.apply_cli_overrides.assert_called_once_with(config, [f"dataset.data_path={prefix}"])


def test_config_container_overrides_are_forwarded_directly():
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--model",
            "gpt_oss_20b",
            "--mode",
            "pretrain",
            "train.train_iters=3",
            "train.global_batch_size=8",
            "train.micro_batch_size=1",
        ]
    )

    handles.recipe_runner.apply_cli_overrides.assert_called_once_with(
        config,
        ["train.train_iters=3", "train.global_batch_size=8", "train.micro_batch_size=1"],
    )
    handles.recipe_runner.apply_runtime_environment.assert_called_once_with(config)
