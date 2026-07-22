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

"""Behavioral tests for the shared training recipe runner."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.unit_tests.training.test_run_recipe_qwen3_omni import _load_recipe_runner_module


pytestmark = pytest.mark.unit


@pytest.fixture
def recipe_runner() -> ModuleType:
    """Load the real runner source with lightweight dependency stubs."""
    module, _ = _load_recipe_runner_module()
    return module


def test_recipe_builder_type_error_is_not_retried(recipe_runner: ModuleType) -> None:
    builder = Mock(side_effect=TypeError("raised inside recipe"))

    with pytest.raises(TypeError, match="raised inside recipe"):
        recipe_runner._load_with_optional_kwargs(
            builder,
            peft_scheme=None,
            seq_length=128,
        )

    builder.assert_called_once_with(seq_length=128)


def test_recipe_without_configurable_scheme_rejects_dora(recipe_runner: ModuleType) -> None:
    def lora_only_recipe() -> object:
        return object()

    with pytest.raises(ValueError, match="--mode dora is unsupported"):
        recipe_runner._load_with_optional_kwargs(lora_only_recipe, peft_scheme="dora")


def test_recipe_without_configurable_scheme_allows_default_lora(recipe_runner: ModuleType) -> None:
    config = object()

    def lora_only_recipe() -> object:
        return config

    assert recipe_runner._load_with_optional_kwargs(lora_only_recipe, peft_scheme="lora") is config


def test_load_recipe_falls_back_to_library_when_name_is_not_benchmark(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = object()

    def library_recipe() -> object:
        return config

    library_finder = Mock(return_value=library_recipe)
    benchmark_finder = Mock()
    monkeypatch.setattr(recipe_runner, "resolved_benchmark_recipe_metadata", lambda _name: None)
    monkeypatch.setattr(recipe_runner, "find_library_recipe", library_finder)
    monkeypatch.setattr(recipe_runner, "find_benchmark_recipe", benchmark_finder)

    assert recipe_runner.load_recipe("shared_recipe_config") is config
    library_finder.assert_called_once_with("shared_recipe_config")
    benchmark_finder.assert_not_called()


def test_load_recipe_auto_detects_exact_benchmark_export(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = object()

    def benchmark_recipe() -> object:
        return config

    library_finder = Mock()
    benchmark_finder = Mock(return_value=benchmark_recipe)
    monkeypatch.setattr(recipe_runner, "resolved_benchmark_recipe_metadata", lambda _name: object())
    monkeypatch.setattr(recipe_runner, "find_library_recipe", library_finder)
    monkeypatch.setattr(recipe_runner, "find_benchmark_recipe", benchmark_finder)

    assert recipe_runner.load_recipe("shared_recipe_config") is config
    benchmark_finder.assert_called_once_with("shared_recipe_config")
    library_finder.assert_not_called()


def test_load_recipe_reports_both_packages_when_missing(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recipe_runner, "resolved_benchmark_recipe_metadata", lambda _name: None)
    monkeypatch.setattr(recipe_runner, "find_library_recipe", lambda _name: None)

    with pytest.raises(AttributeError, match=r"megatron\.bridge\.recipes or megatron\.bridge\.perf_recipes"):
        recipe_runner.load_recipe("missing_recipe_config")


def test_duplicate_name_warns_and_selects_benchmark(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recipe_name = "qwen3_235b_a22b_pretrain_256gpu_h100_bf16_config"
    config = object()

    monkeypatch.setattr(recipe_runner, "resolved_benchmark_recipe_metadata", lambda _name: object())
    benchmark_finder = Mock(return_value=lambda: config)
    library_finder = Mock()
    monkeypatch.setattr(recipe_runner, "find_benchmark_recipe", benchmark_finder)
    monkeypatch.setattr(recipe_runner, "find_library_recipe", library_finder)

    assert recipe_runner.load_recipe(recipe_name) is config
    assert "selecting the benchmark definition" in caplog.text
    benchmark_finder.assert_called_once_with(recipe_name)
    library_finder.assert_not_called()


@pytest.mark.parametrize(
    "recipe_name",
    [
        "llama3_70b_sft_32gpu_h100_bf16_config",
        "llama3_70b_peft_8gpu_h100_bf16_config",
    ],
)
def test_finetuning_collision_selects_benchmark(
    recipe_runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    recipe_name: str,
) -> None:
    config = object()

    monkeypatch.setattr(recipe_runner, "resolved_benchmark_recipe_metadata", lambda _name: object())
    library_finder = Mock()
    benchmark_finder = Mock(return_value=lambda: config)
    monkeypatch.setattr(recipe_runner, "find_library_recipe", library_finder)
    monkeypatch.setattr(recipe_runner, "find_benchmark_recipe", benchmark_finder)

    assert recipe_runner.load_recipe(recipe_name) is config
    assert "selecting the benchmark definition" in caplog.text
    benchmark_finder.assert_called_once_with(recipe_name)
    library_finder.assert_not_called()


def test_find_benchmark_recipe_imports_exporting_family_lazily(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_builder = Mock(name="benchmark_recipe")
    family = SimpleNamespace(target_benchmark_recipe_config=config_builder)
    imported_modules: list[str] = []

    def import_module(module_name: str) -> SimpleNamespace:
        imported_modules.append(module_name)
        return family

    monkeypatch.setattr(recipe_runner, "benchmark_recipe_family", lambda _name: "qwen")
    monkeypatch.setattr(recipe_runner.importlib, "import_module", import_module)

    assert recipe_runner.find_benchmark_recipe("target_benchmark_recipe_config") is config_builder
    assert imported_modules == ["megatron.bridge.perf_recipes.qwen"]


def test_find_benchmark_recipe_does_not_import_unregistered_family(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_module = Mock()
    monkeypatch.setattr(recipe_runner, "benchmark_recipe_family", Mock(side_effect=ValueError("unknown family")))
    monkeypatch.setattr(recipe_runner.importlib, "import_module", import_module)

    assert recipe_runner.find_benchmark_recipe("unknown_recipe_config") is None
    import_module.assert_not_called()


def test_load_forward_step_imports_only_selected_module(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = Mock(name="unit_forward_step")
    imported_modules: list[str] = []

    def import_module(module_name: str) -> SimpleNamespace:
        imported_modules.append(module_name)
        return SimpleNamespace(forward_step=step)

    recipe_runner._load_step_function.cache_clear()
    monkeypatch.setitem(recipe_runner.STEP_FUNCTIONS, "unit_step", ("unit.step", "forward_step"))
    monkeypatch.setattr(recipe_runner.importlib, "import_module", import_module)

    assert recipe_runner.load_forward_step("unit_step") is step
    assert imported_modules == ["unit.step"]
    recipe_runner._load_step_function.cache_clear()


def test_qwen_vl_registry_loads_its_specialized_forward_step(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = Mock(name="qwen3_vl_forward_step")
    import_module = Mock(return_value=SimpleNamespace(forward_step=step))
    monkeypatch.setattr(recipe_runner.importlib, "import_module", import_module)

    recipe_runner._load_step_function.cache_clear()
    assert recipe_runner.load_forward_step("qwen3_vl_step", mode="pretrain") is step
    import_module.assert_called_once_with("megatron.bridge.models.qwen_vl.qwen3_vl_step")
    recipe_runner._load_step_function.cache_clear()


def test_wan_registry_constructs_forward_step_with_recipe_mode(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WanForwardStep:
        def __init__(self, mode: str | None = None) -> None:
            self.mode = mode

    import_module = Mock(return_value=SimpleNamespace(WanForwardStep=WanForwardStep))
    monkeypatch.setattr(recipe_runner.importlib, "import_module", import_module)

    recipe_runner._load_step_function.cache_clear()
    step = recipe_runner.load_forward_step("wan_step", mode="pretrain")
    assert isinstance(step, WanForwardStep)
    assert step.mode == "pretrain"
    import_module.assert_called_once_with("megatron.bridge.diffusion.models.wan.wan_step")
    recipe_runner._load_step_function.cache_clear()


def test_sync_model_dataset_sequence_length_uses_canonical_dataset_field(recipe_runner: ModuleType) -> None:
    config = SimpleNamespace(model=SimpleNamespace(seq_length=1024), dataset=SimpleNamespace(seq_length=256))

    assert recipe_runner.sync_model_dataset_sequence_length(config) is config
    assert config.model.seq_length == 256


def test_sync_model_pipeline_layout_uses_overridden_topology(recipe_runner: ModuleType) -> None:
    layout = [["first"], ["middle"], ["middle"], ["last"]]
    layout_builder = Mock(return_value=layout)
    config = SimpleNamespace(
        model=SimpleNamespace(
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=1,
            pipeline_model_parallel_layout=[["stale"]] * 16,
            _pipeline_model_parallel_layout_builder=layout_builder,
        )
    )

    assert (
        recipe_runner.sync_model_pipeline_layout(
            config,
            cli_overrides=[
                "model.pipeline_model_parallel_size=4",
                "model.virtual_pipeline_model_parallel_size=1",
            ],
        )
        is config
    )
    assert config.model.pipeline_model_parallel_layout == layout
    layout_builder.assert_called_once_with(4, 1)


def test_sync_model_pipeline_layout_preserves_explicit_layout_override(recipe_runner: ModuleType) -> None:
    layout = [["custom"]]
    layout_builder = Mock()
    config = SimpleNamespace(
        model=SimpleNamespace(
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=1,
            pipeline_model_parallel_layout=layout,
            _pipeline_model_parallel_layout_builder=layout_builder,
        )
    )

    recipe_runner.sync_model_pipeline_layout(
        config,
        cli_overrides=[
            "model.pipeline_model_parallel_size=1",
            "model.pipeline_model_parallel_layout=[[custom]]",
        ],
    )

    assert config.model.pipeline_model_parallel_layout == layout
    layout_builder.assert_not_called()


@pytest.mark.parametrize(
    "dataset",
    [
        SimpleNamespace(enable_offline_packing=False),
        SimpleNamespace(enable_in_batch_packing=False),
    ],
)
def test_sync_finetuning_cp_invariants_supports_unpacked_text_and_vlm(
    recipe_runner: ModuleType, dataset: SimpleNamespace
) -> None:
    config = SimpleNamespace(
        dataset=dataset,
        model=SimpleNamespace(context_parallel_size=2, calculate_per_token_loss=False),
        dist=SimpleNamespace(eval_context_parallel_size=None),
        ddp=SimpleNamespace(average_in_collective=True),
    )

    assert recipe_runner.sync_finetuning_cp_invariants(config, mode="finetune") is config
    assert config.model.calculate_per_token_loss is True
    assert config.ddp.average_in_collective is False


def test_sync_finetuning_cp_invariants_does_not_change_pretraining(recipe_runner: ModuleType) -> None:
    config = SimpleNamespace(
        dataset=SimpleNamespace(enable_in_batch_packing=False),
        model=SimpleNamespace(context_parallel_size=2, calculate_per_token_loss=False),
        dist=SimpleNamespace(eval_context_parallel_size=None),
        ddp=SimpleNamespace(average_in_collective=True),
    )

    assert recipe_runner.sync_finetuning_cp_invariants(config, mode="pretrain") is config
    assert config.model.calculate_per_token_loss is False
    assert config.ddp.average_in_collective is True


def test_sync_offline_packing_alignment_uses_final_train_and_eval_topology(recipe_runner: ModuleType) -> None:
    packing_specs = SimpleNamespace(packed_sequence_size=4096, pad_seq_to_mult=1)
    config = SimpleNamespace(
        dataset=SimpleNamespace(
            enable_offline_packing=True,
            offline_packing_specs=packing_specs,
            seq_length=2048,
        ),
        model=SimpleNamespace(
            context_parallel_size=2,
            tensor_model_parallel_size=3,
            sequence_parallel=True,
            calculate_per_token_loss=False,
        ),
        dist=SimpleNamespace(eval_context_parallel_size=4),
        ddp=SimpleNamespace(average_in_collective=True),
    )

    assert recipe_runner.sync_offline_packing_alignment(config) is config
    assert packing_specs.packed_sequence_size == 2048
    assert packing_specs.pad_seq_to_mult == 24


def test_sync_offline_packing_alignment_preserves_stricter_user_multiple(recipe_runner: ModuleType) -> None:
    packing_specs = SimpleNamespace(packed_sequence_size=1024, pad_seq_to_mult=5)
    config = SimpleNamespace(
        dataset=SimpleNamespace(enable_offline_packing=True, offline_packing_specs=packing_specs),
        model=SimpleNamespace(
            context_parallel_size=1,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        ),
        dist=SimpleNamespace(eval_context_parallel_size=None),
    )

    recipe_runner.sync_offline_packing_alignment(config)

    assert packing_specs.pad_seq_to_mult == 10


@pytest.mark.parametrize("packing_specs", [None, {"pad_seq_to_mult": 3}])
def test_sync_offline_packing_alignment_materializes_direct_config_overrides(
    recipe_runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    packing_specs: dict[str, int] | None,
) -> None:
    class PackedSequenceSpecs:
        def __init__(self, *, pad_seq_to_mult: int = 1) -> None:
            self.packed_sequence_size = -1
            self.pad_seq_to_mult = pad_seq_to_mult

    packing_module = ModuleType("megatron.bridge.data.packing")
    packing_module.PackedSequenceSpecs = PackedSequenceSpecs
    monkeypatch.setitem(sys.modules, "megatron.bridge.data.packing", packing_module)
    config = SimpleNamespace(
        dataset=SimpleNamespace(
            enable_offline_packing=True,
            offline_packing_specs=packing_specs,
            seq_length=2048,
        ),
        model=SimpleNamespace(
            context_parallel_size=1,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
        ),
        dist=SimpleNamespace(eval_context_parallel_size=None),
    )

    recipe_runner.sync_offline_packing_alignment(config)

    assert isinstance(config.dataset.offline_packing_specs, PackedSequenceSpecs)
    assert config.dataset.offline_packing_specs.packed_sequence_size == 2048
    assert config.dataset.offline_packing_specs.pad_seq_to_mult == (3 if packing_specs else 1)


def test_apply_runtime_environment_uses_resolved_nccl_ub_config(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(ddp=SimpleNamespace(nccl_ub=True))
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    monkeypatch.delenv("NCCL_CTA_POLICY", raising=False)

    assert recipe_runner.apply_runtime_environment(config) is config
    assert recipe_runner.os.environ["NCCL_NVLS_ENABLE"] == "1"
    assert recipe_runner.os.environ["NCCL_CTA_POLICY"] == "1"


def test_apply_runtime_environment_preserves_explicit_process_values(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(ddp=SimpleNamespace(nccl_ub=True))
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")
    monkeypatch.setenv("NCCL_CTA_POLICY", "explicit")

    recipe_runner.apply_runtime_environment(config)

    assert recipe_runner.os.environ["NCCL_NVLS_ENABLE"] == "0"
    assert recipe_runner.os.environ["NCCL_CTA_POLICY"] == "explicit"


def test_determinism_defers_process_environment_until_after_cli_overrides(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(env_vars={})
    determinism_module = ModuleType("megatron.bridge.recipes.utils.determinism_utils")

    def apply_determinism_overrides(current_config: SimpleNamespace) -> None:
        current_config.env_vars["NCCL_ALGO"] = "Ring"

    determinism_module.apply_determinism_overrides = apply_determinism_overrides
    monkeypatch.setitem(sys.modules, "megatron.bridge.recipes.utils.determinism_utils", determinism_module)
    monkeypatch.delenv("NCCL_ALGO", raising=False)

    recipe_runner.apply_determinism(config, deterministic=True)

    assert config.env_vars["NCCL_ALGO"] == "Ring"
    assert "NCCL_ALGO" not in recipe_runner.os.environ


def test_benchmark_bootstrap_applies_environment_before_self_exec(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = object()
    events: list[str] = []
    exec_calls: list[tuple[str, list[str], dict[str, str]]] = []
    monkeypatch.delenv(recipe_runner.RECIPE_ENV_BOOTSTRAP_MARKER, raising=False)
    monkeypatch.delenv("RECIPE_DEFAULT", raising=False)

    def apply_environment(current_config: object) -> object:
        events.append("environment")
        recipe_runner.os.environ.setdefault("RECIPE_DEFAULT", "enabled")
        return current_config

    def execvpe(executable: str, command: list[str], environment: dict[str, str]) -> None:
        events.append("exec")
        exec_calls.append((executable, command, environment))

    monkeypatch.setattr(recipe_runner, "apply_runtime_environment", apply_environment)
    monkeypatch.setattr(recipe_runner.os, "execvpe", execvpe)

    with pytest.raises(RuntimeError, match="returned unexpectedly"):
        recipe_runner.bootstrap_recipe_environment(
            config,
            script_path="/repo/scripts/training/run_recipe.py",
            argv=["--recipe", "benchmark_recipe_config"],
        )

    assert events == ["environment", "exec"]
    executable, command, environment = exec_calls[0]
    assert executable == recipe_runner.sys.executable
    assert command == [
        recipe_runner.sys.executable,
        "/repo/scripts/training/run_recipe.py",
        "--recipe",
        "benchmark_recipe_config",
    ]
    assert environment["RECIPE_DEFAULT"] == "enabled"
    assert environment[recipe_runner.RECIPE_ENV_BOOTSTRAP_MARKER] == str(recipe_runner.os.getpid())


def test_benchmark_bootstrap_marker_skips_second_exec(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = object()
    apply_environment = Mock(return_value=config)
    execvpe = Mock()
    monkeypatch.setenv(recipe_runner.RECIPE_ENV_BOOTSTRAP_MARKER, str(recipe_runner.os.getpid()))
    monkeypatch.setattr(recipe_runner, "apply_runtime_environment", apply_environment)
    monkeypatch.setattr(recipe_runner.os, "execvpe", execvpe)

    assert (
        recipe_runner.bootstrap_recipe_environment(
            config,
            script_path="run_recipe.py",
            argv=[],
        )
        is config
    )
    apply_environment.assert_called_once_with(config)
    execvpe.assert_not_called()


def test_training_stack_is_registered_for_lazy_import(recipe_runner: ModuleType) -> None:
    assert "torch" not in recipe_runner.__dict__
    assert recipe_runner.TRAIN_FUNCTIONS == {
        "pretrain": ("megatron.bridge.training.pretrain", "pretrain"),
        "finetune": ("megatron.bridge.training.finetune", "finetune"),
    }


def test_run_config_dryrun_saves_without_training(recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    config = object()
    events: list[str] = []
    train = Mock(name="train")
    monkeypatch.setattr(recipe_runner, "save_config", lambda *_: events.append("save"))
    monkeypatch.setitem(recipe_runner.TRAIN_FUNCTIONS, "pretrain", train)

    returned_early = recipe_runner.run_config(
        config=config,
        mode="pretrain",
        step_func=object(),
        dryrun=True,
        save_config_filepath="config.yaml",
    )

    assert returned_early is True
    assert events == ["save"]
    train.assert_not_called()


def test_run_config_dryrun_validates_requested_world_size(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(logger=SimpleNamespace(save_config_filepath="resolved.yaml"))
    runtime_updates: list[tuple[str | None, str | None, object]] = []
    config_module = ModuleType("megatron.bridge.training.config")
    config_module.runtime_config_update = lambda current_config: runtime_updates.append(
        (
            recipe_runner.os.environ.get("WORLD_SIZE"),
            recipe_runner.os.environ.get("RANK"),
            current_config,
        )
    )
    monkeypatch.setitem(sys.modules, "megatron.bridge.training.config", config_module)
    monkeypatch.setattr(recipe_runner, "save_config", lambda *_args: None)
    monkeypatch.setenv("WORLD_SIZE", "existing-world-size")
    monkeypatch.delenv("RANK", raising=False)

    recipe_runner.run_config(
        config=config,
        mode="pretrain",
        step_func=object(),
        dryrun=True,
        dryrun_world_size=16,
    )

    assert runtime_updates == [("16", "0", config)]
    assert recipe_runner.os.environ["WORLD_SIZE"] == "existing-world-size"
    assert "RANK" not in recipe_runner.os.environ


def test_run_config_dryrun_restores_environment_when_runtime_update_fails(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_runtime_update(_config: object) -> None:
        raise RuntimeError("runtime update failed")

    config_module = ModuleType("megatron.bridge.training.config")
    config_module.runtime_config_update = fail_runtime_update
    monkeypatch.setitem(sys.modules, "megatron.bridge.training.config", config_module)
    monkeypatch.setenv("WORLD_SIZE", "existing-world-size")
    monkeypatch.delenv("RANK", raising=False)

    with pytest.raises(RuntimeError, match="runtime update failed"):
        recipe_runner.run_config(
            config=object(),
            mode="pretrain",
            step_func=object(),
            dryrun=True,
            dryrun_world_size=16,
        )

    assert recipe_runner.os.environ["WORLD_SIZE"] == "existing-world-size"
    assert "RANK" not in recipe_runner.os.environ


def test_run_config_dryrun_uses_logger_save_path(recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(logger=SimpleNamespace(save_config_filepath="resolved.yaml"))
    saved_paths: list[str] = []
    monkeypatch.setattr(recipe_runner, "save_config", lambda _config, path: saved_paths.append(path))

    recipe_runner.run_config(config=config, mode="pretrain", step_func=object(), dryrun=True)

    assert saved_paths == ["resolved.yaml"]
