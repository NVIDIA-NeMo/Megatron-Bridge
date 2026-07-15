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


def test_sync_model_dataset_sequence_length_uses_canonical_dataset_field(recipe_runner: ModuleType) -> None:
    config = SimpleNamespace(model=SimpleNamespace(seq_length=1024), dataset=SimpleNamespace(seq_length=256))

    assert recipe_runner.sync_model_dataset_sequence_length(config) is config
    assert config.model.seq_length == 256


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


def test_run_config_dryrun_uses_logger_save_path(recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(logger=SimpleNamespace(save_config_filepath="resolved.yaml"))
    saved_paths: list[str] = []
    monkeypatch.setattr(recipe_runner, "save_config", lambda _config, path: saved_paths.append(path))

    recipe_runner.run_config(config=config, mode="pretrain", step_func=object(), dryrun=True)

    assert saved_paths == ["resolved.yaml"]
