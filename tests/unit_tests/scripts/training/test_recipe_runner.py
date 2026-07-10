"""Behavioral tests for the shared training recipe runner."""

from __future__ import annotations

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


@pytest.mark.parametrize(
    ("recipe_name", "expected_mode"),
    [
        ("llama3_8b_pretrain_config", "pretrain"),
        ("llama3_8b_finetune_config", "finetune"),
        ("llama3_8b_sft_config", "finetune"),
        ("llama3_8b_peft_config", "finetune"),
        ("llama3_8b_lora_config", "finetune"),
    ],
)
def test_infer_train_mode(recipe_runner: ModuleType, recipe_name: str, expected_mode: str) -> None:
    assert recipe_runner.infer_train_mode(recipe_name) == expected_mode


@pytest.mark.parametrize("recipe_name", ["custom_config", "pretrain_sft_config"])
def test_infer_train_mode_rejects_ambiguous_names(recipe_runner: ModuleType, recipe_name: str) -> None:
    with pytest.raises(ValueError, match="Unable to infer training mode"):
        recipe_runner.infer_train_mode(recipe_name)


def test_perf_recipe_function_name_normalizes_precision_and_variant(recipe_runner: ModuleType) -> None:
    assert (
        recipe_runner.perf_recipe_function_name(
            model_recipe_name="llama3_8b",
            task="pretrain",
            num_gpus=8,
            gpu="h100",
            precision="fp8_cs",
            config_variant="large_scale",
        )
        == "llama3_8b_pretrain_8gpu_h100_fp8cs_large_scale_config"
    )
    assert (
        recipe_runner.perf_recipe_function_name(
            model_recipe_name="llama3_8b",
            task="pretrain",
            num_gpus=8,
            gpu="h100",
            precision="bf16",
            config_variant="v2",
        )
        == "llama3_8b_pretrain_8gpu_h100_bf16_config"
    )


def test_load_recipe_from_perf_namespace_skips_library_lookup(
    recipe_runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = object()
    library_lookup = Mock(side_effect=AssertionError("library lookup must not run"))
    perf_lookup = Mock(return_value=lambda: config)
    monkeypatch.setattr(recipe_runner, "find_library_recipe", library_lookup)
    monkeypatch.setattr(recipe_runner, "find_perf_recipe", perf_lookup)

    assert recipe_runner.load_recipe("unit_perf_config", source="perf_recipes") is config
    library_lookup.assert_not_called()
    perf_lookup.assert_called_once_with("unit_perf_config")


def test_recipe_builder_type_error_is_not_retried(recipe_runner: ModuleType) -> None:
    builder = Mock(side_effect=TypeError("raised inside recipe"))

    with pytest.raises(TypeError, match="raised inside recipe"):
        recipe_runner._load_with_optional_kwargs(
            builder,
            peft_scheme=None,
            seq_length=128,
        )

    builder.assert_called_once_with(seq_length=128)


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


def test_apply_tokenizer_override_builds_null_tokenizer(recipe_runner: ModuleType) -> None:
    config = SimpleNamespace(tokenizer=object())

    result = recipe_runner.apply_tokenizer_override(
        config,
        tokenizer_type="NullTokenizer",
        tokenizer_model=None,
        vocab_size=32000,
    )

    assert result is config
    assert config.tokenizer.tokenizer_type == "NullTokenizer"
    assert config.tokenizer.vocab_size == 32000


def test_apply_tokenizer_override_requires_model_path(recipe_runner: ModuleType) -> None:
    config = SimpleNamespace(tokenizer=object())

    with pytest.raises(ValueError, match="--tokenizer-model is required"):
        recipe_runner.apply_tokenizer_override(
            config,
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=None,
            vocab_size=32000,
        )


def test_sync_model_dataset_sequence_length_supports_both_dataset_field_names(recipe_runner: ModuleType) -> None:
    for dataset in (SimpleNamespace(seq_length=256), SimpleNamespace(sequence_length=512)):
        config = SimpleNamespace(model=SimpleNamespace(seq_length=1024), dataset=dataset)

        assert recipe_runner.sync_model_dataset_sequence_length(config) is config
        assert config.model.seq_length in {256, 512}


def test_run_config_dryrun_applies_environment_and_validates_target_topology(
    recipe_runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = object()
    events: list[str] = []
    train = Mock(name="train")
    monkeypatch.setattr(recipe_runner, "apply_environment_variables", lambda _: events.append("environment"))
    monkeypatch.setattr(recipe_runner, "runtime_config_update", lambda _: events.append("runtime"))
    monkeypatch.setattr(recipe_runner, "save_config", lambda *_: events.append("save"))
    monkeypatch.setitem(recipe_runner.TRAIN_FUNCTIONS, "pretrain", train)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "7")

    returned_early = recipe_runner.run_config(
        config=config,
        mode="pretrain",
        step_func=object(),
        dryrun=True,
        save_config_filepath="config.yaml",
        dryrun_num_gpus=8,
    )

    assert returned_early is True
    assert events == ["environment", "runtime", "save"]
    assert recipe_runner.os.environ["WORLD_SIZE"] == "8"
    assert recipe_runner.os.environ["RANK"] == "0"
    train.assert_not_called()
