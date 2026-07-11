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
        "apply_launcher_overrides",
        "load_forward_step",
        "load_recipe",
        "run_config",
        "sync_model_dataset_sequence_length",
    ):
        setattr(recipe_runner, name, Mock(name=name))
    recipe_runner.apply_cli_overrides.side_effect = lambda config, _: config
    recipe_runner.apply_determinism.side_effect = lambda config, **_: config
    recipe_runner.apply_launcher_overrides.side_effect = lambda config, *_args, **_kwargs: config
    recipe_runner.sync_model_dataset_sequence_length.side_effect = lambda config: config
    recipe_runner.load_forward_step.return_value = object()

    dataset_utils = types.ModuleType("megatron.bridge.recipes.utils.dataset_utils")
    dataset_utils.apply_public_dataset_override = Mock(side_effect=lambda config, **_kwargs: config)
    stub_modules = {
        "recipe_runner": recipe_runner,
        "megatron": _package("megatron"),
        "megatron.bridge": _package("megatron.bridge"),
        "megatron.bridge.recipes": _package("megatron.bridge.recipes"),
        "megatron.bridge.recipes.utils": _package("megatron.bridge.recipes.utils"),
        "megatron.bridge.recipes.utils.dataset_utils": dataset_utils,
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
        apply_public_dataset_override=dataset_utils.apply_public_dataset_override,
        recipe_runner=recipe_runner,
    )


@pytest.mark.parametrize(
    "option",
    ["--task", "--source", "--family", "--domain", "--data", "--use-recipes", "--step", "--set"],
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


@pytest.mark.parametrize(
    "selection",
    [["--model", "gpt_oss_20b"], ["--recipe", "gpt_oss_20b_pretrain_config"]],
)
def test_every_selection_requires_explicit_mode(selection):
    module, _ = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(selection)


@pytest.mark.parametrize("mode", ["lora", "dora"])
def test_model_adapter_modes_select_peft_recipe_and_scheme(mode):
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    module.main(["--model", "gpt_oss_20b", "--mode", mode])

    handles.recipe_runner.load_recipe.assert_called_once_with(
        "gpt_oss_20b_peft_config",
        peft_scheme=mode,
        seq_length=None,
        hf_path=None,
        source="recipes",
    )
    assert handles.recipe_runner.run_config.call_args.kwargs["mode"] == "finetune"


def test_full_recipe_uses_only_library_namespace_and_default_llm_step():
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()

    module.main(["--recipe", "gpt_oss_20b_pretrain_config", "--mode", "pretrain"])

    handles.recipe_runner.load_recipe.assert_called_once_with(
        "gpt_oss_20b_pretrain_config",
        peft_scheme=None,
        seq_length=None,
        hf_path=None,
        source="recipes",
    )
    handles.recipe_runner.load_forward_step.assert_called_once_with("llm_step", mode="pretrain")


def test_full_recipe_rejects_incompatible_mode():
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()

    with pytest.raises(ValueError, match="incompatible with recipe"):
        module.main(["--recipe", "gpt_oss_20b_sft_config", "--mode", "lora"])


def test_only_public_dataset_names_are_accepted():
    module, _ = _load_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--recipe", "gpt_oss_20b_sft_config", "--mode", "sft", "--dataset", "llm-finetune"])


def test_named_finetuning_dataset_maps_to_internal_config():
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    module.main(["--recipe", "gpt_oss_20b_sft_config", "--mode", "sft", "--dataset", "squad"])

    handles.apply_public_dataset_override.assert_called_once_with(
        config,
        dataset_name="squad",
        packed_sequence=False,
        pad_seq_to_mult=1,
        seq_length=None,
        cli_overrides=[],
    )


@pytest.mark.parametrize(
    ("public_name", "overrides"),
    [
        ("local-jsonl", ["dataset.dataset_root=/data/sft"]),
        (
            "preloaded-vlm",
            [
                "dataset.train_data_path=/data/vlm.jsonl",
                "dataset.image_folder=/data/images",
                "dataset.hf_processor_path=Qwen/Qwen3-VL-8B-Instruct",
            ],
        ),
    ],
)
def test_local_dataset_names_replace_the_dataset_provider(public_name, overrides):
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    module.main(["--recipe", "gpt_oss_20b_sft_config", "--mode", "sft", "--dataset", public_name, *overrides])

    handles.apply_public_dataset_override.assert_called_once_with(
        config,
        dataset_name=public_name,
        packed_sequence=False,
        pad_seq_to_mult=1,
        seq_length=None,
        cli_overrides=overrides,
    )


@pytest.mark.parametrize(
    ("mode", "dataset"),
    [("pretrain", "squad"), ("sft", "dclm")],
)
def test_dataset_mode_mismatch_is_rejected(mode, dataset):
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()

    with pytest.raises(ValueError, match="incompatible with dataset"):
        module.main(["--model", "gpt_oss_20b", "--mode", mode, "--dataset", dataset])


@pytest.mark.parametrize(
    ("option", "attribute"),
    [
        ("--step-func", "step_func"),
        ("--step_func", "step_func"),
        ("--dataset-path", "dataset_paths"),
        ("--dataset_path", "dataset_paths"),
        ("--save-config", "save_config_filepath"),
        ("--save_config", "save_config_filepath"),
    ],
)
def test_public_options_accept_only_hyphen_and_underscore_spellings(option, attribute):
    module, _ = _load_module()

    args, _ = module.parse_args(["--model", "gpt_oss_20b", "--mode", "pretrain", option, "value"])

    expected = ["value"] if attribute == "dataset_paths" else "value"
    assert getattr(args, attribute) == expected


def test_thinking_dataset_enables_packing_and_cp_padding():
    module, handles = _load_module()
    config = SimpleNamespace(model=SimpleNamespace(seq_length=1024, context_parallel_size=1), dataset=object())
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--recipe",
            "gpt_oss_20b_sft_config",
            "--mode",
            "sft",
            "--dataset",
            "openmathinstruct2-thinking",
            "--cp",
            "2",
        ]
    )

    handles.apply_public_dataset_override.assert_called_once_with(
        config,
        dataset_name="openmathinstruct2-thinking",
        packed_sequence=True,
        pad_seq_to_mult=4,
        seq_length=None,
        cli_overrides=[],
    )


def test_indexed_dataset_paths_are_validated(tmp_path):
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
            "dclm",
            "--dataset-path",
            str(prefix),
        ]
    )

    assert config.dataset.data_path == [str(prefix)]


def test_trailing_overrides_are_applied_after_shortcuts():
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    module.main(
        [
            "--model",
            "gpt_oss_20b",
            "--mode",
            "pretrain",
            "--max-steps",
            "2",
            "train.train_iters=3",
        ]
    )

    handles.recipe_runner.apply_launcher_overrides.assert_called_once()
    handles.recipe_runner.apply_cli_overrides.assert_called_once_with(config, ["train.train_iters=3"])
