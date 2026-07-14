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
    recipe_runner.STEP_MODALITIES = {
        "audio_lm_step": "audio",
        "gpt_step": "text",
        "llm_step": "text",
        "vlm_step": "vlm",
        "qwen3_vl_step": "vlm",
    }
    for name in (
        "apply_cli_overrides",
        "apply_determinism",
        "apply_launcher_overrides",
        "load_forward_step",
        "load_recipe",
        "run_config",
        "sync_finetuning_cp_invariants",
        "sync_offline_packing_alignment",
        "sync_model_dataset_sequence_length",
    ):
        setattr(recipe_runner, name, Mock(name=name))
    recipe_runner.apply_cli_overrides.side_effect = lambda config, _: config
    recipe_runner.apply_determinism.side_effect = lambda config, **_: config
    recipe_runner.apply_launcher_overrides.side_effect = lambda config, *_args, **_kwargs: config
    recipe_runner.sync_finetuning_cp_invariants.side_effect = lambda config, **_: config
    recipe_runner.sync_offline_packing_alignment.side_effect = lambda config: config
    recipe_runner.sync_model_dataset_sequence_length.side_effect = lambda config: config
    recipe_runner.load_forward_step.return_value = object()

    dataset_utils = types.ModuleType("megatron.bridge.recipes.utils.dataset_utils")
    dataset_utils.PUBLIC_DATASETS = {
        "mock": SimpleNamespace(
            train_mode="pretrain", modality="text", supports_offline_packing=False, indexed_data=False
        ),
        "megatron-indexed": SimpleNamespace(
            train_mode="pretrain", modality="text", supports_offline_packing=False, indexed_data=True
        ),
        "squad": SimpleNamespace(
            train_mode="finetune", modality="text", supports_offline_packing=True, indexed_data=False
        ),
        "openmathinstruct2": SimpleNamespace(
            train_mode="finetune", modality="text", supports_offline_packing=True, indexed_data=False
        ),
        "openmathinstruct2-thinking": SimpleNamespace(
            train_mode="finetune", modality="text", supports_offline_packing=True, indexed_data=False
        ),
        "gsm8k": SimpleNamespace(
            train_mode="finetune", modality="text", supports_offline_packing=True, indexed_data=False
        ),
        "local-jsonl": SimpleNamespace(
            train_mode="finetune", modality="text", supports_offline_packing=True, indexed_data=False
        ),
        "local-vlm": SimpleNamespace(
            train_mode="finetune", modality="vlm", supports_offline_packing=False, indexed_data=False
        ),
        **{
            name: SimpleNamespace(
                train_mode="finetune", modality="vlm", supports_offline_packing=False, indexed_data=False
            )
            for name in ("cord-v2", "llava-video-178k", "medpix", "raven", "rdr")
        },
    }
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


def test_conventional_recipe_name_infers_mode():
    module, _ = _load_module()

    args, _ = module.parse_args(["--recipe", "gpt_oss_20b_pretrain_config"])

    assert args.mode == "pretrain"


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
    )
    handles.recipe_runner.load_forward_step.assert_called_once_with("llm_step", mode="pretrain")


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

    handles.apply_public_dataset_override.assert_called_once_with(
        config,
        dataset_name="squad",
        enable_offline_packing=False,
        seq_length=None,
        dataset_root=None,
        train_data_path=None,
        validation_data_path=None,
        test_data_path=None,
        media_root=None,
        hf_processor_path=None,
    )


@pytest.mark.parametrize(
    ("public_name", "options", "expected"),
    [
        ("local-jsonl", ["--dataset-root", "/data/sft"], {"dataset_root": "/data/sft"}),
        (
            "local-vlm",
            [
                "--train-data-path",
                "/data/vlm.jsonl",
                "--hf-processor-path",
                "Qwen/Qwen3-VL-8B-Instruct",
            ],
            {
                "train_data_path": "/data/vlm.jsonl",
                "hf_processor_path": "Qwen/Qwen3-VL-8B-Instruct",
            },
        ),
    ],
)
def test_local_dataset_names_replace_the_dataset_provider(public_name, options, expected):
    module, handles = _load_module()
    config = SimpleNamespace()
    handles.recipe_runner.load_recipe.return_value = config

    step_options = ["--step-func", "vlm_step"] if public_name == "local-vlm" else []
    module.main(
        ["--recipe", "gpt_oss_20b_sft_config", "--mode", "sft", "--dataset", public_name, *step_options, *options]
    )

    handles.apply_public_dataset_override.assert_called_once_with(
        config,
        dataset_name=public_name,
        enable_offline_packing=False,
        seq_length=None,
        dataset_root=expected.get("dataset_root"),
        train_data_path=expected.get("train_data_path"),
        validation_data_path=None,
        test_data_path=None,
        media_root=expected.get("media_root"),
        hf_processor_path=expected.get("hf_processor_path"),
    )


@pytest.mark.parametrize(
    ("mode", "dataset"),
    [("pretrain", "squad"), ("sft", "megatron-indexed")],
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
        ("--dataset-root", "dataset_root"),
        ("--dataset_root", "dataset_root"),
        ("--train-data-path", "train_data_path"),
        ("--train_data_path", "train_data_path"),
        ("--save-config", "save_config_filepath"),
        ("--save_config", "save_config_filepath"),
    ],
)
def test_public_options_accept_only_hyphen_and_underscore_spellings(option, attribute):
    module, _ = _load_module()

    args, _ = module.parse_args(["--model", "gpt_oss_20b", "--mode", "pretrain", option, "value"])

    expected = ["value"] if attribute == "dataset_paths" else "value"
    assert getattr(args, attribute) == expected


@pytest.mark.parametrize("option", ["--offline-packing", "--offline_packing"])
def test_offline_packing_spellings_enable_offline_packing(option):
    module, _ = _load_module()

    args, _ = module.parse_args(["--model", "gpt_oss_20b", "--mode", "sft", option])

    assert args.enable_offline_packing is True


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

    handles.apply_public_dataset_override.assert_called_once_with(
        config,
        dataset_name="openmathinstruct2-thinking",
        enable_offline_packing=False,
        seq_length=None,
        dataset_root=None,
        train_data_path=None,
        validation_data_path=None,
        test_data_path=None,
        media_root=None,
        hf_processor_path=None,
    )


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

    def apply_launcher(current_config, *_args, **_kwargs):
        events.append("launcher")
        current_config.model.context_parallel_size = 2
        current_config.model.tensor_model_parallel_size = 3
        return current_config

    def apply_trailing(current_config, _overrides):
        events.append("trailing")
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

    handles.recipe_runner.apply_launcher_overrides.side_effect = apply_launcher
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
            "--offline-packing",
            "--cp",
            "2",
            "--tp",
            "3",
            "dataset.seq_length=2048",
        ]
    )

    assert handles.apply_public_dataset_override.call_args.kwargs["enable_offline_packing"] is True
    assert events == ["launcher", "trailing", "cp-invariants", ("packing", 2, 3, 2048)]


def test_offline_packing_is_rejected_for_vlm_dataset():
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()

    with pytest.raises(ValueError, match="does not support --offline-packing"):
        module.main(
            [
                "--recipe",
                "qwen3_vl_8b_sft_config",
                "--mode",
                "sft",
                "--dataset",
                "medpix",
                "--offline-packing",
            ]
        )


def test_media_root_is_rejected_for_local_vlm():
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()

    with pytest.raises(ValueError, match="used only by llava-video-178k"):
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
                "--train-data-path",
                "/data/vlm.jsonl",
                "--media-root",
                "/data/media",
            ]
        )


@pytest.mark.parametrize(
    "dataset_options",
    [
        ["--offline-packing"],
        ["--dataset-path", "/data/indexed"],
        ["--dataset-root", "/data/sft"],
        ["--train-data-path", "/data/vlm.jsonl"],
        ["--media-root", "/data/media"],
        ["--hf-processor-path", "Qwen/Qwen3-VL-8B-Instruct"],
    ],
)
def test_dataset_scoped_options_require_dataset(dataset_options):
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()

    with pytest.raises(ValueError, match="require --dataset"):
        module.main(["--model", "gpt_oss_20b", "--mode", "sft", *dataset_options])

    handles.recipe_runner.load_recipe.assert_not_called()


def test_indexed_options_are_rejected_for_non_indexed_dataset():
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()

    with pytest.raises(ValueError, match="used only by megatron-indexed"):
        module.main(
            [
                "--model",
                "gpt_oss_20b",
                "--mode",
                "sft",
                "--dataset",
                "squad",
                "--dataset-path",
                "/data/indexed",
            ]
        )

    handles.recipe_runner.load_recipe.assert_not_called()


@pytest.mark.parametrize("step_func", [None, "gpt_step", "LLM_STEP", "audio_lm_step"])
def test_vlm_dataset_requires_compatible_forward_step(step_func):
    module, handles = _load_module()
    handles.recipe_runner.load_recipe.return_value = SimpleNamespace()
    step_options = [] if step_func is None else ["--step-func", step_func]

    with pytest.raises(ValueError, match="requires a VLM-compatible --step-func"):
        module.main(["--model", "qwen3_vl_8b", "--mode", "sft", "--dataset", "medpix", *step_options])

    handles.recipe_runner.load_recipe.assert_not_called()


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

    assert handles.apply_public_dataset_override.call_args.kwargs["dataset_name"] == "medpix"


def test_vlm_step_modality_validation_matches_case_insensitive_loader():
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
            "megatron-indexed",
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
