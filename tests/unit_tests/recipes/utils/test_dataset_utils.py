# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.data.builders import (
    DirectHFSFTDatasetConfig,
    EnergonDatasetConfig,
    GPTSFTDatasetConfig,
    HFEnergonTaskEncoderConfig,
    MockVLMSFTDatasetConfig,
    QwenVLEnergonTaskEncoderConfig,
)
from megatron.bridge.data.sft_processing import ChatSFTPreprocessingConfig
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig
from megatron.bridge.recipes.utils.dataset_utils import (
    DATASET_PRESETS,
    build_dataset_config,
    dataset_train_mode,
    get_blend_fields_from_data_paths,
)
from megatron.bridge.training.config import GPTDatasetConfig, MockGPTDatasetConfig
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


pytestmark = pytest.mark.unit


class TestGetBlendFieldsFromDataPaths:
    def test_mock_mode_without_data(self):
        blend, blend_per_split, split = get_blend_fields_from_data_paths()

        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"

    def test_explicit_mock_ignores_paths(self):
        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths=["/path/to/data"],
            mock=True,
        )

        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"

    def test_data_paths_use_blend(self):
        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths=["0.6", "/path/to/data1", "0.4", "/path/to/data2"]
        )

        assert blend == (["/path/to/data1", "/path/to/data2"], [0.6, 0.4])
        assert blend_per_split is None
        assert split == "9999,8,2"

    def test_split_paths_use_blend_per_split(self):
        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            train_data_path=["/path/to/train"],
            valid_data_path=["/path/to/valid"],
            test_data_path=["/path/to/test"],
        )

        assert blend is None
        assert blend_per_split == [
            (["/path/to/train"], None),
            (["/path/to/valid"], None),
            (["/path/to/test"], None),
        ]
        assert split is None

    @patch("megatron.bridge.recipes.utils.dataset_utils.get_blend_and_blend_per_split")
    def test_empty_resolved_blend_falls_back_to_mock(self, mock_get_blend):
        mock_get_blend.return_value = (None, None)

        blend, blend_per_split, split = get_blend_fields_from_data_paths(data_paths=["/path/to/data"])

        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"


def _make_config(dataset=None, *, model_seq_length=4096):
    config = MagicMock()
    config.dataset = dataset
    config.model.seq_length = model_seq_length
    return config


def _make_vlm_config(*, do_validation=True, do_test=True):
    dataset = DirectHFSFTDatasetConfig(
        seq_length=4096,
        source=HFDatasetSourceConfig(dataset_name="cord_v2"),
        preprocessing=ChatSFTPreprocessingConfig(),
        hf_processor_path="Qwen/Qwen3-VL-8B-Instruct",
        do_validation=do_validation,
        do_test=do_test,
        dataloader_type="single",
        num_workers=3,
        persistent_workers=False,
        enable_in_batch_packing=True,
    )
    config = _make_config(dataset)
    config.model.temporal_patch_size = 2
    config.model.spatial_merge_size = 2
    config.model.patch_size = 16
    return config


class TestDatasetPresets:
    def test_registry_contains_only_public_names_and_factories(self):
        assert set(DATASET_PRESETS) == {
            "mock",
            "megatron-indexed",
            "energon",
            "squad",
            "tulu3",
            "coderforge",
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
        assert all(callable(factory) for factory in DATASET_PRESETS.values())

    @pytest.mark.parametrize(
        ("dataset_name", "expected_type"),
        [("mock", MockGPTDatasetConfig), ("megatron-indexed", GPTDatasetConfig)],
    )
    def test_pretraining_presets_use_model_sequence_length(self, dataset_name, expected_type):
        dataset = build_dataset_config(_make_config(model_seq_length=2048), dataset_name)

        assert isinstance(dataset, expected_type)
        assert dataset.seq_length == 2048
        assert dataset_train_mode(dataset) == "pretrain"

    def test_pretraining_seq_length_override_syncs_to_mcore_during_finalize(self):
        dataset = build_dataset_config(_make_config(model_seq_length=2048), "mock")

        process_config_with_overrides(dataset, cli_overrides=["seq_length=4096"])

        assert dataset.seq_length == 4096
        assert dataset.sequence_length == 2048
        serialized = dataset.to_cfg_dict()
        assert serialized["seq_length"] == 4096
        assert "sequence_length" not in serialized
        dataset.finalize()
        assert dataset.sequence_length == 4096

    def test_energon_preset_uses_recipe_processor_and_batch_size(self):
        config = _make_vlm_config()
        config.train.micro_batch_size = 2
        dataset = build_dataset_config(config, "energon")

        assert isinstance(dataset, EnergonDatasetConfig)
        assert dataset.path is None
        assert dataset.seq_length == 4096
        assert dataset.micro_batch_size == 2
        assert dataset.num_workers == 3
        assert isinstance(dataset.task_encoder, HFEnergonTaskEncoderConfig)
        assert dataset.task_encoder.hf_processor_path == "Qwen/Qwen3-VL-8B-Instruct"
        assert dataset.enable_in_batch_packing is True
        assert dataset.defer_in_batch_packing_to_step is False
        assert dataset_train_mode(dataset) is None

        process_config_with_overrides(
            dataset,
            cli_overrides=[
                "path=/data/datacomp-energon",
                "task_encoder.hf_processor_path=Qwen/Qwen3.6-35B-A3B",
                "task_encoder.hf_processor_revision=0123456789abcdef",
            ],
        )

        assert dataset.task_encoder.hf_processor_revision == "0123456789abcdef"
        dataset.validate()

    def test_energon_preset_preserves_recipe_owned_task_encoder(self):
        original = EnergonDatasetConfig(
            path="/data/original",
            seq_length=2048,
            micro_batch_size=4,
            num_workers=1,
            task_encoder=QwenVLEnergonTaskEncoderConfig(
                hf_processor_path="Qwen/model",
                patch_size=16,
            ),
            shuffle_buffer_size=321,
            defer_in_batch_packing_to_step=True,
        )
        config = _make_config(original, model_seq_length=4096)
        config.train.micro_batch_size = 2

        dataset = build_dataset_config(config, "energon")

        assert dataset is not original
        assert dataset.path is None
        assert dataset.seq_length == 4096
        assert dataset.micro_batch_size == 2
        assert dataset.num_workers == 1
        assert dataset.shuffle_buffer_size == 321
        assert dataset.defer_in_batch_packing_to_step is True
        assert isinstance(dataset.task_encoder, QwenVLEnergonTaskEncoderConfig)
        assert dataset.task_encoder.patch_size == 16
        assert dataset_train_mode(dataset) is None

    def test_energon_preset_accepts_mock_vlm_pretraining_recipe(self):
        source = MockVLMSFTDatasetConfig(
            seq_length=4096,
            hf_processor_path="Qwen/Qwen3.5-35B-A3B",
            num_workers=1,
            persistent_workers=False,
            pad_to_max_length=True,
        )
        config = _make_config(source)
        config.train.micro_batch_size = 1

        dataset = build_dataset_config(config, "energon")

        assert dataset.do_validation is True
        assert dataset.persistent_workers is False
        assert dataset.pad_to_max_length is True
        assert dataset.task_encoder.hf_processor_path == "Qwen/Qwen3.5-35B-A3B"

    def test_energon_preset_rejects_incompatible_recipe(self):
        with pytest.raises(ValueError, match="EnergonDatasetConfig or exposing dataset.hf_processor_path"):
            build_dataset_config(_make_config(), "energon")

    def test_energon_preset_rejects_invalid_workers(self):
        config = _make_vlm_config()
        config.train.micro_batch_size = 1
        config.dataset.num_workers = -1

        with pytest.raises(ValueError, match="non-negative"):
            build_dataset_config(config, "energon")

    @pytest.mark.parametrize(
        ("dataset_name", "source_name"),
        [
            ("squad", "squad"),
            ("tulu3", "tulu3"),
            ("coderforge", "coderforge"),
            ("openmathinstruct2", "openmathinstruct2"),
            ("openmathinstruct2-thinking", "openmathinstruct2_thinking"),
            ("gsm8k", "gsm8k"),
        ],
    )
    def test_text_sft_presets_return_complete_configs(self, dataset_name, source_name):
        dataset = build_dataset_config(_make_config(model_seq_length=2048), dataset_name)

        assert isinstance(dataset, GPTSFTDatasetConfig)
        assert dataset.hf_dataset is not None
        assert dataset.hf_dataset.dataset_name == source_name
        assert dataset.seq_length == 2048
        assert dataset.enable_offline_packing is False
        assert dataset_train_mode(dataset) == "finetune"

    def test_tulu3_preset_allows_typed_config_overrides(self):
        dataset = build_dataset_config(_make_config(model_seq_length=2048), "tulu3")

        process_config_with_overrides(
            dataset,
            cli_overrides=[
                "max_train_samples=128",
                "hf_validation_proportion=0.1",
            ],
        )

        assert dataset.max_train_samples == 128
        assert dataset.hf_validation_proportion == 0.1
        dataset.validate()

    def test_local_jsonl_is_a_config_preset_with_direct_overrides(self):
        dataset = build_dataset_config(_make_config(model_seq_length=1024), "local-jsonl")

        assert isinstance(dataset, GPTSFTDatasetConfig)
        assert dataset.dataset_root is None

        process_config_with_overrides(
            dataset,
            cli_overrides=[
                "dataset_root=/data/sft",
                "dataset_kwargs={custom_option:17}",
            ],
        )

        assert dataset.dataset_root == "/data/sft"
        assert dataset.dataset_kwargs == {"custom_option": 17}
        dataset.validate()

    def test_local_vlm_preserves_recipe_config_and_exposes_source_fields(self):
        dataset = build_dataset_config(_make_vlm_config(), "local-vlm")

        assert isinstance(dataset, DirectHFSFTDatasetConfig)
        assert dataset.num_workers == 3
        assert dataset.persistent_workers is False
        assert dataset.enable_in_batch_packing is True
        assert dataset.do_validation is False
        assert dataset.do_test is False

        process_config_with_overrides(
            dataset,
            cli_overrides=[
                "source.load_kwargs.data_files.train=/data/vlm/train.jsonl",
                "validation_source.load_kwargs.data_files.validation=/data/vlm/validation.jsonl",
                "do_validation=true",
                "hf_processor_path=org/processor",
            ],
        )

        assert dataset.source.load_kwargs == {"data_files": {"train": "/data/vlm/train.jsonl"}}
        assert dataset.validation_source is not None
        assert dataset.validation_source.load_kwargs == {"data_files": {"validation": "/data/vlm/validation.jsonl"}}
        assert dataset.hf_processor_path == "org/processor"
        dataset.validate()

    def test_local_vlm_requires_a_train_path_after_overrides(self):
        dataset = build_dataset_config(_make_vlm_config(), "local-vlm")

        with pytest.raises(ValueError, match="data_files must contain non-empty paths"):
            dataset.validate()

    @pytest.mark.parametrize(
        ("dataset_name", "source_name"),
        [("cord-v2", "cord_v2"), ("medpix", "medpix")],
    )
    def test_hf_vlm_presets_select_sources_and_preserve_processor(self, dataset_name, source_name):
        dataset = build_dataset_config(_make_vlm_config(), dataset_name)

        assert isinstance(dataset, DirectHFSFTDatasetConfig)
        assert dataset.source.dataset_name == source_name
        assert dataset.hf_processor_path == "Qwen/Qwen3-VL-8B-Instruct"
        assert dataset.num_workers == 3

    @pytest.mark.parametrize("dataset_name", ["raven", "rdr"])
    def test_train_only_vlm_presets_define_deterministic_validation_slices(self, dataset_name):
        dataset = build_dataset_config(_make_vlm_config(), dataset_name)

        assert isinstance(dataset, DirectHFSFTDatasetConfig)
        assert dataset.source.split == "train[:95%]"
        assert dataset.validation_source is not None
        assert dataset.validation_source.split == "train[95%:]"
        assert dataset.do_test is False

    def test_llava_video_exposes_required_adapter_kwargs(self):
        dataset = build_dataset_config(_make_vlm_config(), "llava-video-178k")

        with pytest.raises(ValueError, match="requires adapter_kwargs"):
            dataset.validate()

        process_config_with_overrides(
            dataset,
            cli_overrides=["source.adapter_kwargs.video_root_path=/data/llava-video"],
        )

        assert dataset.source.adapter_kwargs == {"video_root_path": "/data/llava-video"}
        dataset.validate()
        assert dataset.validation_source is not None
        assert dataset.validation_source.adapter_kwargs == {"video_root_path": "/data/llava-video"}

    def test_vlm_preset_rejects_a_non_direct_hf_recipe(self):
        with pytest.raises(ValueError, match="DirectHFSFTDatasetConfig"):
            build_dataset_config(_make_config(), "medpix")

    def test_unknown_dataset_name_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown dataset name"):
            build_dataset_config(_make_config(), "unknown")
