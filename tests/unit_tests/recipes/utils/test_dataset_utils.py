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

from megatron.bridge.recipes.utils.dataset_utils import (
    DATASET_TYPES,
    LLM_FINETUNE_PRESETS,
    PUBLIC_DATASET_NAMES,
    PUBLIC_DATASETS,
    PUBLIC_HF_VLM_DATASETS,
    apply_dataset_override,
    apply_public_dataset_override,
    extract_and_remove_override,
    get_blend_fields_from_data_paths,
    infer_mode_from_dataset,
)
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


@pytest.mark.unit
class TestGetBlendFieldsFromDataPaths:
    """Test cases for the get_blend_fields_from_data_paths function."""

    def test_mock_mode_explicit(self):
        """Test function with explicit mock=True."""
        blend, blend_per_split, split = get_blend_fields_from_data_paths(mock=True)

        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"

    def test_mock_mode_no_data_config(self):
        """Test function with no data configuration (should default to mock)."""
        blend, blend_per_split, split = get_blend_fields_from_data_paths()

        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"

    def test_mock_mode_with_data_paths_but_mock_true(self):
        """Test function with data paths but mock=True (should ignore data paths)."""
        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths=["/path/to/data1", "/path/to/data2"], mock=True
        )

        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"

    def test_data_paths(self):
        """Test function with data_paths and blend weights returned."""
        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths=["/path/to/data1", "/path/to/data2"]
        )

        assert blend == (["/path/to/data1", "/path/to/data2"], None)
        assert blend_per_split is None
        assert split == "9999,8,2"

        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths=["0.6", "/path/to/data1", "0.4", "/path/to/data2"]
        )

        assert blend == (["/path/to/data1", "/path/to/data2"], [0.6, 0.4])
        assert blend_per_split is None
        assert split == "9999,8,2"

    def test_data_args_path_with_blend_weights(self):
        """Test function with data_args_path and blend weights returned."""

        import tempfile

        content = "0.6\n/path/to/data1\n0.4\n/path/to/data2\n"
        with tempfile.NamedTemporaryFile(prefix="datasrc_") as data_args_file:
            data_args_file.write(str.encode(content))
            data_args_file.seek(0)

            blend, blend_per_split, split = get_blend_fields_from_data_paths(data_args_path=data_args_file.name)

            assert blend == (["/path/to/data1", "/path/to/data2"], [0.6, 0.4])
            assert blend_per_split is None
            assert split == "9999,8,2"

    def test_per_split_paths_with_blend_per_split_weights(self):
        """Test function with train/valid/test paths and blend_per_split weights."""
        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            train_data_path=["/path/to/train1", "/path/to/train2"],
            valid_data_path=["/path/to/valid1"],
            test_data_path=["/path/to/test1", "/path/to/test2"],
        )

        assert blend is None
        assert blend_per_split == [
            (["/path/to/train1", "/path/to/train2"], None),
            (["/path/to/valid1"], None),
            (["/path/to/test1", "/path/to/test2"], None),
        ]
        assert split is None

        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            train_data_path=["0.8", "/path/to/train1", "0.2", "/path/to/train2"],
            valid_data_path=["0.7", "/path/to/valid1", "0.3", "/path/to/valid2"],
            test_data_path=["0.6", "/path/to/test1", "0.4", "/path/to/test2"],
        )

        assert blend is None
        assert blend_per_split == [
            (["/path/to/train1", "/path/to/train2"], [0.8, 0.2]),
            (["/path/to/valid1", "/path/to/valid2"], [0.7, 0.3]),
            (["/path/to/test1", "/path/to/test2"], [0.6, 0.4]),
        ]
        assert split is None

    def test_per_split_data_args_path_with_blend_per_split_weights(self):
        """Test function with per_split_data_args_path and blend_per_split weights."""

        import json
        import tempfile

        content = {
            "train": ["0.8", "/path/to/train1", "0.2", "/path/to/train2"],
            "valid": ["0.7", "/path/to/valid1", "0.3", "/path/to/valid2"],
            "test": ["0.6", "/path/to/test1", "0.4", "/path/to/test2"],
        }
        with tempfile.NamedTemporaryFile("w+", prefix="datasrc_", suffix=".json") as per_split_data_args_file:
            json.dump(content, per_split_data_args_file)
            per_split_data_args_file.seek(0)

            blend, blend_per_split, split = get_blend_fields_from_data_paths(
                per_split_data_args_path=per_split_data_args_file.name
            )

            assert blend is None
            assert blend_per_split == [
                (["/path/to/train1", "/path/to/train2"], [0.8, 0.2]),
                (["/path/to/valid1", "/path/to/valid2"], [0.7, 0.3]),
                (["/path/to/test1", "/path/to/test2"], [0.6, 0.4]),
            ]
            assert split is None

    def test_prioritize_blend_over_blend_per_split(self):
        """Test that data_paths takes priority over split data paths when both are provided."""

        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths=["/path/to/data1", "/path/to/data2"],
            train_data_path=["/path/to/train1", "/path/to/train2"],
            valid_data_path=["/path/to/valid1", "/path/to/valid2"],
            test_data_path=["/path/to/test1", "/path/to/test2"],
        )

        # Should prioritize blend over blend_per_split
        assert blend == (["/path/to/data1", "/path/to/data2"], None)
        assert blend_per_split is None
        assert split == "9999,8,2"

    @patch("megatron.bridge.recipes.utils.dataset_utils.get_blend_and_blend_per_split")
    def test_fallback_to_mock_when_no_weights(self, mock_get_blend):
        """Test function falls back to mock mode when no weights are returned."""
        mock_get_blend.return_value = (None, None)

        blend, blend_per_split, split = get_blend_fields_from_data_paths(data_paths=["/some/path"])

        # Should fall back to mock mode
        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"

    def test_blend_per_split_with_empty_paths(self):
        """Test blend_per_split with empty paths (should create None entries)."""
        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            valid_data_path=["/path/to/valid1"],  # Only valid paths
            test_data_path=None,  # No test paths
        )

        assert blend is None
        assert blend_per_split == [
            None,  # train_paths is empty, so None
            (["/path/to/valid1"], None),  # valid_paths exists
            None,  # test_paths is None, so None
        ]
        assert split is None

    def test_edge_case_empty_lists(self):
        """Test edge case with empty lists for all path parameters."""

        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths=[],
            train_data_path=[],
            valid_data_path=[],
            test_data_path=[],
        )

        assert blend is None
        assert blend_per_split is None
        assert split == "1,1,1"


# ---------------------------------------------------------------------------
# Helper to build a lightweight mock ConfigContainer for apply_dataset_override
# ---------------------------------------------------------------------------


def _make_mock_config(dataset=None, model_seq_length=4096, micro_batch_size=2, global_batch_size=32):
    """Return a MagicMock that quacks like ConfigContainer for dataset override tests."""
    config = MagicMock()
    config.dataset = dataset
    config.model.seq_length = model_seq_length
    config.train.micro_batch_size = micro_batch_size
    config.train.global_batch_size = global_batch_size
    return config


def _make_vlm_config(*, do_validation=True, do_test=True):
    """Return a config backed by the declarative direct-HF VLM provider."""
    from megatron.bridge.data.builders import DirectHFSFTDatasetConfig
    from megatron.bridge.data.sft_processing import ChatSFTPreprocessingConfig
    from megatron.bridge.data.sources.hf import HFDatasetSourceConfig

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
    return _make_mock_config(dataset=dataset)


# ---------------------------------------------------------------------------
# Tests for extract_and_remove_override
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractAndRemoveOverride:
    """Test cases for the extract_and_remove_override helper."""

    def test_extracts_matching_override(self):
        overrides = ["dataset.hf_dataset.dataset_name=gsm8k", "train.train_iters=100"]
        result = extract_and_remove_override(overrides, "dataset.hf_dataset.dataset_name")
        assert result == "gsm8k"
        assert overrides == ["train.train_iters=100"]

    def test_returns_default_when_not_found(self):
        overrides = ["train.train_iters=100"]
        result = extract_and_remove_override(overrides, "dataset.hf_dataset.dataset_name", default="squad")
        assert result == "squad"
        assert overrides == ["train.train_iters=100"]

    def test_returns_none_when_not_found_and_no_default(self):
        overrides = ["train.train_iters=100"]
        result = extract_and_remove_override(overrides, "dataset.hf_dataset.dataset_name")
        assert result is None

    def test_handles_empty_list(self):
        overrides = []
        result = extract_and_remove_override(overrides, "dataset.path", default="/data")
        assert result == "/data"
        assert overrides == []

    def test_handles_value_with_equals_sign(self):
        overrides = ["dataset.blend=path/a=b"]
        result = extract_and_remove_override(overrides, "dataset.blend")
        assert result == "path/a=b"
        assert overrides == []

    def test_only_removes_first_match(self):
        overrides = ["dataset.path=/a", "dataset.path=/b"]
        result = extract_and_remove_override(overrides, "dataset.path")
        assert result == "/a"
        assert overrides == ["dataset.path=/b"]


# ---------------------------------------------------------------------------
# Tests for infer_mode_from_dataset
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInferModeFromDataset:
    """Test cases for infer_mode_from_dataset."""

    @pytest.mark.parametrize("dataset_type", ["llm-pretrain", "llm-pretrain-mock"])
    def test_pretrain_types(self, dataset_type):
        assert infer_mode_from_dataset(dataset_type) == "pretrain"

    @pytest.mark.parametrize(
        "dataset_type",
        ["llm-finetune", "llm-finetune-preloaded", "vlm-energon", "vlm-hf"],
    )
    def test_finetune_types(self, dataset_type):
        assert infer_mode_from_dataset(dataset_type) == "finetune"

    def test_all_dataset_types_covered(self):
        """Every entry in DATASET_TYPES should return a valid mode."""
        for dt in DATASET_TYPES:
            mode = infer_mode_from_dataset(dt)
            assert mode in ("pretrain", "finetune"), f"Unexpected mode '{mode}' for dataset type '{dt}'"


# ---------------------------------------------------------------------------
# Tests for apply_dataset_override
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyDatasetOverride:
    """Test cases for apply_dataset_override."""

    # -- LLM pretrain ---------------------------------------------------------

    def test_llm_pretrain_creates_gpt_dataset_config(self):
        from megatron.bridge.training.config import GPTDatasetConfig

        config = _make_mock_config()
        result = apply_dataset_override(config, "llm-pretrain", seq_length=2048)
        assert isinstance(result.dataset, GPTDatasetConfig)
        assert result.dataset.sequence_length == 2048

    def test_llm_pretrain_uses_model_seq_length_as_fallback(self):
        from megatron.bridge.training.config import GPTDatasetConfig

        config = _make_mock_config(model_seq_length=8192)
        result = apply_dataset_override(config, "llm-pretrain")
        assert isinstance(result.dataset, GPTDatasetConfig)
        assert result.dataset.sequence_length == 8192

    # -- LLM pretrain mock ----------------------------------------------------

    def test_llm_pretrain_mock_creates_mock_gpt_dataset_config(self):
        from megatron.bridge.training.config import MockGPTDatasetConfig

        config = _make_mock_config()
        result = apply_dataset_override(config, "llm-pretrain-mock", seq_length=1024)
        assert isinstance(result.dataset, MockGPTDatasetConfig)

    # -- LLM finetune ---------------------------------------------------------

    def test_llm_finetune_defaults_to_squad(self):
        from megatron.bridge.data.builders import GPTSFTDatasetConfig

        config = _make_mock_config()
        result = apply_dataset_override(config, "llm-finetune", seq_length=512)
        assert isinstance(result.dataset, GPTSFTDatasetConfig)
        assert result.dataset.hf_dataset.dataset_name == "squad"

    def test_llm_finetune_extracts_dataset_name_from_cli(self):
        from megatron.bridge.data.builders import GPTSFTDatasetConfig

        config = _make_mock_config()
        overrides = ["dataset.hf_dataset.dataset_name=gsm8k", "train.train_iters=10"]
        result = apply_dataset_override(config, "llm-finetune", seq_length=2048, cli_overrides=overrides)
        assert isinstance(result.dataset, GPTSFTDatasetConfig)
        assert result.dataset.hf_dataset.dataset_name == "gsm8k"
        assert "dataset.hf_dataset.dataset_name=gsm8k" not in overrides
        assert "train.train_iters=10" in overrides

    def test_llm_finetune_openmathinstruct2(self):
        from megatron.bridge.data.builders import GPTSFTDatasetConfig

        config = _make_mock_config()
        overrides = ["dataset.hf_dataset.dataset_name=openmathinstruct2"]
        result = apply_dataset_override(config, "llm-finetune", seq_length=4096, cli_overrides=overrides)
        assert isinstance(result.dataset, GPTSFTDatasetConfig)
        assert result.dataset.hf_dataset.dataset_name == "openmathinstruct2"

    def test_llm_finetune_unknown_preset_raises(self):
        config = _make_mock_config()
        overrides = ["dataset.hf_dataset.dataset_name=nonexistent"]
        with pytest.raises(ValueError, match="Unknown finetune dataset preset"):
            apply_dataset_override(config, "llm-finetune", cli_overrides=overrides)

    # -- LLM finetune preloaded -----------------------------------------------

    def test_llm_finetune_preloaded_creates_finetuning_config(self):
        from megatron.bridge.data.builders import GPTSFTDatasetConfig

        config = _make_mock_config()
        overrides = ["dataset.dataset_root=/data/sft", "train.train_iters=10"]
        result = apply_dataset_override(
            config,
            "llm-finetune-preloaded",
            seq_length=2048,
            cli_overrides=overrides,
        )
        assert isinstance(result.dataset, GPTSFTDatasetConfig)
        assert result.dataset.seq_length == 2048
        assert result.dataset.dataset_root == "/data/sft"
        assert result.dataset.dataloader_type == "batch"
        assert "dataset.dataset_root=/data/sft" not in overrides
        assert "train.train_iters=10" in overrides

    def test_llm_finetune_preloaded_requires_local_source(self):
        config = _make_mock_config()

        with pytest.raises(ValueError, match="requires dataset.dataset_root"):
            apply_dataset_override(config, "llm-finetune-preloaded", seq_length=2048)

    # -- VLM energon ----------------------------------------------------------

    def test_vlm_energon_keeps_existing_declarative_config(self):
        from megatron.bridge.data.builders import EnergonDatasetConfig, HFEnergonTaskEncoderConfig

        existing = EnergonDatasetConfig(
            path="/data/shards",
            seq_length=4096,
            micro_batch_size=2,
            task_encoder=HFEnergonTaskEncoderConfig(hf_processor_path="org/model"),
        )
        config = _make_mock_config(dataset=existing)
        result = apply_dataset_override(config, "vlm-energon")
        assert result.dataset is existing

    def test_vlm_energon_applies_explicit_sequence_length_to_dataset_and_model(self):
        from megatron.bridge.data.builders import EnergonDatasetConfig, HFEnergonTaskEncoderConfig

        existing = EnergonDatasetConfig(
            path="/data/shards",
            seq_length=4096,
            micro_batch_size=2,
            task_encoder=HFEnergonTaskEncoderConfig(hf_processor_path="org/model"),
        )
        config = _make_mock_config(dataset=existing, model_seq_length=4096)

        result = apply_dataset_override(config, "vlm-energon", seq_length=8192)

        assert result.dataset is existing
        assert result.dataset.seq_length == 8192
        assert result.model.seq_length == 8192

    def test_vlm_energon_requires_recipe_specific_task_encoder_config(self):
        config = _make_mock_config(dataset=None, micro_batch_size=4, global_batch_size=64)

        with pytest.raises(ValueError, match="model-specific task_encoder"):
            apply_dataset_override(config, "vlm-energon", seq_length=4096)

    # -- VLM HF ---------------------------------------------------------------

    def test_vlm_hf_creates_config(self):
        from megatron.bridge.data.builders import DirectHFSFTDatasetConfig

        config = _make_mock_config()
        result = apply_dataset_override(config, "vlm-hf", seq_length=4096)
        assert isinstance(result.dataset, DirectHFSFTDatasetConfig)
        assert result.dataset.seq_length == 4096
        assert result.dataset.source.dataset_name == "cord_v2"

    def test_vlm_hf_accepts_tutorial_json_source_overrides(self):
        config = _make_mock_config()
        result = apply_dataset_override(config, "vlm-hf", seq_length=1024)

        process_config_with_overrides(
            result.dataset,
            cli_overrides=[
                "source.dataset_name=null",
                "source.path_or_dataset=json",
                "source.split=train",
                "source.load_kwargs={data_files:{train:/tmp/training.jsonl}}",
            ],
        )

        assert result.dataset.seq_length == 1024
        assert result.model.seq_length == 1024
        assert result.dataset.source.path_or_dataset == "json"
        assert result.dataset.source.load_kwargs == {"data_files": {"train": "/tmp/training.jsonl"}}

    # -- Unknown type ---------------------------------------------------------

    def test_unknown_dataset_type_raises(self):
        config = _make_mock_config()
        with pytest.raises(ValueError, match="Unknown dataset type"):
            apply_dataset_override(config, "not-a-real-type")

    # -- seq_length sync ------------------------------------------------------

    def test_model_seq_length_synced_when_explicit(self):
        config = _make_mock_config(model_seq_length=4096)
        apply_dataset_override(config, "llm-pretrain-mock", seq_length=2048)
        assert config.model.seq_length == 2048

    def test_model_seq_length_not_overwritten_when_implicit(self):
        """When seq_length is None, model.seq_length should not be changed."""
        config = _make_mock_config(model_seq_length=8192)
        apply_dataset_override(config, "llm-pretrain-mock")
        assert config.model.seq_length == 8192


# ---------------------------------------------------------------------------
# Tests for apply_public_dataset_override
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyPublicDatasetOverride:
    """Test cases for public launcher dataset names."""

    def test_megatron_indexed_public_name_selects_gpt_dataset_config(self):
        from megatron.bridge.training.config import GPTDatasetConfig

        config = _make_mock_config()
        result = apply_public_dataset_override(config, "megatron-indexed", seq_length=2048)

        assert isinstance(result.dataset, GPTDatasetConfig)
        assert result.dataset.sequence_length == 2048
        assert result.dataset.blend is None
        assert result.dataset.blend_per_split is None

    def test_squad_public_name_selects_squad_preset(self):
        from megatron.bridge.data.builders import GPTSFTDatasetConfig

        config = _make_mock_config()
        result = apply_public_dataset_override(config, "squad", seq_length=2048)

        assert isinstance(result.dataset, GPTSFTDatasetConfig)
        assert result.dataset.hf_dataset.dataset_name == "squad"

    def test_local_jsonl_public_name_uses_explicit_local_source(self):
        from megatron.bridge.data.builders import GPTSFTDatasetConfig

        config = _make_mock_config()
        result = apply_public_dataset_override(config, "local-jsonl", seq_length=2048, dataset_root="/data/sft")

        assert isinstance(result.dataset, GPTSFTDatasetConfig)
        assert result.dataset.dataset_root == "/data/sft"

    def test_openmathinstruct2_thinking_public_name_does_not_imply_packing(self):
        from megatron.bridge.data.builders import GPTSFTDatasetConfig

        config = _make_mock_config()
        result = apply_public_dataset_override(config, "openmathinstruct2-thinking", seq_length=2048)

        assert isinstance(result.dataset, GPTSFTDatasetConfig)
        assert result.dataset.hf_dataset.dataset_name == "openmathinstruct2_thinking"
        assert result.dataset.enable_offline_packing is False
        assert result.dataset.offline_packing_specs is None

    def test_offline_packing_is_an_independent_text_dataset_option(self):
        config = _make_mock_config()
        result = apply_public_dataset_override(
            config,
            "openmathinstruct2-thinking",
            offline_packing=True,
            seq_length=2048,
            pad_seq_to_mult=4,
        )

        assert result.dataset.enable_offline_packing is True
        assert result.dataset.offline_packing_specs.pad_seq_to_mult == 4
        assert result.dataset.dataset_kwargs == {"pad_to_max_length": True}

    def test_local_vlm_uses_explicit_paths_and_inherits_recipe_processor(self):
        from megatron.bridge.data.vlm_datasets.preloaded_provider import PreloadedVLMConversationProvider

        config = _make_vlm_config()
        result = apply_public_dataset_override(
            config,
            "local-vlm",
            train_data_path="/data/vlm/train.jsonl",
            validation_data_path="/data/vlm/validation.jsonl",
            test_data_path="/data/vlm/test.jsonl",
            media_root="/data/vlm/media",
        )

        assert isinstance(result.dataset, PreloadedVLMConversationProvider)
        assert result.dataset.train_data_path == "/data/vlm/train.jsonl"
        assert result.dataset.valid_data_path == "/data/vlm/validation.jsonl"
        assert result.dataset.test_data_path == "/data/vlm/test.jsonl"
        assert result.dataset.image_folder == "/data/vlm/media"
        assert result.dataset.hf_processor_path == "Qwen/Qwen3-VL-8B-Instruct"
        assert result.dataset.num_workers == 3
        assert result.dataset.persistent_workers is False
        assert result.dataset.enable_in_batch_packing is True

    def test_medpix_selects_existing_hf_vlm_preset(self):
        config = _make_vlm_config()
        result = apply_public_dataset_override(config, "medpix")

        assert result.dataset.source.dataset_name == "medpix"
        assert result.dataset.validation_source is None
        assert result.dataset.do_validation is True
        assert result.dataset.do_test is False
        assert result.dataset.hf_processor_path == "Qwen/Qwen3-VL-8B-Instruct"

    @pytest.mark.parametrize("dataset_name", ["raven", "rdr"])
    def test_train_only_hf_vlm_presets_get_deterministic_validation_slice(self, dataset_name):
        config = _make_vlm_config()
        result = apply_public_dataset_override(config, dataset_name)

        assert result.dataset.source.dataset_name == PUBLIC_HF_VLM_DATASETS[dataset_name]
        assert result.dataset.source.split == "train[:95%]"
        assert result.dataset.validation_source.split == "train[95%:]"
        assert result.dataset.do_test is False

    def test_llava_video_requires_and_forwards_media_root(self):
        config = _make_vlm_config()
        with pytest.raises(ValueError, match="requires --media-root"):
            apply_public_dataset_override(config, "llava-video-178k")

        result = apply_public_dataset_override(
            config,
            "llava-video-178k",
            media_root="/data/llava-video",
        )
        assert result.dataset.source.dataset_name == "llava_video_178k"
        assert result.dataset.source.adapter_kwargs == {"video_root_path": "/data/llava-video"}
        assert result.dataset.source.split == "train[:95%]"
        assert result.dataset.validation_source.split == "train[95%:]"

    def test_offline_packing_is_rejected_for_vlm_presets(self):
        config = _make_vlm_config()

        with pytest.raises(ValueError, match="supported only for text SFT"):
            apply_public_dataset_override(config, "medpix", offline_packing=True)

    def test_unknown_public_name_raises(self):
        config = _make_mock_config()

        with pytest.raises(ValueError, match="Unknown dataset name"):
            apply_public_dataset_override(config, "llm-finetune")


# ---------------------------------------------------------------------------
# Tests for registry constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegistryConstants:
    """Sanity checks for DATASET_TYPES and LLM_FINETUNE_PRESETS."""

    def test_dataset_types_has_expected_entries(self):
        assert "llm-pretrain" in DATASET_TYPES
        assert "llm-pretrain-mock" in DATASET_TYPES
        assert "llm-finetune" in DATASET_TYPES
        assert "llm-finetune-preloaded" in DATASET_TYPES
        assert "vlm-energon" in DATASET_TYPES
        assert "vlm-hf" in DATASET_TYPES
        assert "vlm-local" not in DATASET_TYPES
        assert "vlm-preloaded" not in DATASET_TYPES

    def test_llm_finetune_presets_has_expected_keys(self):
        assert "squad" in LLM_FINETUNE_PRESETS
        assert "gsm8k" in LLM_FINETUNE_PRESETS
        assert "openmathinstruct2" in LLM_FINETUNE_PRESETS

    def test_llm_finetune_presets_are_callable(self):
        for name, factory in LLM_FINETUNE_PRESETS.items():
            assert callable(factory), f"Preset '{name}' is not callable"

    def test_public_dataset_names_has_expected_entries(self):
        assert "squad" in PUBLIC_DATASET_NAMES
        assert "openmathinstruct2" in PUBLIC_DATASET_NAMES
        assert "openmathinstruct2-thinking" in PUBLIC_DATASET_NAMES
        assert "local-jsonl" in PUBLIC_DATASET_NAMES
        assert "local-vlm" in PUBLIC_DATASET_NAMES
        assert "megatron-indexed" in PUBLIC_DATASET_NAMES
        assert set(PUBLIC_HF_VLM_DATASETS).issubset(PUBLIC_DATASET_NAMES)
        assert "squad-packed" not in PUBLIC_DATASET_NAMES
        assert "preloaded-vlm" not in PUBLIC_DATASET_NAMES
        assert {"dclm", "rp2", "c4"}.isdisjoint(PUBLIC_DATASET_NAMES)

    def test_public_dataset_views_are_derived_from_one_registry(self):
        assert PUBLIC_DATASET_NAMES == list(PUBLIC_DATASETS)
        assert PUBLIC_HF_VLM_DATASETS == {
            name: spec.hf_dataset_name for name, spec in PUBLIC_DATASETS.items() if spec.hf_dataset_name is not None
        }
        assert all(PUBLIC_DATASETS[name].modality == "vlm" for name in PUBLIC_HF_VLM_DATASETS)
