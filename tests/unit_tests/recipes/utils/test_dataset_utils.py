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
    apply_dataset_override,
    extract_and_remove_override,
    get_blend_fields_from_data_paths,
    infer_mode_from_dataset,
)


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


# ---------------------------------------------------------------------------
# Tests for extract_and_remove_override
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractAndRemoveOverride:
    """Test cases for the extract_and_remove_override helper."""

    def test_extracts_matching_override(self):
        overrides = ["dataset.dataset_name=gsm8k", "train.train_iters=100"]
        result = extract_and_remove_override(overrides, "dataset.dataset_name")
        assert result == "gsm8k"
        assert overrides == ["train.train_iters=100"]

    def test_returns_default_when_not_found(self):
        overrides = ["train.train_iters=100"]
        result = extract_and_remove_override(overrides, "dataset.dataset_name", default="squad")
        assert result == "squad"
        assert overrides == ["train.train_iters=100"]

    def test_returns_none_when_not_found_and_no_default(self):
        overrides = ["train.train_iters=100"]
        result = extract_and_remove_override(overrides, "dataset.dataset_name")
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
        ["llm-finetune", "llm-finetune-preloaded", "vlm-energon", "vlm-hf", "vlm-preloaded"],
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
        from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig

        config = _make_mock_config()
        result = apply_dataset_override(config, "llm-finetune", seq_length=512)
        assert isinstance(result.dataset, HFDatasetConfig)
        assert result.dataset.dataset_name == "squad"

    def test_llm_finetune_extracts_dataset_name_from_cli(self):
        from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig

        config = _make_mock_config()
        overrides = ["dataset.dataset_name=gsm8k", "train.train_iters=10"]
        result = apply_dataset_override(config, "llm-finetune", seq_length=2048, cli_overrides=overrides)
        assert isinstance(result.dataset, HFDatasetConfig)
        assert result.dataset.dataset_name == "openai/gsm8k"
        assert "dataset.dataset_name=gsm8k" not in overrides
        assert "train.train_iters=10" in overrides

    def test_llm_finetune_openmathinstruct2(self):
        from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig

        config = _make_mock_config()
        overrides = ["dataset.dataset_name=openmathinstruct2"]
        result = apply_dataset_override(config, "llm-finetune", seq_length=4096, cli_overrides=overrides)
        assert isinstance(result.dataset, HFDatasetConfig)
        assert result.dataset.dataset_name == "nvidia/OpenMathInstruct-2"

    def test_llm_finetune_unknown_preset_raises(self):
        config = _make_mock_config()
        overrides = ["dataset.dataset_name=nonexistent"]
        with pytest.raises(ValueError, match="Unknown finetune dataset preset"):
            apply_dataset_override(config, "llm-finetune", cli_overrides=overrides)

    # -- LLM finetune preloaded -----------------------------------------------

    def test_llm_finetune_preloaded_creates_finetuning_config(self):
        from megatron.bridge.training.config import FinetuningDatasetConfig

        config = _make_mock_config()
        result = apply_dataset_override(config, "llm-finetune-preloaded", seq_length=2048)
        assert isinstance(result.dataset, FinetuningDatasetConfig)
        assert result.dataset.seq_length == 2048
        assert result.dataset.dataloader_type == "batch"

    # -- VLM energon ----------------------------------------------------------

    def test_vlm_energon_keeps_existing_provider(self):
        from megatron.bridge.data.energon.energon_provider import EnergonProvider

        existing = MagicMock(spec=EnergonProvider)
        config = _make_mock_config(dataset=existing)
        result = apply_dataset_override(config, "vlm-energon")
        assert result.dataset is existing

    def test_vlm_energon_creates_bare_provider_when_missing(self):
        from megatron.bridge.data.energon.energon_provider import EnergonProvider

        config = _make_mock_config(dataset=None, micro_batch_size=4, global_batch_size=64)
        result = apply_dataset_override(config, "vlm-energon", seq_length=4096)
        assert isinstance(result.dataset, EnergonProvider)
        assert result.dataset.path == ""
        assert result.dataset.seq_length == 4096
        assert result.dataset.micro_batch_size == 4
        assert result.dataset.global_batch_size == 64

    # -- VLM HF ---------------------------------------------------------------

    def test_vlm_hf_creates_provider(self):
        from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider

        config = _make_mock_config()
        result = apply_dataset_override(config, "vlm-hf", seq_length=4096)
        assert isinstance(result.dataset, HFDatasetConversationProvider)
        assert result.dataset.seq_length == 4096
        assert result.dataset.maker_name == "make_cord_v2_dataset"

    # -- VLM preloaded --------------------------------------------------------

    def test_vlm_preloaded_creates_provider(self):
        from megatron.bridge.data.vlm_datasets.preloaded_provider import PreloadedVLMConversationProvider

        config = _make_mock_config()
        result = apply_dataset_override(config, "vlm-preloaded", seq_length=2048)
        assert isinstance(result.dataset, PreloadedVLMConversationProvider)
        assert result.dataset.seq_length == 2048

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
        assert "vlm-preloaded" in DATASET_TYPES

    def test_llm_finetune_presets_has_expected_keys(self):
        assert "squad" in LLM_FINETUNE_PRESETS
        assert "gsm8k" in LLM_FINETUNE_PRESETS
        assert "openmathinstruct2" in LLM_FINETUNE_PRESETS

    def test_llm_finetune_presets_are_callable(self):
        for name, factory in LLM_FINETUNE_PRESETS.items():
            assert callable(factory), f"Preset '{name}' is not callable"
