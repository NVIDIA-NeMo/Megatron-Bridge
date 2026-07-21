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

import subprocess
from pathlib import PosixPath

import pytest
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.data.builders import (
    ChatSFTPreprocessingConfig,
    GPTSFTDatasetConfig,
)
from megatron.bridge.data.builders.gpt_sft import GPTSFTDatasetBuilder
from megatron.bridge.data.packing import PackedSequenceSpecs
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


def get_dataset(
    ensure_test_data,
    dataset_dirname="finetune",
    packed_sequence_size=1,
    packed_train_data_path=None,
    packed_val_data_path=None,
    tokenizer_name="null",
):
    path = f"{ensure_test_data}/datasets/{dataset_dirname}"
    # path = "/home/data/finetune_dataset"
    if tokenizer_name == "null":
        tokenizer_config = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=131072)
        tokenizer_model_name = "null"
    elif tokenizer_name == "hf":
        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=f"{ensure_test_data}/tokenizers/huggingface",
        )
        tokenizer_model_name = None
    else:
        tokenizer_config = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=131072)
        tokenizer_model_name = None
    tokenizer = build_tokenizer(tokenizer_config)
    enable_offline_packing = packed_sequence_size > 0
    offline_packing_specs = (
        PackedSequenceSpecs(
            packed_sequence_size=packed_sequence_size,
            tokenizer_model_name=tokenizer_model_name,
            packed_train_data_path=packed_train_data_path,
            packed_val_data_path=packed_val_data_path,
        )
        if enable_offline_packing
        else None
    )

    dataset = GPTSFTDatasetBuilder(
        config=GPTSFTDatasetConfig(
            dataset_root=path,
            seq_length=2048,
            enable_offline_packing=enable_offline_packing,
            offline_packing_specs=offline_packing_specs,
        ),
        tokenizer=tokenizer,
    )

    return dataset, path


class TestGPTSFTDatasetBuilder:
    def test_extract_tokenizer_model_name(self, ensure_test_data):
        dataset, _ = get_dataset(ensure_test_data)
        tokenizer_name = dataset._extract_tokenizer_model_name()

        assert tokenizer_name == "null"

        dataset, _ = get_dataset(ensure_test_data, tokenizer_name="hf")
        tokenizer_name = dataset._extract_tokenizer_model_name()

        name = f"{ensure_test_data}/tokenizers/huggingface"
        name = name.replace("/", "--")
        assert tokenizer_name == name

        dataset, _ = get_dataset(ensure_test_data, tokenizer_name=None)
        tokenizer_name = dataset._extract_tokenizer_model_name()

        assert "unknown_tokenizer" in tokenizer_name

    def test_default_pack_path(self, ensure_test_data):
        dataset, path = get_dataset(ensure_test_data)
        default_pack_path = dataset.default_pack_path

        assert PosixPath(default_pack_path).parent == PosixPath(f"{path}/packed")
        assert PosixPath(default_pack_path).name.startswith("null_pad_seq_to_mult1_sft_")

    def test_train_path_packed(self, ensure_test_data):
        npy_path = f"{ensure_test_data}/datasets/finetune/test.npy"
        subprocess.run(["touch", npy_path])
        dataset, _ = get_dataset(ensure_test_data, packed_train_data_path=npy_path)
        train_path_packed = dataset.train_path_packed

        assert PosixPath(train_path_packed) == PosixPath(npy_path)

        dataset, _ = get_dataset(ensure_test_data)
        train_path_packed = dataset.train_path_packed

        assert PosixPath(train_path_packed) == PosixPath(dataset.default_pack_path / "training_1.idx.parquet")

        dataset, _ = get_dataset(ensure_test_data, packed_sequence_size=-1)

        with pytest.raises(ValueError):
            train_path_packed = dataset.train_path_packed

    def test_validation_path_packed(self, ensure_test_data):
        npy_path = f"{ensure_test_data}/datasets/finetune/test.npy"
        subprocess.run(["touch", npy_path])
        dataset, _ = get_dataset(ensure_test_data, packed_val_data_path=npy_path)
        validation_path_packed = dataset.validation_path_packed

        assert PosixPath(validation_path_packed) == PosixPath(npy_path)

        dataset, _ = get_dataset(ensure_test_data)
        validation_path_packed = dataset.validation_path_packed

        assert PosixPath(validation_path_packed) == PosixPath(dataset.default_pack_path / "validation_1.idx.parquet")

        dataset, _ = get_dataset(ensure_test_data, packed_sequence_size=-1)
        try:
            validation_path_packed = dataset.validation_path_packed
        except ValueError:
            None

    def test_prepare_packed_data(self, ensure_test_data):
        dataset, path = get_dataset(ensure_test_data)

        with pytest.raises(KeyError):
            dataset.prepare_packed_data()

    def test_paths_packed_with_msc_url(self, ensure_test_data):
        MultiStorageClientFeature.enable()

        npy_path = f"msc://default{ensure_test_data}/datasets/finetune/test.npy"
        msc = MultiStorageClientFeature.import_package()
        msc.Path(npy_path).touch(exist_ok=True)

        # Train
        dataset, _ = get_dataset(ensure_test_data, packed_train_data_path=npy_path)
        train_path_packed = dataset.train_path_packed

        assert train_path_packed == msc.Path(npy_path)

        dataset, _ = get_dataset(ensure_test_data)
        train_path_packed = dataset.train_path_packed

        assert train_path_packed == msc.Path(str(dataset.default_pack_path / "training_1.idx.parquet"))

        # Validation
        dataset, _ = get_dataset(ensure_test_data, packed_val_data_path=npy_path)
        validation_path_packed = dataset.validation_path_packed

        assert validation_path_packed == msc.Path(npy_path)

        dataset, _ = get_dataset(ensure_test_data)
        validation_path_packed = dataset.validation_path_packed

        assert validation_path_packed == msc.Path(str(dataset.default_pack_path / "validation_1.idx.parquet"))

        dataset, _ = get_dataset(ensure_test_data, packed_sequence_size=-1)

        with pytest.raises(ValueError):
            train_path_packed = dataset.train_path_packed

        with pytest.raises(ValueError):
            validation_path_packed = dataset.validation_path_packed

    def test_build_dataset_with_msc_url(self, ensure_test_data):
        MultiStorageClientFeature.enable()

        dataset_dirname = "finetune_msc"
        jsonl_example = '{"input": "John von Neumann Von Neumann made fundamental contributions ... Q: What did the math of artificial viscosity do?", "output": "smoothed the shock transition without sacrificing basic physics"}\n'

        msc = MultiStorageClientFeature.import_package()
        msc.Path(f"{ensure_test_data}/datasets/{dataset_dirname}").mkdir(exist_ok=True)

        with open(f"{ensure_test_data}/datasets/{dataset_dirname}/training.jsonl", "w") as f:
            for i in range(10):
                f.write(jsonl_example)

        with open(f"{ensure_test_data}/datasets/{dataset_dirname}/validation.jsonl", "w") as f:
            for i in range(10):
                f.write(jsonl_example)

        with open(f"{ensure_test_data}/datasets/{dataset_dirname}/test.jsonl", "w") as f:
            for i in range(10):
                f.write(jsonl_example)

        dataset, _ = get_dataset(
            ensure_test_data, dataset_dirname=dataset_dirname, packed_sequence_size=2048, tokenizer_name="hf"
        )
        assert not dataset.pack_metadata.exists()

        datasets = dataset.build()
        assert dataset.pack_metadata.exists()

        assert datasets[0] is not None
        assert datasets[1] is not None
        assert datasets[2] is not None


class TestGPTSFTDatasetBuilderWithChatTemplates:
    """Test GPTSFTDatasetBuilder with chat template and tool schema support."""

    def test_dataset_kwargs_passed_to_packing(self, tmp_path, monkeypatch):
        """Test that dataset_kwargs are passed to prepare_gpt_sft_packed_data."""
        from unittest.mock import patch

        # Create builder with dataset_kwargs
        dataset_kwargs = {"tool_schemas": {"type": "function"}}

        tokenizer_config = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=131072)
        tokenizer = build_tokenizer(tokenizer_config)

        offline_packing_specs = PackedSequenceSpecs(
            packed_sequence_size=2048,
            tokenizer_model_name="test_tokenizer",
        )

        builder = GPTSFTDatasetBuilder(
            config=GPTSFTDatasetConfig(
                dataset_root=tmp_path,
                seq_length=2048,
                enable_offline_packing=True,
                offline_packing_specs=offline_packing_specs,
                preprocessing=ChatSFTPreprocessingConfig(),
                dataset_kwargs=dataset_kwargs,
            ),
            tokenizer=tokenizer,
        )

        # Mock prepare_gpt_sft_packed_data to verify it receives dataset_kwargs
        with patch("megatron.bridge.data.packing.offline.prepare_gpt_sft_packed_data") as mock_prepare:
            # Mock file existence checks
            monkeypatch.setattr("pathlib.Path.is_file", lambda self: False)

            builder.prepare_packed_data()

            # Verify prepare_gpt_sft_packed_data was called twice (train and val)
            assert mock_prepare.call_count == 2

            # Verify dataset_kwargs were passed
            for call in mock_prepare.call_args_list:
                call_kwargs = call[1]
                assert call_kwargs["dataset_kwargs"] == {
                    "chat": True,
                    "use_hf_tokenizer_chat_template": True,
                    "chat_loss_mode": "assistant",
                    **dataset_kwargs,
                }

    def test_implicit_local_preprocessing_uses_legacy_defaults(self, tmp_path):
        """Test that omitted preprocessing preserves established local behavior."""
        tokenizer_config = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=131072)
        tokenizer = build_tokenizer(tokenizer_config)

        builder = GPTSFTDatasetBuilder(
            config=GPTSFTDatasetConfig(dataset_root=tmp_path, seq_length=2048),
            tokenizer=tokenizer,
        )

        assert builder.dataset_kwargs == {}

    def test_chat_template_with_packing(self, tmp_path):
        """Test that chat templates work with packed sequences."""
        from unittest.mock import MagicMock

        # Mock HF tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_id = 2

        offline_packing_specs = PackedSequenceSpecs(
            packed_sequence_size=2048,
            tokenizer_model_name="test_model",
        )

        builder = GPTSFTDatasetBuilder(
            config=GPTSFTDatasetConfig(
                dataset_root=tmp_path,
                seq_length=2048,
                enable_offline_packing=True,
                offline_packing_specs=offline_packing_specs,
                preprocessing=ChatSFTPreprocessingConfig(),
            ),
            tokenizer=mock_tokenizer,
        )

        # Verify dataset_kwargs are stored
        assert builder.dataset_kwargs["chat"] is True
        assert builder.dataset_kwargs["use_hf_tokenizer_chat_template"] is True

    def test_tool_schemas_with_packing(self, tmp_path):
        """Test that tool_schemas work with packed sequences."""
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_id = 2

        # Add tool schemas
        tool_schemas = [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather"},
            }
        ]

        dataset_kwargs = {"tool_schemas": tool_schemas}

        offline_packing_specs = PackedSequenceSpecs(
            packed_sequence_size=2048,
            tokenizer_model_name="test_model",
        )

        builder = GPTSFTDatasetBuilder(
            config=GPTSFTDatasetConfig(
                dataset_root=tmp_path,
                seq_length=2048,
                enable_offline_packing=True,
                offline_packing_specs=offline_packing_specs,
                preprocessing=ChatSFTPreprocessingConfig(),
                dataset_kwargs=dataset_kwargs,
            ),
            tokenizer=mock_tokenizer,
        )

        # Verify tool_schemas are stored in dataset_kwargs
        assert "tool_schemas" in builder.dataset_kwargs
        assert builder.dataset_kwargs["tool_schemas"] == tool_schemas


class TestPackedSequenceDatasetKwargs:
    """Test that dataset_kwargs flow through the packing pipeline."""

    def test_tokenize_dataset_with_chat_kwargs(self):
        """Test tokenize_dataset receives and uses dataset_kwargs."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from megatron.bridge.data.packing.offline import tokenize_dataset

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_id = 2

        dataset_kwargs = {
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
            "tool_schemas": [{"type": "function"}],
        }

        # Mock the builder-owned split helper to verify it receives kwargs
        with patch("megatron.bridge.data.builders.gpt_sft.build_gpt_sft_split") as mock_build:
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 0
            mock_build.return_value = mock_dataset

            tokenize_dataset(
                path=Path("test.jsonl"),
                tokenizer=mock_tokenizer,
                max_seq_length=512,
                seed=1234,
                dataset_kwargs=dataset_kwargs,
                dataset_builder=mock_build,
            )

            # Verify build_gpt_sft_split was called with kwargs
            mock_build.assert_called_once()
            dataset_options = mock_build.call_args.kwargs["dataset_kwargs"]
            assert dataset_options["chat"] is True
            assert dataset_options["use_hf_tokenizer_chat_template"] is True
            # tool_schemas should be converted to JSON string
            assert "tool_schemas" in dataset_options

    def test_tokenize_dataset_converts_tool_schemas_to_json(self):
        """Test that tool_schemas dict is converted to JSON string."""
        import json
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from megatron.bridge.data.packing.offline import tokenize_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_id = 2

        tool_schemas_dict = [{"type": "function", "function": {"name": "test"}}]
        dataset_kwargs = {"tool_schemas": tool_schemas_dict}

        with patch("megatron.bridge.data.builders.gpt_sft.build_gpt_sft_split") as mock_build:
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 0
            mock_build.return_value = mock_dataset

            tokenize_dataset(
                path=Path("test.jsonl"),
                tokenizer=mock_tokenizer,
                max_seq_length=512,
                seed=1234,
                dataset_kwargs=dataset_kwargs,
                dataset_builder=mock_build,
            )

            # Verify tool_schemas was converted to JSON string
            dataset_options = mock_build.call_args.kwargs["dataset_kwargs"]
            assert isinstance(dataset_options["tool_schemas"], str)
            # Should be parseable back to original
            parsed = json.loads(dataset_options["tool_schemas"])
            assert parsed == tool_schemas_dict

    def test_tokenize_dataset_sets_chat_template_on_tokenizer(self):
        """Test that chat_template from dataset_kwargs is set on tokenizer."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from megatron.bridge.data.packing.offline import tokenize_dataset

        # Mock HF tokenizer
        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        custom_template = "{% for msg in messages %}{{ msg.content }}{% endfor %}"
        dataset_kwargs = {"chat_template": custom_template}

        with patch("megatron.bridge.data.builders.gpt_sft.build_gpt_sft_split") as mock_build:
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 0
            mock_build.return_value = mock_dataset

            tokenize_dataset(
                path=Path("test.jsonl"),
                tokenizer=mock_tokenizer,
                max_seq_length=512,
                seed=1234,
                dataset_kwargs=dataset_kwargs,
                dataset_builder=mock_build,
            )

            # Verify chat_template was set on tokenizer
            assert mock_hf_tokenizer.chat_template == custom_template

            # Verify chat_template was popped before runtime dataset construction.
            dataset_options = mock_build.call_args.kwargs["dataset_kwargs"]
            assert "chat_template" not in dataset_options
