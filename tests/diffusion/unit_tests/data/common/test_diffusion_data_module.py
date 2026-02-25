# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import Mock, patch

from megatron.bridge.diffusion.data.common.diffusion_energon_datamodule import (
    DiffusionDataModuleConfig,
)


def test_diffusion_data_module_config_initialization():
    """Test DiffusionDataModuleConfig initialization and default values."""

    # Mock the DiffusionDataModule to avoid actual dataset loading
    with patch("megatron.bridge.diffusion.data.common.diffusion_energon_datamodule.DiffusionDataModule") as mock_data_module:
        # Setup the mock to return a mock dataset with seq_length attribute
        mock_dataset_instance = Mock()
        mock_dataset_instance.seq_length = 2048
        mock_data_module.return_value = mock_dataset_instance

        # Create a DiffusionDataModuleConfig with required parameters
        config = DiffusionDataModuleConfig(
            path="/path/to/dataset",
            seq_length=2048,
            micro_batch_size=4,
            task_encoder_seq_length=512,
            packing_buffer_size=100,
            global_batch_size=32,
            num_workers=8,
        )

        # Verify default values
        assert config.dataloader_type == "external", "Expected default dataloader_type to be 'external'"
        assert config.use_train_split_for_val is False, "Expected default use_train_split_for_val to be False"

        # Verify required parameters are set correctly
        assert config.path == "/path/to/dataset"
        assert config.seq_length == 2048
        assert config.micro_batch_size == 4
        assert config.task_encoder_seq_length == 512
        assert config.packing_buffer_size == 100
        assert config.global_batch_size == 32
        assert config.num_workers == 8

        # Verify that DiffusionDataModule was created in __post_init__
        assert mock_data_module.called, "DiffusionDataModule should be instantiated in __post_init__"

        # Verify the dataset attribute was set
        assert config.dataset == mock_dataset_instance

        # Verify sequence_length was set from the dataset
        assert config.sequence_length == 2048, "Expected sequence_length to be set from dataset.seq_length"

        # Verify the DiffusionDataModule was created with correct parameters
        call_kwargs = mock_data_module.call_args.kwargs
        assert call_kwargs["path"] == "/path/to/dataset"
        assert call_kwargs["seq_length"] == 2048
        assert call_kwargs["micro_batch_size"] == 4
        assert call_kwargs["packing_buffer_size"] == 100
        assert call_kwargs["global_batch_size"] == 32
        assert call_kwargs["num_workers"] == 8
        assert call_kwargs["use_train_split_for_val"] is False
