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

from megatron.bridge.diffusion.data.common.diffusion_energon_datamodule import (
    DiffusionDataModuleConfig,
)


def test_diffusion_data_module_config_initialization():
    """Test DiffusionDataModuleConfig initialization and default values."""

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
