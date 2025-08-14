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

import pytest

from megatron.bridge.recipes.nemotron.nemotron4_15b_16k import pretrain_config
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestNemotron4_15B_16KPretrainConfig:
    """Test cases for the Nemotron4 15B 16K pretrain_config function."""

    def test_pretrain_config_default_sequence_length(self):
        """Test that pretrain_config uses 16K sequence length by default."""
        config = pretrain_config()

        assert isinstance(config, ConfigContainer)
        # Should use 16K sequence length
        assert config.dataset.sequence_length == 16384

    def test_pretrain_config(self):
        """Test that the 16K variant configuration."""
        config = pretrain_config()

        assert config.model.tensor_model_parallel_size == 2
        assert config.model.pipeline_model_parallel_size == 2
        assert config.model.context_parallel_size == 2
        assert config.model.sequence_parallel is True
        assert config.train.global_batch_size == 32
        assert config.train.micro_batch_size == 2
        assert config.optimizer.lr == 4.5e-5
        assert config.optimizer.min_lr == 4.5e-5
