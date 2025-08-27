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

from megatron.bridge.models.nemotron import Nemotron4ModelProvider15B
from megatron.bridge.recipes.nemotron.nemotron4_15b import model_config, pretrain_config
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestNemotron4_15BModelConfig:
    """Test cases for the Nemotron4 15B model_config function."""

    def test_model_config_default_parameters(self):
        """Test model_config with default parameters."""
        config = model_config()

        assert isinstance(config, Nemotron4ModelProvider15B)
        # Nemotron4 15B specific defaults
        assert config.tensor_model_parallel_size == 4
        assert config.pipeline_model_parallel_size == 1
        assert config.virtual_pipeline_model_parallel_size is None
        assert config.context_parallel_size == 1
        assert config.sequence_parallel is True  # Nemotron4 15B uses sequence parallelism


@pytest.mark.unit
class TestNemotron4_15BPretrainConfig:
    """Test cases for the Nemotron4 15B pretrain_config function."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters."""
        config = pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, Nemotron4ModelProvider15B)

        # Check Nemotron4 15B specific defaults
        assert config.train.global_batch_size == 32
        assert config.train.micro_batch_size == 2
        assert config.dataset.sequence_length == 4096
        assert config.optimizer.lr == 4.5e-5
        assert config.optimizer.min_lr == 4.5e-5  # Same as max for this model
        assert config.scheduler.lr_warmup_iters == 500

        # Check model parallelism defaults
        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.context_parallel_size == 1
        assert config.model.sequence_parallel is True
