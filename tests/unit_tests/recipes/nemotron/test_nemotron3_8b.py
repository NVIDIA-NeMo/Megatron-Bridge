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
import torch

from megatron.bridge.models.nemotron import Nemotron3ModelProvider8B
from megatron.bridge.recipes.nemotron.nemotron3_8b import model_config, pretrain_config
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestNemotron3_8BModelConfig:
    """Test cases for the Nemotron3 8B model_config function."""

    def test_model_config_default_parameters(self):
        """Test model_config with default parameters."""
        config = model_config()

        assert isinstance(config, Nemotron3ModelProvider8B)
        assert config.tensor_model_parallel_size == 2
        assert config.pipeline_model_parallel_size == 1
        assert config.pipeline_dtype == torch.bfloat16
        assert config.virtual_pipeline_model_parallel_size is None
        assert config.context_parallel_size == 1
        assert config.sequence_parallel is False


@pytest.mark.unit
class TestNemotron3_8BPretrainConfig:
    """Test cases for the Nemotron3 8B pretrain_config function."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters."""
        config = pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, Nemotron3ModelProvider8B)

        # Check Nemotron3 8B specific defaults
        assert config.train.global_batch_size == 32
        assert config.train.micro_batch_size == 2
        assert config.dataset.sequence_length == 4096
        assert config.scheduler.lr_warmup_iters == 500

        # Check model parallelism defaults
        assert config.model.tensor_model_parallel_size == 2
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.context_parallel_size == 1
        assert config.model.sequence_parallel is False
