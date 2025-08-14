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

from megatron.bridge.models.nemotron import Nemotron4ModelProvider340B
from megatron.bridge.recipes.nemotron.nemotron4_340b import model_config, pretrain_config
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestNemotron4_340BModelConfig:
    """Test cases for the Nemotron4 340B model_config function."""

    def test_model_config_default_parameters(self):
        """Test model_config with default parameters."""
        config = model_config()

        assert isinstance(config, Nemotron4ModelProvider340B)
        # Nemotron4 340B specific defaults
        assert config.tensor_model_parallel_size == 8
        assert config.pipeline_model_parallel_size == 12
        assert config.pipeline_dtype == torch.bfloat16
        assert config.virtual_pipeline_model_parallel_size == 8
        assert config.context_parallel_size == 2
        assert config.sequence_parallel is True


@pytest.mark.unit
class TestNemotron4_340BPretrainConfig:
    """Test cases for the Nemotron4 340B pretrain_config function."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters."""
        config = pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, Nemotron4ModelProvider340B)

        # Check Nemotron4 340B specific defaults
        assert config.train.global_batch_size == 2304  # Large batch for 340B model
        assert config.train.micro_batch_size == 1
        assert config.dataset.sequence_length == 4096
        assert config.optimizer.lr == 1.0e-4
        assert config.optimizer.min_lr == 1.0e-5
        assert config.scheduler.lr_warmup_iters == 500

        # Check model parallelism defaults
        assert config.model.tensor_model_parallel_size == 8
        assert config.model.pipeline_model_parallel_size == 12
        assert config.model.virtual_pipeline_model_parallel_size == 8
        assert config.model.context_parallel_size == 2
        assert config.model.sequence_parallel is True
