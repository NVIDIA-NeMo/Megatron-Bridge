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

from megatron.bridge.models.nemotron import Nemotron3ModelProvider4B
from megatron.bridge.recipes.nemotron.nemotron3_4b import model_config, pretrain_config
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestModelConfig:
    """Test cases for the model_config function."""

    def test_model_config_default_parameters(self):
        """Test model_config with default parameters."""
        config = model_config()

        assert isinstance(config, Nemotron3ModelProvider4B)
        assert config.tensor_model_parallel_size == 2
        assert config.pipeline_model_parallel_size == 1
        assert config.pipeline_dtype == torch.bfloat16
        assert config.virtual_pipeline_model_parallel_size is None
        assert config.context_parallel_size == 1
        assert config.sequence_parallel is False


@pytest.mark.unit
class TestPretrainConfig:
    """Test cases for the pretrain_config function."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters (mock mode)."""
        config = pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, Nemotron3ModelProvider4B)

        # Check training configuration
        assert config.train.train_iters == 1_168_251
        assert config.train.global_batch_size == 32
        assert config.train.micro_batch_size == 2
        assert config.train.eval_interval == 2000
        assert config.train.eval_iters == 32

        # Check optimizer configuration
        assert config.optimizer.optimizer == "adam"
        assert config.optimizer.lr == 3e-4
        assert config.optimizer.min_lr == 3e-5
        assert config.optimizer.weight_decay == 0.1
        assert config.optimizer.bf16 is True
        assert config.optimizer.fp16 is False

        # Check dataset configuration (should be in mock mode by default)
        assert config.dataset.sequence_length == 4096
        assert config.dataset.split == "1,1,1"
        assert config.dataset.blend is None
        assert config.dataset.blend_per_split is None
