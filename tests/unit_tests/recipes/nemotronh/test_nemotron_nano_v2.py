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

"""
Unit tests for Nemotron Nano v2 pretrain recipe configurations.

Pretrain configs use the parameterless API - they return a fixed ConfigContainer
with default settings. These tests verify the default configurations are correct.
"""

import pytest

from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.recipes.nemotronh import (
    nemotron_nano_9b_v2_pretrain_config,
    nemotron_nano_12b_v2_pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestNemotronNano9Bv2:
    """Test cases for Nemotron Nano 9B v2 recipe."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters."""
        config = nemotron_nano_9b_v2_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, MambaModelProvider)

        # Check model configuration defaults
        assert config.model.tensor_model_parallel_size == 2
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is True

        # Check training configuration
        assert config.train.train_iters == 1_168_251
        assert config.train.global_batch_size == 768
        assert config.train.micro_batch_size == 1

        # Check dataset configuration
        assert config.dataset.seq_length == 8192
        assert config.dataset.split == "9999,8,2"

        # Check tokenizer (default is NullTokenizer for pretraining)
        assert config.tokenizer.tokenizer_type == "NullTokenizer"
        assert config.tokenizer.tokenizer_model is None

        # Check precision
        assert config.mixed_precision == "bf16_mixed"

        # Check comm overlap
        assert config.comm_overlap is not None
        assert config.comm_overlap.tp_comm_overlap is True


@pytest.mark.unit
class TestNemotronNano12Bv2:
    """Test cases for Nemotron Nano 12B v2 recipe."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters."""
        config = nemotron_nano_12b_v2_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, MambaModelProvider)

        # Check model configuration defaults
        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is True

        # Check tokenizer (default is NullTokenizer for pretraining)
        assert config.tokenizer.tokenizer_type == "NullTokenizer"
        assert config.tokenizer.tokenizer_model is None

        # Check precision config (uses FP8)
        assert config.mixed_precision == "nanov2_bf16_with_fp8_current_scaling_mixed"

        # Check logger config
        assert config.logger.log_interval == 10

        # Check comm overlap is not set by default for 12B v2
        assert config.comm_overlap is None


@pytest.mark.unit
class TestNemotronNanoV2Common:
    """Test cases common to all Nemotron Nano v2 variants."""

    @pytest.mark.parametrize(
        "recipe_fn,provider_cls",
        [
            (nemotron_nano_9b_v2_pretrain_config, MambaModelProvider),
            (nemotron_nano_12b_v2_pretrain_config, MambaModelProvider),
        ],
    )
    def test_config_container_structure(self, recipe_fn, provider_cls):
        """Test that all configs return proper ConfigContainer with correct model provider."""
        config = recipe_fn()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, provider_cls)

    @pytest.mark.parametrize(
        "recipe_fn",
        [
            nemotron_nano_9b_v2_pretrain_config,
            nemotron_nano_12b_v2_pretrain_config,
        ],
    )
    def test_ddp_configuration(self, recipe_fn):
        """Test distributed data parallel configuration."""
        config = recipe_fn()

        assert config.ddp.check_for_nan_in_grad is True
        assert config.ddp.grad_reduce_in_fp32 is True
        assert config.ddp.overlap_grad_reduce is True
        assert config.ddp.overlap_param_gather is False
        assert config.ddp.use_distributed_optimizer is True

    @pytest.mark.parametrize(
        "recipe_fn",
        [
            nemotron_nano_9b_v2_pretrain_config,
            nemotron_nano_12b_v2_pretrain_config,
        ],
    )
    def test_tokenizer_defaults(self, recipe_fn):
        """Test that all pretrain configs use NullTokenizer by default."""
        config = recipe_fn()

        assert config.tokenizer.tokenizer_type == "NullTokenizer"
        assert config.tokenizer.tokenizer_model is None
        assert config.tokenizer.vocab_size is not None
