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
Unit tests for NemotronH pretrain recipe configurations.

Pretrain configs use the parameterless API - they return a fixed ConfigContainer
with default settings. These tests verify the default configurations are correct.
"""

import pytest

from megatron.bridge.models.nemotronh import (
    NemotronHModelProvider4B,
    NemotronHModelProvider8B,
    NemotronHModelProvider47B,
    NemotronHModelProvider56B,
)
from megatron.bridge.recipes.nemotronh import (
    nemotronh_4b_pretrain_config,
    nemotronh_8b_pretrain_config,
    nemotronh_47b_pretrain_config,
    nemotronh_56b_pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestNemotronH4B:
    """Test cases for NemotronH 4B recipe."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters."""
        config = nemotronh_4b_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, NemotronHModelProvider4B)

        # Check model configuration defaults
        assert config.model.tensor_model_parallel_size == 1
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is False

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

        # Check comm overlap
        assert config.comm_overlap is not None
        assert config.comm_overlap.tp_comm_overlap is True
        assert config.comm_overlap.tp_comm_bootstrap_backend == "nccl"


@pytest.mark.unit
class TestNemotronH8B:
    """Test cases for NemotronH 8B recipe."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters."""
        config = nemotronh_8b_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, NemotronHModelProvider8B)

        # Check model configuration defaults
        assert config.model.tensor_model_parallel_size == 2
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is True

        # Check tokenizer (default is NullTokenizer for pretraining)
        assert config.tokenizer.tokenizer_type == "NullTokenizer"
        assert config.tokenizer.tokenizer_model is None

        # Check comm overlap
        assert config.comm_overlap is not None
        assert config.comm_overlap.tp_comm_overlap is True
        assert config.comm_overlap.tp_comm_bootstrap_backend == "nccl"


@pytest.mark.unit
class TestNemotronH47B:
    """Test cases for NemotronH 47B recipe."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters."""
        config = nemotronh_47b_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, NemotronHModelProvider47B)

        # Check model configuration defaults
        assert config.model.tensor_model_parallel_size == 8
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is True

        # Check tokenizer (default is NullTokenizer for pretraining)
        assert config.tokenizer.tokenizer_type == "NullTokenizer"
        assert config.tokenizer.tokenizer_model is None

        # Check precision config
        assert config.mixed_precision == "nemotron_h_bf16_with_fp8_current_scaling_mixed"

        # Check logger config
        assert config.logger.log_interval == 10

        # Check comm overlap
        assert config.comm_overlap is not None
        assert config.comm_overlap.tp_comm_overlap is True
        assert config.comm_overlap.tp_comm_bootstrap_backend == "nccl"


@pytest.mark.unit
class TestNemotronH56B:
    """Test cases for NemotronH 56B recipe."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters."""
        config = nemotronh_56b_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, NemotronHModelProvider56B)

        # Check model configuration defaults
        assert config.model.tensor_model_parallel_size == 8
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is True

        # Check tokenizer (default is NullTokenizer for pretraining)
        assert config.tokenizer.tokenizer_type == "NullTokenizer"
        assert config.tokenizer.tokenizer_model is None

        # Check precision config
        assert config.mixed_precision == "nemotron_h_bf16_with_fp8_current_scaling_mixed"

        # Check logger config
        assert config.logger.log_interval == 10

        # Check comm overlap
        assert config.comm_overlap is not None
        assert config.comm_overlap.tp_comm_overlap is True
        assert config.comm_overlap.tp_comm_bootstrap_backend == "nccl"


@pytest.mark.unit
class TestNemotronHCommon:
    """Test cases common to all NemotronH variants."""

    @pytest.mark.parametrize(
        "recipe_fn,provider_cls",
        [
            (nemotronh_4b_pretrain_config, NemotronHModelProvider4B),
            (nemotronh_8b_pretrain_config, NemotronHModelProvider8B),
            (nemotronh_47b_pretrain_config, NemotronHModelProvider47B),
            (nemotronh_56b_pretrain_config, NemotronHModelProvider56B),
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
            nemotronh_4b_pretrain_config,
            nemotronh_8b_pretrain_config,
            nemotronh_47b_pretrain_config,
            nemotronh_56b_pretrain_config,
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
            nemotronh_4b_pretrain_config,
            nemotronh_8b_pretrain_config,
            nemotronh_47b_pretrain_config,
            nemotronh_56b_pretrain_config,
        ],
    )
    def test_tokenizer_defaults(self, recipe_fn):
        """Test that all pretrain configs use NullTokenizer by default."""
        config = recipe_fn()

        assert config.tokenizer.tokenizer_type == "NullTokenizer"
        assert config.tokenizer.tokenizer_model is None
        assert config.tokenizer.vocab_size is not None
