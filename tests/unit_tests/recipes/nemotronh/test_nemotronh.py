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

import os
import tempfile
from unittest.mock import patch

import pytest

from megatron.bridge.models.nemotronh import (
    NemotronHModel4BProvider,
    NemotronHModel8BProvider,
    NemotronHModel47BProvider,
    NemotronHModel56BProvider,
)
from megatron.bridge.recipes.nemotronh import (
    nemotronh_4b_pretrain_config,
    nemotronh_8b_pretrain_config,
    nemotronh_47b_pretrain_config,
    nemotronh_56b_pretrain_config,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


@pytest.mark.unit
class TestNemotronH4B:
    """Test cases for NemotronH 4B recipe."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters (mock mode)."""
        config = nemotronh_4b_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, NemotronHModel4BProvider)

        # Check model configuration defaults
        assert config.model.tensor_model_parallel_size == 1
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is False

        # Check training configuration
        assert config.train.train_iters == 1_168_251
        assert config.train.global_batch_size == 768
        assert config.train.micro_batch_size == 1

        # Check dataset configuration (should be in mock mode)
        assert config.dataset.sequence_length == 8192
        assert config.dataset.split == "1,1,1"

        # Check tokenizer (default is NullTokenizer for pretraining)
        assert config.tokenizer.tokenizer_type == "NullTokenizer"
        assert config.tokenizer.tokenizer_model is None

        # Check comm overlap
        assert config.comm_overlap is not None
        assert config.comm_overlap.tp_comm_overlap is True

    def test_pretrain_config_custom_parallelism(self):
        """Test pretrain_config with custom parallelism."""
        config = nemotronh_4b_pretrain_config(
            tensor_parallelism=4,
            pipeline_parallelism=2,
            context_parallelism=8,
            sequence_parallelism=True,
        )

        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 2
        assert config.model.context_parallel_size == 8
        assert config.model.sequence_parallel is True

    def test_pretrain_config_with_data_paths(self):
        """Test pretrain_config with data paths provided."""
        data_paths = ["/path/to/data1", "/path/to/data2", "/path/to/data3"]
        config = nemotronh_4b_pretrain_config(data_paths=data_paths)

        assert config.dataset.split == "9999,8,2"
        assert config.dataset.blend is not None

    @pytest.mark.parametrize("precision", ["fp16_mixed", "bf16_mixed"])
    def test_precision_recipes(self, precision):
        """Test precision configuration."""
        cfg = nemotronh_4b_pretrain_config(precision_config=precision)
        assert cfg.mixed_precision == precision

    def test_huggingface_tokenizer(self):
        """Test with HuggingFace tokenizer instead of NullTokenizer."""
        cfg = nemotronh_4b_pretrain_config(use_null_tokenizer=False)
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model == "nvidia/Nemotron-H-4B-Base-8K"
        assert cfg.tokenizer.vocab_size is None


@pytest.mark.unit
class TestNemotronH8B:
    """Test cases for NemotronH 8B recipe."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters (mock mode)."""
        config = nemotronh_8b_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, NemotronHModel8BProvider)

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

    def test_pretrain_config_custom_parallelism(self):
        """Test pretrain_config with custom parallelism."""
        config = nemotronh_8b_pretrain_config(
            tensor_parallelism=4,
            pipeline_parallelism=2,
            sequence_parallelism=False,
        )

        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 2
        assert config.model.sequence_parallel is False


@pytest.mark.unit
class TestNemotronH47B:
    """Test cases for NemotronH 47B recipe."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters (mock mode)."""
        config = nemotronh_47b_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, NemotronHModel47BProvider)

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

        # Check comm overlap is not set by default for 47B
        # (due to issues with fp8 current scaling)
        assert config.comm_overlap is None

    def test_pretrain_config_custom_parallelism(self):
        """Test pretrain_config with custom parallelism."""
        config = nemotronh_47b_pretrain_config(
            tensor_parallelism=4,
            pipeline_parallelism=2,
        )

        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 2


@pytest.mark.unit
class TestNemotronH56B:
    """Test cases for NemotronH 56B recipe."""

    def test_pretrain_config_default_parameters(self):
        """Test pretrain_config with default parameters (mock mode)."""
        config = nemotronh_56b_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, NemotronHModel56BProvider)

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

        # Check comm overlap is not set by default for 56B
        assert config.comm_overlap is None

    def test_pretrain_config_custom_parallelism(self):
        """Test pretrain_config with custom parallelism."""
        config = nemotronh_56b_pretrain_config(
            tensor_parallelism=4,
            pipeline_parallelism=2,
        )

        assert config.model.tensor_model_parallel_size == 4
        assert config.model.pipeline_model_parallel_size == 2


@pytest.mark.unit
class TestNemotronHCommon:
    """Test cases common to all NemotronH variants."""

    @pytest.mark.parametrize(
        "recipe_fn,provider_cls",
        [
            (nemotronh_4b_pretrain_config, NemotronHModel4BProvider),
            (nemotronh_8b_pretrain_config, NemotronHModel8BProvider),
            (nemotronh_47b_pretrain_config, NemotronHModel47BProvider),
            (nemotronh_56b_pretrain_config, NemotronHModel56BProvider),
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
    def test_custom_training_parameters(self, recipe_fn):
        """Test custom training parameters across all variants."""
        config = recipe_fn(
            train_iters=10000,
            global_batch_size=256,
            micro_batch_size=2,
            seq_length=4096,
            lr=1e-4,
            min_lr=1e-5,
            lr_warmup_iters=1000,
        )

        assert config.train.train_iters == 10000
        assert config.train.global_batch_size == 256
        assert config.train.micro_batch_size == 2
        assert config.dataset.sequence_length == 4096
        assert config.optimizer.lr == 1e-4
        assert config.optimizer.min_lr == 1e-5

    @pytest.mark.parametrize(
        "recipe_fn",
        [
            nemotronh_4b_pretrain_config,
            nemotronh_8b_pretrain_config,
            nemotronh_47b_pretrain_config,
            nemotronh_56b_pretrain_config,
        ],
    )
    def test_with_custom_directory(self, recipe_fn):
        """Test custom directory configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = recipe_fn(dir=temp_dir, name="test_run")

            expected_run_dir = os.path.join(temp_dir, "test_run")
            expected_checkpoint_dir = os.path.join(expected_run_dir, "checkpoints")
            expected_tensorboard_dir = os.path.join(expected_run_dir, "tb_logs")

            assert config.checkpoint.save == expected_checkpoint_dir
            assert config.logger.tensorboard_dir == expected_tensorboard_dir

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
    def test_custom_comm_overlap(self, recipe_fn):
        """Test custom CommOverlapConfig."""
        custom_overlap = CommOverlapConfig(
            tp_comm_overlap=False,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=50,
            data_parallel_size=1,
        )
        config = recipe_fn(comm_overlap_config=custom_overlap)

        assert config.comm_overlap is not None
        assert config.comm_overlap.tp_comm_overlap is False

    @pytest.mark.parametrize(
        "recipe_fn",
        [
            nemotronh_4b_pretrain_config,
            nemotronh_8b_pretrain_config,
            nemotronh_47b_pretrain_config,
            nemotronh_56b_pretrain_config,
        ],
    )
    def test_with_train_valid_test_paths(self, recipe_fn):
        """Test with separate train/valid/test paths."""
        config = recipe_fn(
            train_data_path=["/path/to/train1", "/path/to/train2"],
            valid_data_path=["/path/to/valid1", "/path/to/valid2"],
            test_data_path=["/path/to/test1", "/path/to/test2"],
        )

        assert config.dataset.split is None
        assert config.dataset.blend is None
        assert config.dataset.blend_per_split is not None

    @pytest.mark.parametrize(
        "recipe_fn",
        [
            nemotronh_4b_pretrain_config,
            nemotronh_8b_pretrain_config,
            nemotronh_47b_pretrain_config,
            nemotronh_56b_pretrain_config,
        ],
    )
    @patch("megatron.bridge.recipes.utils.dataset_utils.get_blend_and_blend_per_split")
    def test_fallback_to_mock_when_no_weights(self, mock_get_blend, recipe_fn):
        """Test fallback to mock when no weights are returned."""
        mock_get_blend.return_value = (None, None)

        config = recipe_fn(data_paths=["/some/path"])

        assert config.dataset.blend is None
        assert config.dataset.blend_per_split is None
        assert config.dataset.split == "1,1,1"
