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
import pytest
import torch

from megatron.bridge.models.kimi import KimiK2Provider
from megatron.bridge.recipes.kimi.kimi_k2 import model_config, pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class TestKimiK2ModelConfig:
    """Test cases for model_config function."""

    def test_model_config_default_values(self):
        """Test model_config with default parameters."""
        cfg = model_config()

        # Check it returns a KimiK2Provider instance
        assert isinstance(cfg, KimiK2Provider)

        # Check key parallelism settings
        assert cfg.tensor_model_parallel_size == 2
        assert cfg.pipeline_model_parallel_size == 16
        assert cfg.expert_model_parallel_size == 32
        assert cfg.sequence_parallel is True

        # Check key settings
        assert cfg.recompute_granularity == "selective"
        assert cfg.moe_permute_fusion is True
        assert cfg.apply_rope_fusion is False

    def test_model_config_custom_parallelism(self):
        """Test model_config with custom parallelism settings."""
        cfg = model_config(
            tensor_parallelism=4,
            pipeline_parallelism=8,
            expert_parallelism=16,
            sequence_parallelism=False,
        )

        assert cfg.tensor_model_parallel_size == 4
        assert cfg.pipeline_model_parallel_size == 8
        assert cfg.expert_model_parallel_size == 16
        assert cfg.sequence_parallel is False

    def test_model_config_recomputation_and_fusion(self):
        """Test model_config with recomputation and fusion settings."""
        cfg = model_config(
            recompute_granularity="full",
            recompute_method="block",
            apply_rope_fusion=True,
        )

        assert cfg.recompute_granularity == "full"
        assert cfg.recompute_method == "block"
        assert cfg.apply_rope_fusion is True

    def test_model_config_deepep(self):
        """Test model_config with DeePEP enabled."""
        cfg = model_config(enable_deepep=True)

        assert cfg.moe_token_dispatcher_type == "flex"
        assert cfg.moe_enable_deepep is True
        assert cfg.moe_shared_expert_overlap is False

    def test_model_config_pipeline_layouts(self):
        """Test pipeline layouts for various PP/VP combinations."""
        # PP=1, VP=1 should have no layout
        cfg = model_config(pipeline_parallelism=1, virtual_pipeline_parallelism=1)
        assert cfg.pipeline_model_parallel_layout is None

        # PP=16, VP=1 should have a specific layout
        cfg = model_config(pipeline_parallelism=16, virtual_pipeline_parallelism=1)
        expected_layout = [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]]
        assert cfg.pipeline_model_parallel_layout == expected_layout

        # PP=8, VP=2 should have a specific layout
        cfg = model_config(pipeline_parallelism=8, virtual_pipeline_parallelism=2)
        expected_layout = [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]]
        assert cfg.pipeline_model_parallel_layout == expected_layout

    def test_model_config_invalid_pp_vp_combination(self):
        """Test that invalid PP/VP combinations raise ValueError."""
        with pytest.raises(ValueError, match="Invalid PP and VP size"):
            model_config(pipeline_parallelism=3, virtual_pipeline_parallelism=1)


class TestKimiK2PretrainConfig:
    """Test cases for pretrain_config function."""

    def test_pretrain_config_basic_structure(self):
        """Test that pretrain_config returns a valid ConfigContainer."""
        cfg = pretrain_config(
            name="test_kimi",
            dir="/tmp/test_output",
            mock=True,
            train_iters=100,
            global_batch_size=8,
            micro_batch_size=1,
            seq_length=128,
        )

        # Check it returns a ConfigContainer with all required components
        assert isinstance(cfg, ConfigContainer)
        assert isinstance(cfg.model, KimiK2Provider)
        assert cfg.train is not None
        assert cfg.optimizer is not None
        assert cfg.scheduler is not None
        assert cfg.dataset is not None
        assert cfg.tokenizer is not None
        assert cfg.checkpoint is not None

        # Check training settings
        assert cfg.train.train_iters == 100
        assert cfg.train.global_batch_size == 8
        assert cfg.train.micro_batch_size == 1

    def test_pretrain_config_optimizer_adam(self):
        """Test optimizer configuration for Adam."""
        cfg = pretrain_config(
            name="test",
            mock=True,
            optimizer_type="adam",
            lr=5e-4,
        )

        # Check optimizer has precision-aware settings
        assert cfg.optimizer.use_precision_aware_optimizer is True
        assert cfg.optimizer.main_params_dtype == torch.float32
        assert cfg.scheduler is not None

    def test_pretrain_config_optimizer_muon(self):
        """Test optimizer configuration for Muon."""
        cfg = pretrain_config(
            name="test",
            mock=True,
            optimizer_type="muon",
        )

        # Check DDP settings for Muon
        assert cfg.ddp.overlap_param_gather is False
        assert cfg.ddp.use_distributed_optimizer is False

    def test_pretrain_config_optimizer_invalid(self):
        """Test that invalid optimizer type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid optimizer type"):
            pretrain_config(
                name="test",
                mock=True,
                optimizer_type="invalid_optimizer",
            )

    def test_pretrain_config_dataset_and_tokenizer(self):
        """Test dataset and tokenizer configuration."""
        cfg = pretrain_config(name="test", mock=True, seq_length=4096)

        assert cfg.dataset.sequence_length == 4096
        assert cfg.dataset.data_sharding is True
        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
        assert cfg.tokenizer.vocab_size == 163840

    def test_pretrain_config_output_directories(self):
        """Test that output directories are properly configured."""
        cfg = pretrain_config(
            name="my_experiment",
            dir="/custom/output/path",
            mock=True,
        )

        checkpoint_dir = os.path.join("/custom/output/path", "my_experiment", "checkpoints")
        tensorboard_dir = os.path.join("/custom/output/path", "my_experiment", "tb_logs")

        assert cfg.checkpoint.save == checkpoint_dir
        assert cfg.logger.tensorboard_dir == tensorboard_dir

    def test_pretrain_config_mixed_precision(self):
        """Test mixed precision configuration."""
        cfg = pretrain_config(name="test", mock=True)

        assert isinstance(cfg.mixed_precision, MixedPrecisionConfig)
        assert cfg.mixed_precision.bf16 is True
        assert cfg.mixed_precision.params_dtype == torch.bfloat16

        # Test custom precision
        custom_precision = MixedPrecisionConfig(
            bf16=False,
            fp16=True,
            params_dtype=torch.float16,
        )
        cfg_custom = pretrain_config(name="test", mock=True, precision_config=custom_precision)
        assert cfg_custom.mixed_precision.fp16 is True

    def test_pretrain_config_parallelism_settings(self):
        """Test parallelism configuration."""
        cfg = pretrain_config(
            name="test",
            mock=True,
            tensor_parallelism=4,
            pipeline_parallelism=8,
            expert_parallelism=16,
        )

        assert cfg.model.tensor_model_parallel_size == 4
        assert cfg.model.pipeline_model_parallel_size == 8
        assert cfg.model.expert_model_parallel_size == 16

    def test_pretrain_config_special_features(self):
        """Test special features like RoPE fusion and DeePEP."""
        # Test RoPE fusion
        cfg_rope = pretrain_config(name="test", mock=True, apply_rope_fusion=True)
        assert cfg_rope.model.apply_rope_fusion is True
        assert cfg_rope.dist.enable_megatron_core_experimental is True

        # Test DeePEP
        cfg_deepep = pretrain_config(name="test", mock=True, enable_deepep=True)
        assert cfg_deepep.model.moe_token_dispatcher_type == "flex"
        assert cfg_deepep.model.moe_enable_deepep is True

