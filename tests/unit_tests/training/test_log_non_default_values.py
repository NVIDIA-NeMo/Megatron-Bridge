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

"""Tests for the log_non_default_values functionality in config.py."""

from unittest.mock import patch

from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig as MCoreDistributedDataParallelConfig,
)
from megatron.core.optimizer.optimizer_config import OptimizerConfig as MCoreOptimizerConfig
from megatron.core.transformer.transformer_config import TransformerConfig as MCoreTransformerConfig

from megatron.bridge.models.deepseek.deepseek_provider import DeepSeekModelProvider
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    GPTDatasetConfig,
    LoggerConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    _get_key_config_values,
    _get_mcore_transformer_parent,
    _get_non_default_values,
)


class TestGetMcoreTransformerParent:
    """Tests for _get_mcore_transformer_parent function."""

    def test_gpt_provider_returns_transformer_config(self):
        """GPTModelProvider should return MCoreTransformerConfig as parent."""
        config = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            seq_length=512,
        )
        parent = _get_mcore_transformer_parent(config)
        assert parent is MCoreTransformerConfig

    def test_deepseek_provider_returns_mla_transformer_config(self):
        """DeepSeekModelProvider should return MCoreMLATransformerConfig as parent."""
        from megatron.core.transformer.transformer_config import (
            MLATransformerConfig as MCoreMLATransformerConfig,
        )

        config = DeepSeekModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            seq_length=512,
        )
        parent = _get_mcore_transformer_parent(config)
        assert parent is MCoreMLATransformerConfig


class TestGetNonDefaultValues:
    """Tests for _get_non_default_values function."""

    def test_detects_non_default_optimizer_values(self):
        """Should detect values that differ from Mcore OptimizerConfig defaults."""
        # Create optimizer with non-default values
        optimizer = OptimizerConfig(
            lr=0.001,
            adam_beta2=0.95,  # Mcore default is 0.999
            weight_decay=0.1,  # Mcore default is 0.01
        )

        non_defaults = _get_non_default_values(optimizer, MCoreOptimizerConfig)

        assert "adam_beta2" in non_defaults
        assert non_defaults["adam_beta2"] == (0.95, 0.999)

        assert "weight_decay" in non_defaults
        assert non_defaults["weight_decay"] == (0.1, 0.01)

    def test_does_not_include_matching_defaults(self):
        """Should not include values that match Mcore defaults."""
        # Create optimizer with default adam_eps value
        optimizer = OptimizerConfig(
            lr=0.001,
            adam_eps=1e-8,  # Matches Mcore default
        )

        non_defaults = _get_non_default_values(optimizer, MCoreOptimizerConfig)

        # adam_eps should not be in non_defaults since it matches
        assert "adam_eps" not in non_defaults

    def test_detects_non_default_ddp_values(self):
        """Should detect non-default values in DDP config."""
        ddp = DistributedDataParallelConfig(
            use_distributed_optimizer=True,  # Mcore default is False
            overlap_grad_reduce=True,  # Mcore default is False
        )

        non_defaults = _get_non_default_values(ddp, MCoreDistributedDataParallelConfig)

        assert "use_distributed_optimizer" in non_defaults
        assert non_defaults["use_distributed_optimizer"] == (True, False)

        assert "overlap_grad_reduce" in non_defaults
        assert non_defaults["overlap_grad_reduce"] == (True, False)

    def test_handles_model_config(self):
        """Should detect non-default values in model config."""
        model = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            seq_length=512,
            add_bias_linear=False,  # Mcore default is True
            hidden_dropout=0.0,  # Mcore default is 0.1
        )

        non_defaults = _get_non_default_values(model, MCoreTransformerConfig)

        assert "add_bias_linear" in non_defaults
        assert non_defaults["add_bias_linear"] == (False, True)

        assert "hidden_dropout" in non_defaults
        assert non_defaults["hidden_dropout"] == (0.0, 0.1)

    def test_skips_private_fields(self):
        """Should skip fields that start with underscore."""
        optimizer = OptimizerConfig(lr=0.001)

        non_defaults = _get_non_default_values(optimizer, MCoreOptimizerConfig)

        # No private fields should be in the result
        for field_name in non_defaults:
            assert not field_name.startswith("_")


class TestGetKeyConfigValues:
    """Tests for _get_key_config_values function."""

    def test_extracts_training_config_values(self):
        """Should extract key values from TrainingConfig."""
        train = TrainingConfig(
            global_batch_size=128,
            train_iters=1000,
            micro_batch_size=4,
        )

        values = _get_key_config_values(train)

        assert values["global_batch_size"] == 128
        assert values["train_iters"] == 1000
        assert values["micro_batch_size"] == 4

    def test_skips_none_values(self):
        """Should skip fields that are None."""
        train = TrainingConfig(
            global_batch_size=128,
            train_iters=1000,
        )

        values = _get_key_config_values(train)

        # Check that no None values are included
        for value in values.values():
            assert value is not None

    def test_skips_private_fields(self):
        """Should skip fields that start with underscore."""
        scheduler = SchedulerConfig(lr_decay_style="cosine")

        values = _get_key_config_values(scheduler)

        for field_name in values:
            assert not field_name.startswith("_")

    def test_extracts_checkpoint_config_values(self):
        """Should extract key values from CheckpointConfig."""
        checkpoint = CheckpointConfig(
            ckpt_format="torch_dist",
            save_interval=500,
            async_save=True,
        )

        values = _get_key_config_values(checkpoint)

        assert values["ckpt_format"] == "torch_dist"
        assert values["save_interval"] == 500
        assert values["async_save"] is True


class TestLogNonDefaultValues:
    """Tests for ConfigContainer.log_non_default_values method."""

    def _create_minimal_config_container(self, model_provider=None) -> ConfigContainer:
        """Create a minimal ConfigContainer for testing."""
        if model_provider is None:
            model_provider = GPTModelProvider(
                num_layers=2,
                hidden_size=128,
                num_attention_heads=4,
                seq_length=512,
                add_bias_linear=False,  # Non-default
            )

        return ConfigContainer(
            model=model_provider,
            optimizer=OptimizerConfig(
                lr=0.001,
                adam_beta2=0.95,  # Non-default (Mcore default is 0.999)
            ),
            scheduler=SchedulerConfig(lr_decay_style="cosine"),
            train=TrainingConfig(
                global_batch_size=64,
                train_iters=500,
            ),
            ddp=DistributedDataParallelConfig(
                overlap_grad_reduce=True,  # Non-default
            ),
            checkpoint=CheckpointConfig(ckpt_format="torch_dist"),
            logger=LoggerConfig(),
            tokenizer=TokenizerConfig(),
            rng=RNGConfig(),
            dataset=GPTDatasetConfig(
                random_seed=1234,
                seq_length=512,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
            ),
        )

    @patch("megatron.bridge.training.config.print_rank_0")
    def test_logs_non_default_values(self, mock_print_rank_0):
        """Should log non-default values compared to Mcore."""
        cfg = self._create_minimal_config_container()

        cfg.log_non_default_values()

        # Verify print_rank_0 was called
        mock_print_rank_0.assert_called_once()

        # Get the logged output
        log_output = mock_print_rank_0.call_args[0][0]

        # Verify header is present
        assert "Configuration Summary (Non-Default Values vs Megatron Core)" in log_output

        # Verify optimizer non-defaults are logged
        assert "[optimizer] Non-default values" in log_output
        assert "adam_beta2: 0.95" in log_output
        assert "Mcore default: 0.999" in log_output

        # Verify ddp non-defaults are logged
        assert "[ddp] Non-default values" in log_output
        assert "overlap_grad_reduce: True" in log_output
        assert "Mcore default: False" in log_output

        # Verify model non-defaults are logged
        assert "[model] Non-default values" in log_output
        assert "add_bias_linear: False" in log_output
        assert "Mcore default: True" in log_output

    @patch("megatron.bridge.training.config.print_rank_0")
    def test_logs_other_config_values(self, mock_print_rank_0):
        """Should log key values from non-Mcore configs."""
        cfg = self._create_minimal_config_container()

        cfg.log_non_default_values()

        log_output = mock_print_rank_0.call_args[0][0]

        # Verify other configs section is present
        assert "Other Configuration Values:" in log_output

        # Verify train config values
        assert "[train]:" in log_output
        assert "global_batch_size: 64" in log_output
        assert "train_iters: 500" in log_output

        # Verify scheduler config values
        assert "[scheduler]:" in log_output

    @patch("megatron.bridge.training.config.print_rank_0")
    def test_handles_deepseek_model_correctly(self, mock_print_rank_0):
        """Should use MLATransformerConfig for DeepSeek models."""
        deepseek_model = DeepSeekModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            seq_length=512,
        )

        cfg = self._create_minimal_config_container(model_provider=deepseek_model)

        cfg.log_non_default_values()

        log_output = mock_print_rank_0.call_args[0][0]

        # Should use MLATransformerConfig for comparison
        assert "MLATransformerConfig" in log_output

    @patch("megatron.bridge.training.config.print_rank_0")
    def test_adam_eps_not_logged_when_default(self, mock_print_rank_0):
        """adam_eps should not appear in logs when set to Mcore default (1e-8)."""
        cfg = ConfigContainer(
            model=GPTModelProvider(
                num_layers=2,
                hidden_size=128,
                num_attention_heads=4,
                seq_length=512,
            ),
            optimizer=OptimizerConfig(
                lr=0.001,
                adam_eps=1e-8,  # Mcore default
            ),
            scheduler=SchedulerConfig(lr_decay_style="cosine"),
            train=TrainingConfig(global_batch_size=64, train_iters=500),
            ddp=DistributedDataParallelConfig(),
            checkpoint=CheckpointConfig(ckpt_format="torch_dist"),
            logger=LoggerConfig(),
            tokenizer=TokenizerConfig(),
            rng=RNGConfig(),
            dataset=GPTDatasetConfig(
                random_seed=1234,
                seq_length=512,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
            ),
        )

        cfg.log_non_default_values()

        log_output = mock_print_rank_0.call_args[0][0]

        # adam_eps should NOT be in the log since it matches Mcore default
        assert "adam_eps:" not in log_output
