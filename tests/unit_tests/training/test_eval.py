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

"""Tests for multiple validation sets support in evaluation."""

from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.transformer import MegatronModule

from megatron.bridge.training.config import ConfigContainer, LoggerConfig, TrainingConfig, ValidationConfig
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.state import GlobalState, TrainState


pytestmark = pytest.mark.unit


class TestValidationDataloaderCreation:
    """Test validation dataloader creation with decoupled configuration."""

    def test_validation_config_attributes_exist(self):
        """Test that ValidationConfig attributes exist and can be accessed."""
        from megatron.bridge.training.config import ValidationConfig

        val_config = ValidationConfig(
            val_micro_batch_size=8,
            val_global_batch_size=32,
            eval_iters=100,
            eval_interval=500,
        )

        assert hasattr(val_config, "val_micro_batch_size")
        assert hasattr(val_config, "val_global_batch_size")
        assert val_config.val_micro_batch_size == 8
        assert val_config.val_global_batch_size == 32

    def test_dataloader_config_validation_attributes(self):
        """Test that GPTDatasetConfig has validation-specific dataloader attributes."""
        from megatron.bridge.training.config import GPTDatasetConfig

        dataset_config = GPTDatasetConfig(
            random_seed=1234,
            sequence_length=512,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            val_num_workers=4,
            val_pin_memory=False,
            val_persistent_workers=False,
        )

        assert hasattr(dataset_config, "val_num_workers")
        assert hasattr(dataset_config, "val_pin_memory")
        assert hasattr(dataset_config, "val_persistent_workers")
        assert dataset_config.val_num_workers == 4
        assert dataset_config.val_pin_memory is False
        assert dataset_config.val_persistent_workers is False

    def test_multiple_validation_sets_flag(self):
        """Test that GPTDatasetConfig has multiple_validation_sets flag."""
        from megatron.bridge.training.config import GPTDatasetConfig

        dataset_config = GPTDatasetConfig(
            random_seed=1234,
            sequence_length=512,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            multiple_validation_sets=True,
        )

        assert hasattr(dataset_config, "multiple_validation_sets")
        assert dataset_config.multiple_validation_sets is True

    def test_logger_config_multi_val_naming(self):
        """Test that LoggerConfig has multiple_validation_sets_use_dataset_name field."""
        from megatron.bridge.training.config import LoggerConfig

        logger_config = LoggerConfig(multiple_validation_sets_use_dataset_name=True)
        assert logger_config.multiple_validation_sets_use_dataset_name is True

        logger_config2 = LoggerConfig(multiple_validation_sets_use_dataset_name=False)
        assert logger_config2.multiple_validation_sets_use_dataset_name is False


class TestEvaluateAndPrintResults:
    """Unit tests for evaluate_and_print_results function with multiple validation sets."""

    def _create_mock_global_state(self, multiple_validation_sets=False, blend_per_split=None):
        """Create a mock GlobalState for testing."""
        mock_state = Mock(spec=GlobalState)
        mock_state.train_state = Mock(spec=TrainState)
        mock_state.train_state.step = 100
        mock_state.train_state.consumed_train_samples = 1000

        mock_config = Mock(spec=ConfigContainer)
        mock_config.train = Mock(spec=TrainingConfig)
        mock_config.train.global_batch_size = 32
        mock_config.train.micro_batch_size = 8

        mock_config.validation = Mock(spec=ValidationConfig)
        mock_config.validation.eval_iters = 2
        mock_config.validation.val_micro_batch_size = 8
        mock_config.validation.val_global_batch_size = 32

        mock_config.data_parallel_size = 4
        mock_config.model = Mock()
        mock_config.model.seq_length = 512

        mock_config.logger = Mock(spec=LoggerConfig)
        mock_config.logger.log_validation_ppl_to_tensorboard = True
        mock_config.logger.multiple_validation_sets_use_dataset_name = True

        mock_dataset_config = Mock()
        mock_dataset_config.multiple_validation_sets = multiple_validation_sets
        mock_dataset_config.blend_per_split = blend_per_split
        mock_config.dataset = mock_dataset_config

        mock_state.cfg = mock_config

        mock_state.timers = Mock()
        mock_timer = Mock()
        mock_timer.start = Mock()
        mock_timer.stop = Mock()
        mock_timer.elapsed = Mock(return_value=1.0)
        mock_state.timers.return_value = mock_timer

        mock_state.tensorboard_logger = Mock()
        mock_state.wandb_logger = Mock()

        return mock_state

    def _create_mock_model(self):
        mock_model = Mock(spec=MegatronModule)
        mock_model.eval = Mock()
        mock_model.train = Mock()
        return [mock_model]

    def _create_mock_data_iterator(self, single=True):
        if single:
            return Mock()
        else:
            return [Mock(), Mock()]

    @patch("megatron.bridge.training.eval.evaluate")
    @patch("megatron.bridge.utils.common_utils.is_last_rank")
    @patch("megatron.bridge.utils.common_utils.print_rank_last")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    def test_evaluate_and_print_results_single_dataset(
        self, mock_get_rank, mock_world_size, _mock_print_rank_last, mock_is_last_rank, mock_evaluate
    ):
        """Test original single dataset behavior in evaluate_and_print_results."""
        mock_get_rank.return_value = 0
        mock_world_size.return_value = 1
        mock_is_last_rank.return_value = True
        mock_evaluate.return_value = (
            {"loss": torch.tensor(0.5)},
            None,
            False,
        )

        state = self._create_mock_global_state(multiple_validation_sets=False)
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator(single=True)
        forward_step_func = Mock()

        evaluate_and_print_results(
            state=state,
            prefix="test",
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            config=state.cfg,
            verbose=False,
            write_to_tensorboard=True,
        )

        mock_evaluate.assert_called_once()
        call_args = mock_evaluate.call_args
        assert call_args[0][0] == state
        assert call_args[0][2] == data_iterator
        assert call_args[0][3] == model
        assert call_args[0][5] == state.cfg

        state.tensorboard_logger.add_scalar.assert_called()
        state.wandb_logger.log.assert_called()

    @patch("megatron.bridge.training.eval.evaluate")
    @patch("megatron.bridge.utils.common_utils.is_last_rank")
    @patch("megatron.bridge.utils.common_utils.print_rank_last")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    def test_evaluate_and_print_results_multiple_datasets(
        self, mock_get_rank, mock_world_size, _mock_print_rank_last, mock_is_last_rank, mock_evaluate
    ):
        """Test new multiple datasets behavior in evaluate_and_print_results."""
        mock_get_rank.return_value = 0
        mock_world_size.return_value = 1
        mock_is_last_rank.return_value = True

        mock_evaluate.return_value = ({"loss": torch.tensor(0.5)}, None, False)

        blend_per_split = [
            (["train_paths"], None),
            (["val1", "val2"], None),
            (["test_paths"], None),
        ]
        state = self._create_mock_global_state(multiple_validation_sets=True, blend_per_split=blend_per_split)
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator(single=False)
        forward_step_func = Mock()

        evaluate_and_print_results(
            state=state,
            prefix="test",
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            config=state.cfg,
            verbose=False,
            write_to_tensorboard=True,
        )

        assert mock_evaluate.call_count == 2

        individual_calls = [
            call
            for call in state.tensorboard_logger.add_scalar.call_args_list
            if "validation val1" in str(call) or "validation val2" in str(call)
        ]
        assert len(individual_calls) > 0

        aggregated_calls = [
            call
            for call in state.tensorboard_logger.add_scalar.call_args_list
            if "validation (aggregated)" in str(call)
        ]
        assert len(aggregated_calls) > 0

    @patch("megatron.bridge.training.eval.evaluate")
    @patch("megatron.bridge.utils.common_utils.is_last_rank")
    @patch("megatron.bridge.utils.common_utils.print_rank_last")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    def test_evaluate_and_print_results_timelimit_handling(
        self, mock_get_rank, mock_world_size, _mock_print_rank_last, mock_is_last_rank, mock_evaluate
    ):
        """Test timelimit handling in evaluate_and_print_results."""
        mock_get_rank.return_value = 0
        mock_world_size.return_value = 1
        mock_is_last_rank.return_value = True
        mock_evaluate.return_value = (None, None, True)

        state = self._create_mock_global_state()
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator(single=True)
        forward_step_func = Mock()

        evaluate_and_print_results(
            state=state,
            prefix="test",
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            config=state.cfg,
            verbose=False,
            write_to_tensorboard=True,
        )

        mock_evaluate.assert_called_once()

        state.tensorboard_logger.add_scalar.assert_not_called()
        state.wandb_logger.log.assert_not_called()

    @patch("megatron.bridge.training.eval.evaluate")
    @patch("megatron.bridge.utils.common_utils.is_last_rank")
    @patch("megatron.bridge.utils.common_utils.print_rank_last")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    def test_evaluate_and_print_results_dataset_naming(
        self, mock_get_rank, mock_world_size, _mock_print_rank_last, mock_is_last_rank, mock_evaluate
    ):
        """Test dataset naming from blend_per_split configuration."""
        mock_get_rank.return_value = 0
        mock_world_size.return_value = 1
        mock_is_last_rank.return_value = True
        mock_evaluate.return_value = ({"loss": torch.tensor(0.5)}, None, False)

        blend_per_split = [
            (["train_paths"], None),
            (["val_dataset_1", "val_dataset_2"], None),
            (["test_paths"], None),
        ]
        state = self._create_mock_global_state(multiple_validation_sets=True, blend_per_split=blend_per_split)
        model = self._create_mock_model()
        data_iterator = self._create_mock_data_iterator(single=False)
        forward_step_func = Mock()

        evaluate_and_print_results(
            state=state,
            prefix="test",
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            config=state.cfg,
            verbose=False,
            write_to_tensorboard=True,
        )

        assert mock_evaluate.call_count == 2

        logging_calls = [str(call) for call in state.tensorboard_logger.add_scalar.call_args_list]
        val1_calls = [call for call in logging_calls if "val_dataset_1" in call]
        val2_calls = [call for call in logging_calls if "val_dataset_2" in call]

        assert len(val1_calls) > 0
        assert len(val2_calls) > 0
