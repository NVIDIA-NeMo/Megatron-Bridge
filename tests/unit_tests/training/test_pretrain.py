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

from unittest.mock import MagicMock, Mock, patch

from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.pretrain import pretrain
from tests.unit_tests.training.test_config import (
    create_test_checkpoint_config,
    create_test_config_container,
    create_test_gpt_config,
    restore_get_world_size_safe,
)


class ForwardFunctor:
    """Simple callable class used across tests."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return "ok"


class TestPretrainFunctorSupport:
    """Tests ensuring functor-style forward step works with pretrain."""

    @patch("megatron.bridge.training.pretrain.setup")
    @patch("megatron.bridge.training.pretrain.get_dataset_provider")
    @patch("megatron.bridge.training.pretrain.runtime_config_update")
    def test_pretrain_accepts_callable_functor(self, mock_runtime_update, mock_get_dataset_provider, mock_setup):
        gpt_model_cfg = create_test_gpt_config()
        checkpoint_cfg = create_test_checkpoint_config(save=None)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            checkpoint_config=checkpoint_cfg,
        )

        functor = ForwardFunctor()

        setup_output = MagicMock()
        setup_output.state = MagicMock()
        setup_output.state.cfg = container
        setup_output.state.train_state.do_train = True
        setup_output.state.train_state.step = 0
        setup_output.state.train_state.do_valid = False
        setup_output.state.train_state.do_test = False
        setup_output.model = MagicMock()
        setup_output.optimizer = MagicMock()
        setup_output.scheduler = MagicMock()
        setup_output.train_data_iterator = MagicMock()
        setup_output.valid_data_iterator = None
        setup_output.test_data_iterator = None
        setup_output.checkpointing_context = {}
        mock_setup.return_value = setup_output

        with patch("megatron.bridge.training.pretrain.train") as mock_train:
            try:
                pretrain(container, functor)
            finally:
                restore_get_world_size_safe(og_ws, cfg_mod)

            mock_runtime_update.assert_called_once_with(container)
            mock_get_dataset_provider.assert_called_once()
            mock_setup.assert_called_once()
            mock_train.assert_called_once()
            assert mock_train.call_args[0][0] is functor


class TestFinetuneFunctorSupport:
    """Complementary tests ensuring callable functors work with finetune."""

    def test_finetune_requires_checkpoints_functor(self):
        gpt_model_cfg = create_test_gpt_config()
        checkpoint_cfg = create_test_checkpoint_config(pretrained_checkpoint="/path/to/pretrained.ckpt")

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            checkpoint_config=checkpoint_cfg,
        )

        functor = ForwardFunctor()

        with patch("megatron.bridge.training.finetune.pretrain") as mock_pretrain:
            try:
                finetune(container, functor)
            finally:
                restore_get_world_size_safe(og_ws, cfg_mod)

            mock_pretrain.assert_called_once_with(container, functor)


class TestTrainMaybeInjectStateWithFunctor:
    """Integration test ensuring maybe_inject_state works with functors in train.step."""

    @patch("megatron.bridge.training.train.get_forward_backward_func")
    @patch("megatron.bridge.training.train.get_rerun_state_machine")
    @patch("megatron.bridge.training.train.maybe_inject_state")
    def test_train_step_wraps_functor(self, mock_maybe_inject_state, mock_get_rerun, mock_get_fwb):
        from megatron.bridge.training.train import train_step

        mock_state_machine = Mock()
        mock_state_machine.should_run_forward_backward.side_effect = [True, False]
        mock_state_machine.should_checkpoint_and_exit.return_value = (False, False, 0)
        mock_get_rerun.return_value = mock_state_machine

        def fake_forward_backward_func(**kwargs):
            return [{"loss": Mock(numel=lambda: 1, view=lambda *args, **kwargs: Mock(numel=lambda: 1))}]

        mock_get_fwb.return_value = fake_forward_backward_func

        mock_maybe_inject_state.side_effect = lambda func, state, num_fw_args=None: func

        functor = ForwardFunctor()

        model = [MagicMock()]
        optimizer = MagicMock()
        optimizer.step.return_value = (True, 1.0, None)
        optimizer.param_groups = [MagicMock(is_decoupled_lr=False, lr=0.001)]
        scheduler = MagicMock()

        global_state = MagicMock()
        global_state.cfg.train.decrease_batch_size_if_needed = False
        global_state.cfg.train.empty_unused_memory_level = 0
        global_state.cfg.train.micro_batch_size = 1
        global_state.cfg.data_parallel_size = 1
        global_state.cfg.optimizer.log_num_zeros_in_grad = False
        global_state.train_state.step = 0
        global_state.train_state.consumed_train_samples = 0
        global_state.train_state.floating_point_operations_so_far = 0.0
        global_state.train_state.skipped_train_samples = 0
        global_state.timers = MagicMock()
        global_state.straggler_timer = MagicMock()
        global_state.cfg.rerun_state_machine.check_for_nan_in_loss = False
        global_state.cfg.rerun_state_machine.check_for_spiky_loss = False

        loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros = train_step(
            functor,
            3,
            MagicMock(),
            model,
            optimizer,
            scheduler,
            global_state,
        )

        assert loss_dict == {}
        assert skipped_iter == 0
        assert should_checkpoint is False
        assert should_exit is False
        assert exit_code == 0
        assert grad_norm == 1.0
        assert num_zeros is None
        mock_maybe_inject_state.assert_called_once_with(functor, global_state, num_fw_args=3)
