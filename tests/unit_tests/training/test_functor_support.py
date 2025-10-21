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

"""Tests for functor support in forward step functions."""

import inspect
from functools import partial
from typing import Iterable
from unittest.mock import MagicMock, Mock, patch

import torch
from megatron.core.models.gpt import GPTModel

from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.train_utils import (
    maybe_inject_state,
    needs_global_state_injection,
)
from tests.unit_tests.training.test_config import (
    create_test_checkpoint_config,
    create_test_config_container,
    create_test_gpt_config,
    create_test_training_config,
    restore_get_world_size_safe,
)


class TwoArgForwardFunctor:
    """Functor with 2 arguments: (data_iterator, model)."""

    def __init__(self):
        self.call_count = 0
        self.last_args = None
        self.last_kwargs = None

    def __call__(self, data_iterator: Iterable, model: GPTModel) -> tuple[torch.Tensor, partial]:
        self.call_count += 1
        self.last_args = (data_iterator, model)
        self.last_kwargs = {}
        # Return mock tensor and loss function
        return torch.tensor([1.0]), partial(lambda x: x)


class ThreeArgForwardFunctor:
    """Functor with 3 arguments: (data_iterator, model, return_schedule_plan)."""

    def __init__(self):
        self.call_count = 0
        self.last_args = None
        self.last_kwargs = None

    def __call__(
        self, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, partial]:
        self.call_count += 1
        self.last_args = (data_iterator, model, return_schedule_plan)
        self.last_kwargs = {}
        # Return mock tensor and loss function
        return torch.tensor([1.0]), partial(lambda x: x)


class FourArgForwardFunctor:
    """Functor with 4 arguments: (state, data_iterator, model, return_schedule_plan)."""

    def __init__(self):
        self.call_count = 0
        self.last_args = None
        self.last_kwargs = None

    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model: GPTModel,
        return_schedule_plan: bool = False,
    ) -> tuple[torch.Tensor, partial]:
        self.call_count += 1
        self.last_args = (state, data_iterator, model, return_schedule_plan)
        self.last_kwargs = {}
        # Return mock tensor and loss function
        return torch.tensor([1.0]), partial(lambda x: x)


class StatefulForwardFunctor:
    """Functor that maintains state across calls."""

    def __init__(self, initial_loss: float = 1.0):
        self.initial_loss = initial_loss
        self.call_count = 0
        self.loss_history = []
        self.state_received = None

    def __call__(
        self,
        state: GlobalState,
        data_iterator: Iterable,
        model: GPTModel,
        return_schedule_plan: bool = False,
    ) -> tuple[torch.Tensor, partial]:
        self.call_count += 1
        self.state_received = state

        # Simulate decreasing loss over time
        current_loss = self.initial_loss * (0.9**self.call_count)
        self.loss_history.append(current_loss)

        loss_tensor = torch.tensor([current_loss])
        loss_function = partial(lambda x: loss_tensor)

        return loss_tensor, loss_function

    def get_average_loss(self) -> float | None:
        """Return average loss across all calls."""
        if not self.loss_history:
            return None
        return sum(self.loss_history) / len(self.loss_history)


class TestFunctorStateInjectionDetection:
    """Test that functors are correctly inspected for state injection needs."""

    def test_two_arg_functor_inspection(self):
        """Test that 2-arg functor doesn't need state injection."""
        functor = TwoArgForwardFunctor()
        needs_injection = needs_global_state_injection(functor)
        assert needs_injection is False  # No state parameter

    def test_three_arg_functor_inspection(self):
        """Test that 3-arg functor without state doesn't need injection."""
        functor = ThreeArgForwardFunctor()
        needs_injection = needs_global_state_injection(functor)
        assert needs_injection is False  # No state parameter

    def test_four_arg_functor_inspection(self):
        """Test that 4-arg functor with state needs injection."""
        functor = FourArgForwardFunctor()
        needs_injection = needs_global_state_injection(functor)
        assert needs_injection is True  # Has 'state' parameter name

    def test_functor_signature_inspection_works(self):
        """Test that inspect.signature works correctly on functors."""
        functor = FourArgForwardFunctor()
        signature = inspect.signature(functor)
        params = list(signature.parameters.keys())
        assert params == ["state", "data_iterator", "model", "return_schedule_plan"]


class TestFunctorStateInjection:
    """Test that state injection works correctly with functors."""

    def test_four_arg_functor_gets_state_injected(self):
        """Test that 4-arg functor gets state injected via partial."""
        functor = FourArgForwardFunctor()
        mock_state = Mock(spec=GlobalState)

        wrapped_functor = maybe_inject_state(functor, mock_state)

        # Should return a partial function
        assert isinstance(wrapped_functor, partial)
        assert wrapped_functor.func is functor
        assert wrapped_functor.args == (mock_state,)

    def test_three_arg_functor_no_state_injection(self):
        """Test that 3-arg functor doesn't get state injected."""
        functor = ThreeArgForwardFunctor()
        mock_state = Mock(spec=GlobalState)

        wrapped_functor = maybe_inject_state(functor, mock_state)

        # Should return the original functor unchanged
        assert wrapped_functor is functor

    def test_two_arg_functor_no_state_injection(self):
        """Test that 2-arg functor doesn't get state injected."""
        functor = TwoArgForwardFunctor()
        mock_state = Mock(spec=GlobalState)

        wrapped_functor = maybe_inject_state(functor, mock_state)

        # Should return the original functor unchanged
        assert wrapped_functor is functor


class TestFunctorWithPretrain:
    """Integration tests for functors with the pretrain function."""

    @patch("megatron.bridge.training.pretrain.setup")
    @patch("megatron.bridge.training.pretrain.get_dataset_provider")
    @patch("megatron.bridge.training.pretrain.runtime_config_update")
    @patch("megatron.bridge.training.pretrain.train")
    def test_pretrain_with_four_arg_functor(
        self, mock_train, mock_runtime_update, mock_get_dataset_provider, mock_setup
    ):
        """Test pretrain works with a 4-arg functor."""
        gpt_model_cfg = create_test_gpt_config()
        checkpoint_cfg = create_test_checkpoint_config(save=None)
        train_cfg = create_test_training_config(train_iters=100, skip_train=False)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            checkpoint_config=checkpoint_cfg,
            train_config=train_cfg,
        )

        functor = FourArgForwardFunctor()

        # Mock setup return
        setup_output = MagicMock()
        setup_output.state = MagicMock()
        setup_output.state.cfg = container
        setup_output.state.train_state.do_train = True
        setup_output.state.train_state.step = 0
        setup_output.state.train_state.do_valid = False
        setup_output.state.train_state.do_test = False

        # Mock fault tolerance state to avoid comparison issues
        setup_output.state.fault_tolerance_state.seen_tr_iters_cnt = 0
        setup_output.state.fault_tolerance_state.is_calculating_timeouts = False
        setup_output.state.fault_tolerance_state.is_persistent_chkpt_loaded = False
        setup_output.state.rank_monitor_client = None

        setup_output.model = MagicMock()
        setup_output.optimizer = MagicMock()
        setup_output.scheduler = MagicMock()
        setup_output.train_data_iterator = MagicMock()
        setup_output.valid_data_iterator = None
        setup_output.test_data_iterator = None
        setup_output.checkpointing_context = {}
        mock_setup.return_value = setup_output

        try:
            pretrain(container, functor)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

        # Verify the functor was passed to train
        mock_train.assert_called_once()
        assert mock_train.call_args[0][0] is functor

    @patch("megatron.bridge.training.pretrain.setup")
    @patch("megatron.bridge.training.pretrain.get_dataset_provider")
    @patch("megatron.bridge.training.pretrain.runtime_config_update")
    @patch("megatron.bridge.training.pretrain.train")
    def test_pretrain_with_stateful_functor(
        self, mock_train, mock_runtime_update, mock_get_dataset_provider, mock_setup
    ):
        """Test pretrain works with a stateful functor that tracks calls."""
        gpt_model_cfg = create_test_gpt_config()
        checkpoint_cfg = create_test_checkpoint_config(save=None)
        train_cfg = create_test_training_config(train_iters=100, skip_train=False)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            checkpoint_config=checkpoint_cfg,
            train_config=train_cfg,
        )

        functor = StatefulForwardFunctor(initial_loss=2.0)
        assert functor.call_count == 0
        assert functor.loss_history == []

        # Mock setup return
        setup_output = MagicMock()
        setup_output.state = MagicMock()
        setup_output.state.cfg = container
        setup_output.state.train_state.do_train = True
        setup_output.state.train_state.step = 0
        setup_output.state.train_state.do_valid = False
        setup_output.state.train_state.do_test = False

        # Mock fault tolerance state to avoid comparison issues
        setup_output.state.fault_tolerance_state.seen_tr_iters_cnt = 0
        setup_output.state.fault_tolerance_state.is_calculating_timeouts = False
        setup_output.state.fault_tolerance_state.is_persistent_chkpt_loaded = False
        setup_output.state.rank_monitor_client = None

        setup_output.model = MagicMock()
        setup_output.optimizer = MagicMock()
        setup_output.scheduler = MagicMock()
        setup_output.train_data_iterator = MagicMock()
        setup_output.valid_data_iterator = None
        setup_output.test_data_iterator = None
        setup_output.checkpointing_context = {}
        mock_setup.return_value = setup_output

        try:
            pretrain(container, functor)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

        # Verify the functor was passed to train and maintains its identity
        mock_train.assert_called_once()
        assert mock_train.call_args[0][0] is functor
        # Functor state should be preserved
        assert functor.initial_loss == 2.0


class TestFunctorStateDetectionEdgeCases:
    """Test edge cases in functor state detection."""

    def test_functor_with_typed_state_parameter(self):
        """Test that functors with GlobalState type hints are detected correctly."""

        class TypedStateFunctor:
            def __call__(self, state: GlobalState, data_iterator, model):
                return "typed state"

        functor = TypedStateFunctor()
        needs_injection = needs_global_state_injection(functor)
        assert needs_injection is True  # Has GlobalState type hint

    def test_functor_with_mixed_parameters(self):
        """Test functor with mixed typed and untyped parameters."""

        class MixedFunctor:
            def __call__(self, data_iterator, state: GlobalState, model):
                return "mixed"

        functor = MixedFunctor()
        needs_injection = needs_global_state_injection(functor)
        assert needs_injection is True  # Has GlobalState type hint (not first param)


class TestFunctorVsFunctionEquivalence:
    """Test that functors behave equivalently to regular functions."""

    def test_functor_vs_function_state_injection(self):
        """Test that functors and functions get the same state injection treatment."""

        def four_arg_function(state, data_iterator, model, return_schedule_plan=False):
            return torch.tensor([1.0]), partial(lambda x: x)

        functor = FourArgForwardFunctor()
        mock_state = Mock(spec=GlobalState)

        wrapped_function = maybe_inject_state(four_arg_function, mock_state)
        wrapped_functor = maybe_inject_state(functor, mock_state)

        # Both should be wrapped with partial
        assert isinstance(wrapped_function, partial)
        assert isinstance(wrapped_functor, partial)

        # Both should have the same state injected
        assert wrapped_function.args == (mock_state,)
        assert wrapped_functor.args == (mock_state,)

    def test_functor_vs_function_state_detection(self):
        """Test that functors and functions are inspected the same way for state injection."""

        def three_arg_function(data_iterator, model, return_schedule_plan=False):
            return torch.tensor([1.0]), partial(lambda x: x)

        functor = ThreeArgForwardFunctor()

        func_needs_injection = needs_global_state_injection(three_arg_function)
        functor_needs_injection = needs_global_state_injection(functor)

        assert func_needs_injection == functor_needs_injection == False  # Neither has state


class TestComplexFunctorScenarios:
    """Test complex scenarios with functors."""

    def test_functor_with_inheritance(self):
        """Test that functors work correctly with inheritance."""

        class BaseFunctor:
            def __init__(self):
                self.base_calls = 0

            def __call__(self, state, data_iterator, model, return_schedule_plan=False):
                self.base_calls += 1
                return self._forward(state, data_iterator, model, return_schedule_plan)

            def _forward(self, state, data_iterator, model, return_schedule_plan):
                return torch.tensor([1.0]), partial(lambda x: x)

        class DerivedFunctor(BaseFunctor):
            def __init__(self):
                super().__init__()
                self.derived_calls = 0

            def _forward(self, state, data_iterator, model, return_schedule_plan):
                self.derived_calls += 1
                # Override with different behavior
                return torch.tensor([0.5]), partial(lambda x: x * 0.5)

        functor = DerivedFunctor()
        needs_injection = needs_global_state_injection(functor)
        assert needs_injection is True

        # Test that inheritance works
        mock_state = Mock()
        mock_iterator = Mock()
        mock_model = Mock()

        result = functor(mock_state, mock_iterator, mock_model)
        assert functor.base_calls == 1
        assert functor.derived_calls == 1
        assert result[0].item() == 0.5

    def test_functor_with_decorator(self):
        """Test that functors work with decorators."""

        import functools

        def call_counter(cls):
            """Decorator that adds call counting to a functor while preserving signature."""
            original_call = cls.__call__

            @functools.wraps(original_call)
            def wrapped_call(self, *args, **kwargs):
                if not hasattr(self, "_decorator_calls"):
                    self._decorator_calls = 0
                self._decorator_calls += 1
                return original_call(self, *args, **kwargs)

            cls.__call__ = wrapped_call
            return cls

        @call_counter
        class DecoratedFunctor:
            def __call__(self, state, data_iterator, model, return_schedule_plan=False):
                return torch.tensor([1.0]), partial(lambda x: x)

        functor = DecoratedFunctor()
        needs_injection = needs_global_state_injection(functor)
        assert needs_injection is True

        # Test that decorator works
        mock_state = Mock()
        mock_iterator = Mock()
        mock_model = Mock()

        functor(mock_state, mock_iterator, mock_model)
        assert functor._decorator_calls == 1

        functor(mock_state, mock_iterator, mock_model)
        assert functor._decorator_calls == 2
