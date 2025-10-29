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

"""Tests for state injection logic with type hint detection."""

from functools import partial
from typing import Iterable
from unittest.mock import Mock

import torch
from megatron.core.models.gpt import GPTModel

from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.train_utils import maybe_inject_state


class TestTypeHintBasedStateInjection:
    """Test state injection based on type hints."""

    def test_inject_with_globalstate_type_hint_first_param(self):
        """Test state injection when first parameter has GlobalState type hint."""

        def forward_step(state: GlobalState, data_iterator, model, return_schedule_plan=False):
            return f"state: {state.name}"

        mock_state = Mock()
        mock_state.name = "test_state"

        wrapped = maybe_inject_state(forward_step, mock_state)

        assert isinstance(wrapped, partial)
        assert wrapped.args == (mock_state,)

        # Test calling the wrapped function
        result = wrapped(Mock(), Mock(), True)
        assert result == "state: test_state"

    def test_inject_with_globalstate_type_hint_middle_param(self):
        """Test state injection when GlobalState type hint is in middle parameter."""

        def forward_step(data_iterator, state: GlobalState, model):
            return f"state: {state.name}"

        mock_state = Mock()
        mock_state.name = "test_state"

        wrapped = maybe_inject_state(forward_step, mock_state)

        # Should inject state because GlobalState type hint was found
        assert isinstance(wrapped, partial)
        assert wrapped.args == (mock_state,)

    def test_inject_with_string_type_annotation(self):
        """Test state injection with string type annotation (forward reference)."""

        def forward_step(state: "GlobalState", data_iterator, model):
            return f"state: {state.name}"

        mock_state = Mock()
        mock_state.name = "test_state"

        wrapped = maybe_inject_state(forward_step, mock_state)

        assert isinstance(wrapped, partial)
        assert wrapped.args == (mock_state,)

    def test_no_injection_without_globalstate_type_hint(self):
        """Test no state injection when no GlobalState type hint is present."""

        def forward_step(data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False):
            return "no state needed"

        mock_state = Mock()

        wrapped = maybe_inject_state(forward_step, mock_state)

        # Should return original function unchanged
        assert wrapped is forward_step
        assert not isinstance(wrapped, partial)

    def test_fallback_to_name_based_detection(self):
        """Test fallback to name-based detection when no type hints are present."""

        def forward_step(state, data_iterator, model, return_schedule_plan=False):
            return f"state: {state.name}"

        mock_state = Mock()
        mock_state.name = "test_state"

        wrapped = maybe_inject_state(forward_step, mock_state)

        # Should inject based on parameter name 'state'
        assert isinstance(wrapped, partial)
        assert wrapped.args == (mock_state,)

    def test_no_injection_when_first_param_not_state(self):
        """Test no injection when first parameter is not named 'state' and has no GlobalState type."""

        def forward_step(data_iterator, model, return_schedule_plan=False):
            return "no state"

        mock_state = Mock()

        wrapped = maybe_inject_state(forward_step, mock_state)

        assert wrapped is forward_step
        assert not isinstance(wrapped, partial)


class TestFunctorTypeHintStateInjection:
    """Test state injection with functors using type hints."""

    def test_functor_with_globalstate_type_hint(self):
        """Test functor with GlobalState type hint gets state injected."""

        class TypedForwardFunctor:
            def __init__(self):
                self.seen_state = None

            def __call__(self, state: GlobalState, data_iterator: Iterable, model: GPTModel):
                self.seen_state = state
                return torch.tensor([1.0]), partial(lambda x: x)

        functor = TypedForwardFunctor()
        mock_state = Mock()
        mock_state.name = "test_state"

        wrapped = maybe_inject_state(functor, mock_state)

        assert isinstance(wrapped, partial)
        assert wrapped.args == (mock_state,)

        # Test calling the wrapped functor
        wrapped(Mock(), Mock())
        assert functor.seen_state is mock_state

    def test_functor_without_type_hints_name_fallback(self):
        """Test functor without type hints falls back to name-based detection."""

        class NameBasedFunctor:
            def __init__(self):
                self.seen_state = None

            def __call__(self, state, data_iterator, model):
                self.seen_state = state
                return torch.tensor([1.0]), partial(lambda x: x)

        functor = NameBasedFunctor()
        mock_state = Mock()

        wrapped = maybe_inject_state(functor, mock_state)

        assert isinstance(wrapped, partial)
        assert wrapped.args == (mock_state,)

    def test_functor_no_injection_without_state(self):
        """Test functor without state parameter gets no injection."""

        class NoStateFunctor:
            def __call__(self, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False):
                return torch.tensor([1.0]), partial(lambda x: x)

        functor = NoStateFunctor()
        mock_state = Mock()

        wrapped = maybe_inject_state(functor, mock_state)

        assert wrapped is functor
        assert not isinstance(wrapped, partial)


class TestAmbiguousSignatureResolution:
    """Test resolution of ambiguous signatures using type hints."""

    def test_three_args_with_state_type_hint_injects(self):
        """Test that (state: GlobalState, data_iterator, model) correctly injects state."""

        def forward_step(state: GlobalState, data_iterator, model):
            return f"received state: {state.name}"

        mock_state = Mock()
        mock_state.name = "injected"

        wrapped = maybe_inject_state(forward_step, mock_state)

        # Should inject state because of type hint
        assert isinstance(wrapped, partial)

        result = wrapped(Mock(), Mock())
        assert result == "received state: injected"

    def test_three_args_without_state_type_hint_no_injection(self):
        """Test that (data_iterator, model, return_schedule_plan) doesn't inject state."""

        def forward_step(data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False):
            return f"no state, schedule_plan: {return_schedule_plan}"

        mock_state = Mock()

        wrapped = maybe_inject_state(forward_step, mock_state)

        # Should NOT inject state because no GlobalState type hint
        assert wrapped is forward_step
        assert not isinstance(wrapped, partial)

        result = wrapped(Mock(), Mock(), True)
        assert result == "no state, schedule_plan: True"

    def test_ambiguous_three_args_resolved_by_type_hint(self):
        """Test that type hints resolve the ambiguity between different 3-arg patterns."""

        # Pattern 1: State injection expected
        def state_forward_step(state: GlobalState, data_iterator, model):
            return "with state"

        # Pattern 2: No state injection expected
        def schedule_forward_step(data_iterator, model, return_schedule_plan=False):
            return "with schedule"

        mock_state = Mock()

        wrapped_state = maybe_inject_state(state_forward_step, mock_state)
        wrapped_schedule = maybe_inject_state(schedule_forward_step, mock_state)

        # State function should be wrapped
        assert isinstance(wrapped_state, partial)

        # Schedule function should not be wrapped
        assert wrapped_schedule is schedule_forward_step
        assert not isinstance(wrapped_schedule, partial)


class TestEdgeCases:
    """Test edge cases in type hint detection."""

    def test_mixed_type_hints_first_param_wins(self):
        """Test that when multiple params have types, first GlobalState param wins."""

        def forward_step(data_iterator: Iterable, state: GlobalState, model: GPTModel):
            return f"state: {state.name}"

        mock_state = Mock()
        mock_state.name = "test"

        wrapped = maybe_inject_state(forward_step, mock_state)

        # Should inject because GlobalState type hint was found (even though not first param)
        assert isinstance(wrapped, partial)

    def test_no_type_hints_fallback_to_name(self):
        """Test fallback to name-based detection when no type hints are present."""

        def forward_step(state, data_iterator, model):
            return f"state: {state.name}"

        mock_state = Mock()
        mock_state.name = "fallback"

        wrapped = maybe_inject_state(forward_step, mock_state)

        # Should inject based on parameter name
        assert isinstance(wrapped, partial)

        result = wrapped(Mock(), Mock())
        assert result == "state: fallback"

    def test_wrong_parameter_name_no_injection(self):
        """Test that wrong parameter name with no type hints doesn't inject."""

        def forward_step(wrong_name, data_iterator, model):  # Wrong name
            return "should not inject"

        mock_state = Mock()

        wrapped = maybe_inject_state(forward_step, mock_state)

        # Should NOT inject because first param is not named 'state'
        assert wrapped is forward_step
        assert not isinstance(wrapped, partial)
