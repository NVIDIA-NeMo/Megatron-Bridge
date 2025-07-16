# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Tests for megatron.bridge.models.state module."""

import pytest
import torch

from megatron.bridge.models.state import StateDict


class TestStateDict:
    """Test cases for StateDict class."""

    def test_init_from_dict(self):
        """Test StateDict initialization from a dictionary."""
        d = {
            "model.layer.0.weight": torch.randn(10, 10),
            "model.layer.0.bias": torch.randn(10),
            "model.layer.1.weight": torch.randn(10, 10),
            "model.layer.1.bias": torch.randn(10),
        }
        state = StateDict(d)

        # Test length
        assert len(state) == 4

        # Test key access
        assert "model.layer.0.weight" in state
        assert state["model.layer.0.weight"].shape == (10, 10)

    def test_getitem_single_key(self):
        """Test accessing a single key."""
        d = {
            "model.weight": torch.tensor([1.0, 2.0, 3.0]),
            "model.bias": torch.tensor([0.5]),
        }
        state = StateDict(d)

        weight = state["model.weight"]
        assert torch.allclose(weight, torch.tensor([1.0, 2.0, 3.0]))

        with pytest.raises(KeyError):
            _ = state["nonexistent.key"]

    def test_getitem_multiple_keys(self):
        """Test accessing multiple keys with a list."""
        d = {
            "model.layer.0.weight": torch.randn(5, 5),
            "model.layer.0.bias": torch.randn(5),
            "model.layer.1.weight": torch.randn(5, 5),
        }
        state = StateDict(d)

        # Access with list of keys
        result = state[["model.layer.0.weight", "model.layer.0.bias"]]
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "model.layer.0.weight" in result
        assert "model.layer.0.bias" in result

    def test_getitem_glob_pattern(self):
        """Test accessing keys with glob patterns."""
        d = {
            "model.layer.0.weight": torch.randn(3, 3),
            "model.layer.0.bias": torch.randn(3),
            "model.layer.1.weight": torch.randn(3, 3),
            "model.layer.1.bias": torch.randn(3),
            "model.output.weight": torch.randn(3, 3),
        }
        state = StateDict(d)

        # Test glob pattern
        weights = state["*.weight"]
        assert len(weights) == 3
        assert all(k.endswith(".weight") for k in weights.keys())

        # Test more specific glob
        layer_params = state["model.layer.*"]
        assert len(layer_params) == 4
        assert all(k.startswith("model.layer.") for k in layer_params.keys())

    def test_keys_method(self):
        """Test keys() method."""
        d = {
            "a": torch.tensor(1.0),
            "b": torch.tensor(2.0),
            "c": torch.tensor(3.0),
        }
        state = StateDict(d)

        keys = list(state.keys())
        assert sorted(keys) == ["a", "b", "c"]

    def test_items_method(self):
        """Test items() method."""
        d = {
            "param1": torch.tensor([1.0]),
            "param2": torch.tensor([2.0]),
        }
        state = StateDict(d)

        items = list(state.items())
        assert len(items) == 2

        for key, value in items:
            assert key in d
            assert torch.equal(value, d[key])

    def test_iter_method(self):
        """Test iteration over StateDict."""
        d = {
            "x": torch.tensor(1),
            "y": torch.tensor(2),
            "z": torch.tensor(3),
        }
        state = StateDict(d)

        collected_keys = []
        for key in state:
            collected_keys.append(key)

        assert sorted(collected_keys) == sorted(d.keys())

    def test_contains_method(self):
        """Test __contains__ method."""
        d = {"model.weight": torch.randn(10)}
        state = StateDict(d)

        assert "model.weight" in state
        assert "model.bias" not in state
        assert "nonexistent" not in state

    def test_empty_statedict(self):
        """Test empty StateDict."""
        state = StateDict({})

        assert len(state) == 0
        assert list(state.keys()) == []
        assert list(state.items()) == []

        with pytest.raises(KeyError):
            _ = state["any.key"]
