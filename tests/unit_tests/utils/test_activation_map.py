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

"""Tests for megatron.bridge.utils.activation_map."""

import pytest
import torch.nn.functional as F
from megatron.core.activations import fast_gelu, squared_relu

import torch

from megatron.bridge.utils.activation_map import (
    ACTIVATION_FUNC_MAP,
    DTYPE_MAP,
    callable_to_str,
    str_to_callable,
    str_to_dtype,
)


class TestStrToCallable:
    """Tests for str_to_callable(name: str) -> Callable."""

    def test_silu_short_name(self):
        """'silu' resolves to F.silu."""
        assert str_to_callable("silu") is F.silu

    def test_gelu_short_name(self):
        """'gelu' resolves to F.gelu."""
        assert str_to_callable("gelu") is F.gelu

    def test_relu_short_name(self):
        """'relu' resolves to F.relu."""
        assert str_to_callable("relu") is F.relu

    def test_sigmoid_short_name(self):
        """'sigmoid' resolves to F.sigmoid."""
        assert str_to_callable("sigmoid") is F.sigmoid

    def test_squared_relu_canonical(self):
        """'squared_relu' resolves to the squared_relu function."""
        assert str_to_callable("squared_relu") is squared_relu

    def test_relu2_alias(self):
        """'relu2' is an alias for squared_relu."""
        assert str_to_callable("relu2") is squared_relu

    def test_fast_gelu(self):
        """'fast_gelu' resolves to fast_gelu."""
        assert str_to_callable("fast_gelu") is fast_gelu

    def test_gelu_pytorch_tanh_alias(self):
        """'gelu_pytorch_tanh' is an alias for fast_gelu."""
        assert str_to_callable("gelu_pytorch_tanh") is fast_gelu

    def test_fully_qualified_silu(self):
        """Fully-qualified 'torch.nn.functional.silu' resolves to F.silu."""
        assert str_to_callable("torch.nn.functional.silu") is F.silu

    def test_fully_qualified_gelu(self):
        """Fully-qualified 'torch.nn.functional.gelu' resolves to F.gelu."""
        assert str_to_callable("torch.nn.functional.gelu") is F.gelu

    def test_fully_qualified_relu(self):
        """Fully-qualified 'torch.nn.functional.relu' resolves to F.relu."""
        assert str_to_callable("torch.nn.functional.relu") is F.relu

    def test_fully_qualified_sigmoid(self):
        """Fully-qualified 'torch.nn.functional.sigmoid' resolves to F.sigmoid."""
        assert str_to_callable("torch.nn.functional.sigmoid") is F.sigmoid

    def test_unknown_raises_value_error(self):
        """Unknown name raises ValueError."""
        with pytest.raises(ValueError):
            str_to_callable("unknown_func")

    def test_empty_string_raises_value_error(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            str_to_callable("")

    def test_value_error_message_contains_name(self):
        """ValueError message contains the unknown name."""
        with pytest.raises(ValueError, match="unknown_func"):
            str_to_callable("unknown_func")

    def test_value_error_message_lists_known_names(self):
        """ValueError message lists some known activation names."""
        with pytest.raises(ValueError, match="silu"):
            str_to_callable("unknown_func")

    def test_importlib_fallback_torch_tanh(self):
        """Dotted path not in map falls back to importlib import."""
        import torch

        result = str_to_callable("torch.tanh")
        assert result is torch.tanh


class TestCallableToStr:
    """Tests for callable_to_str(fn: Callable) -> str | None."""

    def test_silu_to_str(self):
        """F.silu returns 'silu'."""
        assert callable_to_str(F.silu) == "silu"

    def test_gelu_to_str(self):
        """F.gelu returns 'gelu'."""
        assert callable_to_str(F.gelu) == "gelu"

    def test_relu_to_str(self):
        """F.relu returns 'relu'."""
        assert callable_to_str(F.relu) == "relu"

    def test_sigmoid_to_str(self):
        """F.sigmoid returns 'sigmoid'."""
        assert callable_to_str(F.sigmoid) == "sigmoid"

    def test_squared_relu_canonical_name(self):
        """squared_relu returns its canonical short name, not 'relu2'."""
        name = callable_to_str(squared_relu)
        assert name == "squared_relu"

    def test_fast_gelu_to_str(self):
        """fast_gelu returns 'fast_gelu'."""
        assert callable_to_str(fast_gelu) == "fast_gelu"

    def test_unknown_callable_returns_none(self):
        """Unregistered callable returns None."""
        assert callable_to_str(lambda x: x) is None

    def test_unknown_builtin_returns_none(self):
        """Builtin not in registry returns None."""
        assert callable_to_str(len) is None


class TestActivationFuncMapContents:
    """Tests for ACTIVATION_FUNC_MAP structure guarantees."""

    def test_short_names_present(self):
        """All expected short names are in the map."""
        expected = {"silu", "gelu", "relu", "sigmoid", "relu2", "squared_relu", "fast_gelu", "gelu_pytorch_tanh"}
        for name in expected:
            assert name in ACTIVATION_FUNC_MAP, f"'{name}' missing from ACTIVATION_FUNC_MAP"

    def test_fully_qualified_aliases_present(self):
        """Fully-qualified torch.nn.functional aliases are present."""
        for name in ("torch.nn.functional.silu", "torch.nn.functional.gelu", "torch.nn.functional.relu"):
            assert name in ACTIVATION_FUNC_MAP, f"'{name}' missing from ACTIVATION_FUNC_MAP"

    def test_all_values_are_callable(self):
        """Every value in the map is callable."""
        for name, fn in ACTIVATION_FUNC_MAP.items():
            assert callable(fn), f"ACTIVATION_FUNC_MAP['{name}'] is not callable"


class TestStrToDtype:
    """Tests for str_to_dtype(name: str) -> torch.dtype."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("fp32", torch.float32),
            ("float32", torch.float32),
            ("32", torch.float32),
            ("32-true", torch.float32),
            ("torch.float32", torch.float32),
            ("fp16", torch.float16),
            ("float16", torch.float16),
            ("16", torch.float16),
            ("16-mixed", torch.float16),
            ("torch.float16", torch.float16),
            ("bf16", torch.bfloat16),
            ("bfloat16", torch.bfloat16),
            ("bf16-mixed", torch.bfloat16),
            ("torch.bfloat16", torch.bfloat16),
        ],
    )
    def test_known_names(self, name, expected):
        assert str_to_dtype(name) == expected

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            str_to_dtype("unknown")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            str_to_dtype("")

    def test_error_message_lists_known_names(self):
        with pytest.raises(ValueError, match="bf16"):
            str_to_dtype("bad_dtype")

    def test_dtype_map_all_values_are_torch_dtype(self):
        for name, dtype in DTYPE_MAP.items():
            assert isinstance(dtype, torch.dtype), f"DTYPE_MAP['{name}'] is not a torch.dtype"


class TestRoundTrip:
    """Round-trip tests: str → callable → str and callable → str → callable."""

    def test_silu_round_trip(self):
        """str_to_callable then callable_to_str returns canonical name."""
        assert callable_to_str(str_to_callable("silu")) == "silu"

    def test_gelu_round_trip(self):
        """gelu round-trip preserves name."""
        assert callable_to_str(str_to_callable("gelu")) == "gelu"

    def test_squared_relu_round_trip(self):
        """squared_relu round-trip preserves canonical name."""
        assert callable_to_str(str_to_callable("squared_relu")) == "squared_relu"

    def test_fqn_to_short_canonical_name(self):
        """Fully-qualified name resolves to callable, then maps to short canonical name."""
        assert callable_to_str(str_to_callable("torch.nn.functional.silu")) == "silu"

    def test_relu2_alias_canonical_name(self):
        """'relu2' alias resolves to callable whose canonical name is 'squared_relu'."""
        assert callable_to_str(str_to_callable("relu2")) == "squared_relu"
