# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import types
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version as PkgVersion

from megatron.bridge.core.utils.import_utils import (
    UnavailableError,
    UnavailableMeta,
    get_torch_version,
    gpu_only_import,
    gpu_only_import_from,
    is_torch_min_version,
    is_unavailable,
    safe_import,
    safe_import_from,
)


class TestUnavailableMeta:
    """Test suite for the UnavailableMeta metaclass."""

    def test_metaclass_creation(self):
        """Test that UnavailableMeta creates a class with the expected properties."""
        # Create a class using UnavailableMeta
        TestClass = UnavailableMeta("TestClass", (), {})

        # The class name should be prefixed with "MISSING"
        assert TestClass.__name__ == "MISSINGTestClass"

        # The default error message should be set
        assert TestClass._msg == "TestClass could not be imported"

    def test_custom_error_message(self):
        """Test that a custom error message can be provided."""
        custom_msg = "Custom error message"
        TestClass = UnavailableMeta("TestClass", (), {"_msg": custom_msg})

        assert TestClass._msg == custom_msg

        # Verify the message is used in exceptions
        with pytest.raises(UnavailableError, match=custom_msg):
            TestClass()

    def test_call_raises_error(self):
        """Test that attempting to instantiate the class raises UnavailableError."""
        TestClass = UnavailableMeta("TestClass", (), {})

        with pytest.raises(UnavailableError):
            TestClass()

        with pytest.raises(UnavailableError):
            TestClass(1, 2, 3, key="value")

    def test_attribute_access_raises_error(self):
        """Test that accessing attributes raises UnavailableError."""
        TestClass = UnavailableMeta("TestClass", (), {})

        with pytest.raises(UnavailableError):
            TestClass.some_attribute

    def test_arithmetic_operations_raise_error(self):
        """Test that arithmetic operations raise UnavailableError."""
        TestClass = UnavailableMeta("TestClass", (), {})

        operations = [
            lambda c: c + 1,
            lambda c: 1 + c,  # __radd__
            lambda c: c - 1,
            lambda c: 1 - c,  # __rsub__
            lambda c: c * 2,
            lambda c: 2 * c,  # __rmul__
            lambda c: c / 2,
            lambda c: 2 / c,  # __rtruediv__
            lambda c: c // 2,
            lambda c: 2 // c,  # __rfloordiv__
            lambda c: c**2,
            lambda c: 2**c,  # __rpow__
            lambda c: -c,  # __neg__
            lambda c: abs(c),  # __abs__
        ]

        for op in operations:
            with pytest.raises(UnavailableError):
                op(TestClass)

    def test_comparison_operations_raise_error(self):
        """Test that comparison operations raise UnavailableError."""
        TestClass = UnavailableMeta("TestClass", (), {})
        another_class = UnavailableMeta("AnotherClass", (), {})

        comparisons = [
            lambda c: c == another_class,
            lambda c: c != another_class,
            lambda c: c < another_class,
            lambda c: c <= another_class,
            lambda c: c > another_class,
            lambda c: c >= another_class,
        ]

        for comp in comparisons:
            with pytest.raises(UnavailableError):
                comp(TestClass)

    def test_container_operations_raise_error(self):
        """Test that container operations raise UnavailableError."""
        TestClass = UnavailableMeta("TestClass", (), {})

        with pytest.raises(UnavailableError):
            len(TestClass)

        with pytest.raises(UnavailableError):
            TestClass[0]

        with pytest.raises(UnavailableError):
            TestClass[0] = 1

        with pytest.raises(UnavailableError):
            del TestClass[0]

        with pytest.raises(UnavailableError):
            iter(TestClass)

    def test_descriptor_operations_raise_error(self):
        """Test that descriptor operations raise UnavailableError."""
        TestClass = UnavailableMeta("TestClass", (), {})

        with pytest.raises(UnavailableError):
            TestClass.__get__(None, None)

        with pytest.raises(UnavailableError):
            TestClass.__delete__(None)


class TestSafeImport:
    """Test suite for the safe_import function."""

    def test_successful_import(self):
        """Test safe_import with a module that exists."""
        module, success = safe_import("os")
        assert success is True
        assert isinstance(module, types.ModuleType)
        assert module.__name__ == "os"

    def test_failed_import(self):
        """Test safe_import with a module that doesn't exist."""
        module, success = safe_import("nonexistent_module")
        assert success is False
        assert is_unavailable(module)
        assert type(module) is UnavailableMeta

    def test_import_with_custom_message(self):
        """Test safe_import with a custom error message."""
        custom_msg = "Custom error message"
        module, success = safe_import("nonexistent_module", msg=custom_msg)

        assert success is False
        assert is_unavailable(module)

        # Verify the custom message is used when trying to use the module
        with pytest.raises(UnavailableError, match=custom_msg):
            module()

    def test_import_with_alternative(self):
        """Test safe_import with an alternative module."""
        alt_module = object()
        module, success = safe_import("nonexistent_module", alt=alt_module)

        assert success is False
        assert module is alt_module

    def test_unavailable_module_raises_error_when_used(self):
        """Test that using a UnavailableMeta placeholder raises UnavailableError."""
        module, success = safe_import("nonexistent_module")

        assert success is False

        # Test various operations that should raise UnavailableError
        with pytest.raises(UnavailableError):
            module()

        with pytest.raises(UnavailableError):
            module.attribute

        with pytest.raises(UnavailableError):
            module + 1

        with pytest.raises(UnavailableError):
            module == 1


class TestSafeImportFrom:
    """Test suite for the safe_import_from function."""

    def test_successful_import_from(self):
        """Test safe_import_from with a symbol that exists."""
        symbol, success = safe_import_from("os", "path")
        assert success is True

        import os

        assert symbol is os.path

    def test_failed_import_from_nonexistent_module(self):
        """Test safe_import_from with a module that doesn't exist."""
        symbol, success = safe_import_from("nonexistent_module", "nonexistent_symbol")
        assert success is False
        assert is_unavailable(symbol)

    def test_failed_import_from_nonexistent_symbol(self):
        """Test safe_import_from with a symbol that doesn't exist in an existing module."""
        symbol, success = safe_import_from("os", "nonexistent_symbol")
        assert success is False
        assert is_unavailable(symbol)

    def test_import_from_with_custom_message(self):
        """Test safe_import_from with a custom error message."""
        custom_msg = "Custom error message for symbol"
        symbol, success = safe_import_from("os", "nonexistent_symbol", msg=custom_msg)

        assert success is False

        # Verify the custom message is used when trying to use the symbol
        with pytest.raises(UnavailableError, match=custom_msg):
            symbol()

    def test_import_from_with_alternative(self):
        """Test safe_import_from with an alternative symbol."""
        alt_symbol = object()
        symbol, success = safe_import_from("os", "nonexistent_symbol", alt=alt_symbol)

        assert success is False
        assert symbol is alt_symbol

    def test_fallback_module(self):
        """Test safe_import_from with a fallback module."""
        # First import fails, but fallback succeeds
        with patch("importlib.import_module") as mock_import:
            # Mock the first import to fail as AttributeError
            def side_effect(name):
                if name == "primary_module":
                    raise AttributeError("Symbol not found")
                elif name == "fallback_module":
                    mock_module = MagicMock()
                    mock_module.symbol = "fallback_symbol"
                    return mock_module
                else:
                    raise ImportError(f"Unexpected module: {name}")

            mock_import.side_effect = side_effect

            symbol, success = safe_import_from("primary_module", "symbol", fallback_module="fallback_module")

            assert success is True
            assert symbol == "fallback_symbol"

    def test_fallback_module_both_fail(self):
        """Test safe_import_from when both primary and fallback modules fail."""
        symbol, success = safe_import_from("nonexistent_primary", "symbol", fallback_module="nonexistent_fallback")

        assert success is False
        assert is_unavailable(symbol)


class TestGPUOnlyImport:
    """Test suite for GPU-only import functions."""

    def test_gpu_only_import_success(self):
        """Test gpu_only_import with a module that exists."""
        module, success = gpu_only_import("os")
        assert success is True
        assert isinstance(module, types.ModuleType)
        assert module.__name__ == "os"

    def test_gpu_only_import_failure(self):
        """Test gpu_only_import with a module that doesn't exist."""
        module, success = gpu_only_import("nonexistent_gpu_module")
        assert success is False
        assert is_unavailable(module)

        # Verify the GPU-specific error message is used
        with pytest.raises(UnavailableError) as exc_info:
            module()
        assert "GPU-enabled installations" in str(exc_info.value)
        assert "nemo-curator[cuda12x]" in str(exc_info.value)

    def test_gpu_only_import_with_alternative(self):
        """Test gpu_only_import with an alternative module."""
        alt_module = object()
        module, success = gpu_only_import("nonexistent_gpu_module", alt=alt_module)

        assert success is False
        assert module is alt_module

    def test_gpu_only_import_from_success(self):
        """Test gpu_only_import_from with a symbol that exists."""
        symbol, success = gpu_only_import_from("os", "path")
        assert success is True

        import os

        assert symbol is os.path

    def test_gpu_only_import_from_failure(self):
        """Test gpu_only_import_from with a symbol that doesn't exist."""
        symbol, success = gpu_only_import_from("os", "nonexistent_symbol")
        assert success is False
        assert is_unavailable(symbol)

        # Verify the GPU-specific error message is used
        with pytest.raises(UnavailableError) as exc_info:
            symbol()
        assert "GPU-enabled installations" in str(exc_info.value)
        assert "nemo-curator[cuda12x]" in str(exc_info.value)

    def test_gpu_only_import_from_with_alternative(self):
        """Test gpu_only_import_from with an alternative symbol."""
        alt_symbol = object()
        symbol, success = gpu_only_import_from("os", "nonexistent_symbol", alt=alt_symbol)

        assert success is False
        assert symbol is alt_symbol


class TestTorchVersionUtils:
    """Test suite for PyTorch version utility functions."""

    def test_get_torch_version_returns_version(self):
        """Test that get_torch_version returns a packaging Version object."""
        version = get_torch_version()
        assert isinstance(version, PkgVersion)

    def test_get_torch_version_fallback(self):
        """Test that get_torch_version falls back to '0.0.0' when torch.__version__ is unavailable."""
        with patch("torch.__version__", side_effect=AttributeError("No version attribute")):
            version = get_torch_version()
            assert version == PkgVersion("0.0.0")

    def test_is_torch_min_version_true(self):
        """Test is_torch_min_version returns True when torch version meets minimum."""
        with patch("megatron.bridge.core.utils.import_utils.get_torch_version", return_value=PkgVersion("2.0.0")):
            assert is_torch_min_version("1.9.0") is True
            assert is_torch_min_version("2.0.0") is True

    def test_is_torch_min_version_false(self):
        """Test is_torch_min_version returns False when torch version is below minimum."""
        with patch("megatron.bridge.core.utils.import_utils.get_torch_version", return_value=PkgVersion("1.8.0")):
            assert is_torch_min_version("1.9.0") is False
            assert is_torch_min_version("2.0.0") is False

    def test_is_torch_min_version_without_equality(self):
        """Test is_torch_min_version with check_equality=False."""
        with patch("megatron.bridge.core.utils.import_utils.get_torch_version", return_value=PkgVersion("2.0.0")):
            assert is_torch_min_version("1.9.0", check_equality=False) is True
            assert is_torch_min_version("2.0.0", check_equality=False) is False
            assert is_torch_min_version("2.1.0", check_equality=False) is False


class TestIsUnavailable:
    """Test suite for the is_unavailable utility function."""

    def test_is_unavailable_with_unavailable_object(self):
        """Test is_unavailable returns True for UnavailableMeta objects."""
        unavailable_obj = UnavailableMeta("TestClass", (), {})
        assert is_unavailable(unavailable_obj) is True

    def test_is_unavailable_with_regular_object(self):
        """Test is_unavailable returns False for regular objects."""
        regular_obj = object()
        assert is_unavailable(regular_obj) is False

        # Test with various types
        assert is_unavailable("string") is False
        assert is_unavailable(123) is False
        assert is_unavailable([1, 2, 3]) is False
        assert is_unavailable({"key": "value"}) is False

    def test_is_unavailable_with_none(self):
        """Test is_unavailable returns False for None."""
        assert is_unavailable(None) is False
