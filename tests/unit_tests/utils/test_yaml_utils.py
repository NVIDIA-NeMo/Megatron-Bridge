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

import enum
import inspect
from dataclasses import dataclass
from functools import partial
from unittest.mock import mock_open, patch

import pytest
import yaml

from nemo_lm.utils.yaml_utils import (
    _enum_representer,
    _function_representer,
    _generation_config_representer,
    _safe_object_representer,
    _torch_dtype_representer,
    dump_dataclass_to_yaml,
    safe_yaml_representers,
)


# Test fixtures
class DummyClass:
    """Dummy class for testing."""

    def test_method(self):
        """Dummy method for testing."""
        pass

    @staticmethod
    def static_method():
        """Dummy static method for testing."""
        pass

    @classmethod
    def class_method(cls):
        """Dummy class method for testing."""
        pass


@dataclass
class DummyDataClass:
    """Dummy dataclass for testing."""

    name: str
    value: int


def dummy_function():
    """Dummy function for testing."""
    pass


def dummy_function_with_args(a, b=2, c=3):
    """Dummy function with args for testing."""
    return a + b + c


def test_context_manager_preserves_representers():
    """Test that the context manager preserves and restores original representers."""
    # Store original representers count
    original_rep_count = len(yaml.SafeDumper.yaml_representers)
    original_multi_rep_count = len(yaml.SafeDumper.yaml_multi_representers)

    # Use context manager
    with safe_yaml_representers():
        # Should have added new representers
        assert len(yaml.SafeDumper.yaml_representers) > original_rep_count
        assert len(yaml.SafeDumper.yaml_multi_representers) > original_multi_rep_count

    # After context, should be back to original counts
    assert len(yaml.SafeDumper.yaml_representers) == original_rep_count
    assert len(yaml.SafeDumper.yaml_multi_representers) == original_multi_rep_count


def test_function_yaml_dump():
    """Test the function representer using yaml.safe_dump."""
    with safe_yaml_representers():
        result = yaml.safe_dump(dummy_function)
        assert "_target_" in result
        assert "tests.unit_tests.utils.test_yaml_utils.dummy_function" in result
        assert "_call_: false" in result


def test_class_representer():
    """Test the class representer."""
    with safe_yaml_representers():
        result = yaml.safe_dump(DummyClass)
        assert "_target_" in result
        assert "tests.unit_tests.utils.test_yaml_utils.DummyClass" in result
        assert "_call_: false" in result


def test_instance_representer():
    """Test the instance representer."""
    instance = DummyClass()
    with safe_yaml_representers():
        result = yaml.safe_dump(instance)
        assert "_target_" in result
        assert "tests.unit_tests.utils.test_yaml_utils.DummyClass" in result
        assert "_call_: true" in result


def test_dataclass_representer():
    """Test representation of dataclasses."""
    instance = DummyDataClass(name="test", value=42)
    with safe_yaml_representers():
        result = yaml.safe_dump(instance)
        assert "_target_" in result
        assert "tests.unit_tests.utils.test_yaml_utils.DummyDataClass" in result
        assert "_call_: true" in result


def test_nested_objects():
    """Test representation of nested objects."""
    data = {
        "function": dummy_function,
        "class": DummyClass,
        "instance": DummyClass(),
        "dataclass": DummyDataClass(name="nested", value=100),
    }

    with safe_yaml_representers():
        result = yaml.safe_dump(data)
        assert "function" in result
        assert "class" in result
        assert "instance" in result
        assert "dataclass" in result
        assert "tests.unit_tests.utils.test_yaml_utils.dummy_function" in result
        assert "tests.unit_tests.utils.test_yaml_utils.DummyClass" in result
        assert "tests.unit_tests.utils.test_yaml_utils.DummyDataClass" in result


def test_safe_yaml_representers_context():
    """Test that the context manager properly adds and removes representers."""
    # Check that custom representers are not registered initially
    assert type(lambda: ...) not in yaml.SafeDumper.yaml_representers

    # Enter context manager
    with safe_yaml_representers():
        # Check that custom representers are registered
        assert type(lambda: ...) in yaml.SafeDumper.yaml_representers

        # Test dumping a function
        result = yaml.safe_dump(dummy_function)
        assert "_target_:" in result
        assert "tests.unit_tests.utils.test_yaml_utils.dummy_function" in result
        assert "_call_: false" in result

    # Check that custom representers are removed after context
    assert type(lambda: ...) not in yaml.SafeDumper.yaml_representers


def test_function_representer():
    """Test the function representer."""
    # Create a dummy dumper
    dumper = yaml.SafeDumper(None)

    # Test with a regular function
    result = _function_representer(dumper, dummy_function)

    # Verify the result - check that it's a MappingNode
    assert hasattr(result, "value")
    assert len(result.value) > 0

    # Extract the values from the node
    mapping_data = {}
    for key_node, value_node in result.value:
        mapping_data[key_node.value] = value_node.value

    # Verify the specific values
    expected_target = f"{inspect.getmodule(dummy_function).__name__}.{dummy_function.__qualname__}"
    assert mapping_data["_target_"] == expected_target
    # YAML ScalarNode for booleans might be 'false' as string, not False as Python bool
    assert mapping_data["_call_"] == "false" or mapping_data["_call_"] is False

    # Test with a lambda function
    lambda_func = lambda x: x + 1  # noqa: E731
    result = _function_representer(dumper, lambda_func)

    # Extract the values from the node
    mapping_data = {}
    for key_node, value_node in result.value:
        mapping_data[key_node.value] = value_node.value

    # Verify the lambda function
    # The module name for lambda can be tricky, so we check endswith qualname
    assert mapping_data["_target_"].endswith(f".{lambda_func.__qualname__}")
    assert mapping_data["_call_"] == "false" or mapping_data["_call_"] is False

    # Test with a method
    method = DummyClass.test_method
    result = _function_representer(dumper, method)

    # Extract the values from the node
    mapping_data = {}
    for key_node, value_node in result.value:
        mapping_data[key_node.value] = value_node.value

    # Verify the method
    expected_target = f"{inspect.getmodule(method).__name__}.{method.__qualname__}"
    assert mapping_data["_target_"] == expected_target
    assert mapping_data["_call_"] == "false" or mapping_data["_call_"] is False


def test_torch_dtype_representer():
    """Test the torch dtype representer."""
    try:
        import torch

        # Create a dummy dumper
        dumper = yaml.SafeDumper(None)

        # Test with torch.float32
        dtype = torch.float32
        result = _torch_dtype_representer(dumper, dtype)

        # Verify the result - check that it's a MappingNode
        assert hasattr(result, "value")
        assert len(result.value) > 0

        # Extract the values from the node
        mapping_data = {}
        for key_node, value_node in result.value:
            mapping_data[key_node.value] = value_node.value

        # Verify the result
        assert mapping_data["_target_"] == str(dtype)
        # YAML ScalarNode for booleans might be 'false' as string, not False as Python bool
        assert mapping_data["_call_"] == "false" or mapping_data["_call_"] is False

        # Test with torch.int64
        dtype = torch.int64
        result = _torch_dtype_representer(dumper, dtype)

        # Extract the values from the node
        mapping_data = {}
        for key_node, value_node in result.value:
            mapping_data[key_node.value] = value_node.value

        # Verify the result
        assert mapping_data["_target_"] == str(dtype)
        assert mapping_data["_call_"] == "false" or mapping_data["_call_"] is False

    except ImportError:
        pytest.skip("PyTorch not installed, skipping torch dtype tests")


# Custom class for testing that properly triggers the first branch
class CustomObjWithQualName:
    """Custom object with __qualname__ for testing."""

    pass


# Add __qualname__ directly to the class object
CustomObjWithQualName.__qualname__ = "CustomQualname"


# Test object with __qualname__ will be serialized with call=False
@patch("inspect.getmodule")
def test_safe_object_representer(mock_getmodule):
    """Test the safe object representer."""
    # Set up the mock to return a module with a name
    # However, for objects defined *within* this test file, inspect.getmodule will correctly get their module
    # So, we need to make sure the expected target reflects the new module path
    current_test_module_name = __name__  # This will be tests.unit_tests.utils.test_yaml_utils

    # Create a dummy dumper
    dumper = yaml.SafeDumper(None)

    # Test case 1: We'll use our custom function object to ensure __qualname__ exists directly on the object
    # Create a callable function-like object that has __qualname__
    def test_func():
        pass

    mock_getmodule.return_value = inspect.getmodule(test_func)  # Ensure it uses the correct module for test_func

    obj = test_func

    # The first branch should be taken (_call_=False) since functions have __qualname__
    result = _safe_object_representer(dumper, obj)

    # Extract the values from the node
    mapping_data = {}
    for key_node, value_node in result.value:
        mapping_data[key_node.value] = value_node.value

    # Verify the result is using the first branch (call=False)
    assert mapping_data["_target_"] == f"{current_test_module_name}.{test_func.__qualname__}"
    assert mapping_data["_call_"] == "false" or mapping_data["_call_"] is False

    # Test case 2: Regular class instance (falls back to __class__)
    class SimpleTestClass:
        """Simple test class for testing."""

        pass

    mock_getmodule.return_value = inspect.getmodule(SimpleTestClass)  # Ensure correct module for SimpleTestClass
    obj = SimpleTestClass()

    # When serializing normal instances, _call_ should be True
    result = _safe_object_representer(dumper, obj)

    # Extract the values from the node
    mapping_data = {}
    for key_node, value_node in result.value:
        mapping_data[key_node.value] = value_node.value

    # Verify it uses the object's class with _call_=True
    assert mapping_data["_target_"] == f"{current_test_module_name}.{SimpleTestClass.__qualname__}"
    assert mapping_data["_call_"] == "true" or mapping_data["_call_"] is True


def test_full_yaml_dump():
    """Test a complete YAML dump with various object types."""
    current_test_module_name = __name__

    def local_function():
        pass

    # Create test data with various types
    test_data = {
        "function": dummy_function,
        "method": DummyClass.test_method,
        "static_method": DummyClass.static_method,
        "class_method": DummyClass.class_method,
        "lambda": lambda x: x * 2,
        "class_instance": DummyClass(),
        "nested": {
            "function": local_function,
        },
    }

    # Try to add torch dtype if available
    try:
        import torch

        test_data["dtype"] = torch.float32
    except ImportError:
        pass

    # Dump to YAML using our context manager
    with safe_yaml_representers():
        yaml_str = yaml.safe_dump(test_data)

    # Verify the result
    assert "_target_" in yaml_str
    assert f"{current_test_module_name}.DummyClass" in yaml_str
    assert "function" in yaml_str
    assert "method" in yaml_str
    assert "_call_: false" in yaml_str
    assert "_call_: true" in yaml_str  # For class instance
    assert f"{current_test_module_name}.dummy_function" in yaml_str
    assert f"{current_test_module_name}.local_function" in yaml_str  # for nested local_function


def test_serialize_methods():
    """Test serialization of different method types."""
    current_test_module_name = __name__
    # Create test methods
    instance_method = DummyClass().test_method
    static_method = DummyClass.static_method
    class_method = DummyClass.class_method

    # Dump to YAML using our context manager
    with safe_yaml_representers():
        instance_yaml = yaml.safe_dump(instance_method)
        static_yaml = yaml.safe_dump(static_method)
        class_yaml = yaml.safe_dump(class_method)

    # Verify the results
    assert f"{current_test_module_name}.DummyClass.test_method" in instance_yaml
    assert "_call_: false" in instance_yaml

    # These assertions will check for the method name in the target
    assert f"{current_test_module_name}.DummyClass.static_method" in static_yaml
    assert "_call_: false" in static_yaml

    assert f"{current_test_module_name}.DummyClass.class_method" in class_yaml
    assert "_call_: false" in class_yaml


def test_torch_module_not_found():
    """Test torch module not found branch."""
    # Save the original state of yaml.SafeDumper
    original_representers = yaml.SafeDumper.yaml_representers.copy()

    try:
        # Mock an import error for torch
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ModuleNotFoundError("Mocked torch not found")
            return original_import(name, *args, **kwargs)

        # Use the mock
        builtins.__import__ = mock_import

        # Call the context manager which should attempt to import torch
        with safe_yaml_representers():
            pass

        # If we get here, the ModuleNotFoundError was properly caught

    finally:
        # Restore the original import function
        builtins.__import__ = original_import
        # Restore the original representers
        yaml.SafeDumper.yaml_representers = original_representers


def test_torch_dtype_representer_direct():
    """Test torch dtype representer directly."""
    try:
        import torch

        # Create a mock dumper that will record the calls
        class MockDumper:
            """Mock dumper for testing."""

            def __init__(self):
                self.represented_data = None

            def represent_data(self, data):
                """Represent data."""
                self.represented_data = data
                return data

        dumper = MockDumper()
        dtype = torch.float32

        # Call the representer directly
        _torch_dtype_representer(dumper, dtype)

        # Verify the call to represent_data
        assert dumper.represented_data is not None
        assert dumper.represented_data["_target_"] == str(dtype)
        assert dumper.represented_data["_call_"] is False

    except ImportError:
        pytest.skip("PyTorch not installed, skipping torch dtype tests")


@patch("nemo_lm.utils.yaml_utils._safe_object_representer")
def test_safe_object_representer_edge_cases(mock_representer):
    """Test edge cases in the safe_object_representer function."""
    current_test_module_name = __name__
    # Create a dummy dumper
    dumper = yaml.SafeDumper(None)

    # Create a test object
    class CustomObj:  # Defined in this scope, so its module is this test file
        """Custom object for testing."""

        pass

    obj = CustomObj()

    # Mock the implementation to return a valid result
    # The _target_ should reflect the new path of CustomObj
    mock_representer.return_value = dumper.represent_data(
        {"_target_": f"{current_test_module_name}.{CustomObj.__qualname__}", "_call_": True}
    )

    # Run this inside the safe_yaml_representers context
    with safe_yaml_representers():
        # This should use our mocked function
        result = yaml.safe_dump(obj)

        # The call to our mocked function should happen
        mock_representer.assert_called_once()

        # Verify the output
        assert CustomObj.__qualname__ in result  # Check for class name
        assert "_call_: true" in result


def test_custom_safe_yaml_representers():
    """Test registering custom representers inside the context manager."""
    current_test_module_name = __name__

    # Create a custom class
    class CustomClass:  # Defined in this scope
        """Custom class for testing."""
        pass

    # Test with a custom representer
    def custom_representer(dumper, data):
        value = {"_special_": "custom"}
        return dumper.represent_data(value)

    with safe_yaml_representers():
        # Add our custom representer inside the context
        yaml.SafeDumper.add_representer(CustomClass, custom_representer)

        # Test the custom representer
        result = yaml.safe_dump(CustomClass())
        assert "_special_: custom" in result
        # The representer for CustomClass should be specific to it.
        # If we dump CustomClass itself (the type), it should use the default obj representer.
        type_result = yaml.safe_dump(CustomClass)
        assert f"{current_test_module_name}.{CustomClass.__qualname__}" in type_result

    # Verify our custom representer was removed
    with pytest.raises(yaml.representer.RepresenterError):
        yaml.safe_dump(CustomClass())


def test_partial_representer():
    """Test the serialization of partial objects to YAML and back."""
    current_test_module_name = __name__
    # Test with a simple partial function without args
    simple_partial = partial(dummy_function)

    # Test with a partial function with args and kwargs
    complex_partial = partial(dummy_function_with_args, 10, c=30)

    # Test with a method
    method_partial = partial(DummyClass.test_method)

    with safe_yaml_representers():
        # Test simple partial
        yaml_str = yaml.safe_dump(simple_partial)
        loaded_data = yaml.safe_load(yaml_str)

        # Verify the loaded data structure
        assert "_target_" in loaded_data
        assert loaded_data["_target_"] == f"{current_test_module_name}.dummy_function"
        assert loaded_data["_partial_"] is True
        assert "_args_" in loaded_data
        assert loaded_data["_args_"] == []

        # Test complex partial with args and kwargs
        yaml_str = yaml.safe_dump(complex_partial)
        loaded_data = yaml.safe_load(yaml_str)

        # Verify the complex partial structure
        assert "_target_" in loaded_data
        assert loaded_data["_target_"] == f"{current_test_module_name}.dummy_function_with_args"
        assert loaded_data["_partial_"] is True
        assert "_args_" in loaded_data
        assert loaded_data["_args_"] == [10]
        assert "c" in loaded_data
        assert loaded_data["c"] == 30

        # Test method partial
        yaml_str = yaml.safe_dump(method_partial)
        loaded_data = yaml.safe_load(yaml_str)

        # Verify the method partial structure
        assert "_target_" in loaded_data
        assert loaded_data["_target_"] == f"{current_test_module_name}.DummyClass.test_method"
        assert loaded_data["_partial_"] is True
        assert "_args_" in loaded_data
        assert loaded_data["_args_"] == []


def test_full_yaml_dump_with_partial():
    """Test YAML dump with partial objects."""
    current_test_module_name = __name__
    # Create test data with partial functions
    test_data = {
        "simple_partial": partial(dummy_function),
        "complex_partial": partial(dummy_function_with_args, 10, c=30),
        "method_partial": partial(DummyClass.test_method),
    }

    # Dump to YAML using our context manager
    with safe_yaml_representers():
        yaml_str = yaml.safe_dump(test_data)

    # Verify the result contains partial information
    assert "_target_" in yaml_str
    assert "_partial_: true" in yaml_str
    assert f"{current_test_module_name}.dummy_function" in yaml_str
    assert f"{current_test_module_name}.dummy_function_with_args" in yaml_str
    assert "c: 30" in yaml_str
    assert f"{current_test_module_name}.DummyClass.test_method" in yaml_str


def test_dump_dataclass_to_yaml_file():
    """Test dump_dataclass_to_yaml with a filename."""
    current_test_module_name = __name__
    instance = DummyDataClass(name="test_file", value=123)
    mock_file = mock_open()
    with patch("builtins.open", mock_file):
        with safe_yaml_representers():  # Ensure representers are active for dump_dataclass_to_yaml
            dump_dataclass_to_yaml(instance, "dummy.yaml")

    mock_file.assert_called_once_with("dummy.yaml", "w+")
    # Check that safe_dump was called with the instance
    # This is a bit indirect, but mock_open().write() is hard to assert on directly for content with yaml.safe_dump
    # So we check that open was called, and trust yaml.safe_dump works if representers are correct
    # A more robust check would involve reading back from a real temp file.
    handle = mock_file()
    # Get all write calls and join them
    written_content = "".join(call_arg[0] for call_arg in handle.write.call_args_list)

    assert "_target_" in written_content
    assert f"{current_test_module_name}.DummyDataClass" in written_content
    assert "name: test_file" in written_content
    assert "value: 123" in written_content


def test_dump_dataclass_to_yaml_string():
    """Test dump_dataclass_to_yaml without a filename (returns string)."""
    current_test_module_name = __name__
    instance = DummyDataClass(name="test_string", value=456)
    with safe_yaml_representers():  # Ensure representers are active
        yaml_string = dump_dataclass_to_yaml(instance)

    assert isinstance(yaml_string, str)
    assert "_target_" in yaml_string
    assert f"{current_test_module_name}.DummyDataClass" in yaml_string
    assert "name: test_string" in yaml_string
    assert "value: 456" in yaml_string


class Color(enum.Enum):
    """Color enum for testing."""

    RED = 1
    GREEN = 2
    BLUE = 3


def test_enum_representer():
    """Test the enum representer."""
    current_test_module_name = __name__
    dumper = yaml.SafeDumper(None)

    # Test with an enum member
    result = _enum_representer(dumper, Color.RED)

    assert hasattr(result, "value")
    mapping_data = {}
    for key_node, value_node in result.value:
        mapping_data[key_node.value] = value_node.value

    # inspect.getmodule(Color) is correct here
    expected_target = f"{inspect.getmodule(Color).__name__}.{Color.__qualname__}"
    assert mapping_data["_target_"] == expected_target
    assert mapping_data["_call_"] is True or mapping_data["_call_"] == "true"
    assert mapping_data["_args_"] == [1]  # Check the value of the enum

    # Test directly with safe_dump
    with safe_yaml_representers():
        yaml_str = yaml.safe_dump(Color.GREEN)
        assert "_target_:" in yaml_str
        assert f"{current_test_module_name}.Color" in yaml_str  # __name__ should be the current module path
        assert "_call_: true" in yaml_str
        assert "_args_:[2]" in yaml_str or "_args_: [2]" in yaml_str  # PyYAML might add space


def test_generation_config_representer():
    """Test the GenerationConfig representer."""
    try:
        from transformers import GenerationConfig

        dumper = yaml.SafeDumper(None)
        config = GenerationConfig(max_length=50, temperature=0.7)

        result = _generation_config_representer(dumper, config)

        assert hasattr(result, "value")
        mapping_data = {}
        for key_node, value_node in result.value:
            # Handle the case where config_dict is a Representer instance itself
            if hasattr(value_node, "tag"):  # It's a node, so extract its value
                if isinstance(key_node.value, str) and key_node.value == "config_dict":
                    # It's the config_dict, which is a mapping node
                    config_dict_val = {}
                    for k_node, v_node in value_node.value:
                        config_dict_val[k_node.value] = v_node.value
                    mapping_data[key_node.value] = config_dict_val
                else:
                    mapping_data[key_node.value] = value_node.value
            else:  # It's a direct value
                mapping_data[key_node.value] = value_node

        expected_target = f"{inspect.getmodule(GenerationConfig).__name__}.{GenerationConfig.__qualname__}.from_dict"
        assert mapping_data["_target_"] == expected_target
        assert mapping_data["_call_"] is True or mapping_data["_call_"] == "true"
        assert mapping_data["config_dict"]["max_length"] == 50
        assert mapping_data["config_dict"]["temperature"] == 0.7

        # Test directly with safe_dump
        with safe_yaml_representers():
            yaml_str = yaml.safe_dump(config)
            assert "_target_:" in yaml_str
            assert "transformers.generation_config.GenerationConfig.from_dict" in yaml_str
            assert "_call_: true" in yaml_str
            assert "max_length: 50" in yaml_str
            assert "temperature: 0.7" in yaml_str

    except ImportError:
        pytest.skip("Transformers not installed, skipping GenerationConfig tests")


def test_generation_config_module_not_found():
    """Test GenerationConfig representer when transformers module is not found."""
    original_representers = yaml.SafeDumper.yaml_representers.copy()
    original_multi_representers = yaml.SafeDumper.yaml_multi_representers.copy()

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "transformers":
            raise ModuleNotFoundError("Mocked transformers not found")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = mock_import

    try:
        with safe_yaml_representers():
            # At this point, the representer for GenerationConfig should not have been added
            # We can check this by trying to dump a dummy object that would otherwise be handled
            # by a more generic representer or cause an error if no representer is found.
            # The absence of an error here during context manager setup implies it handled the ModuleNotFoundError.
            pass
        # Check that GenerationConfig is not in the representers (indirectly)
        # A more robust check would be to try to dump an object that ONLY GenerationConfig would handle
        # but that's hard to set up if the module is truly absent.
        # We can assert that the representer for a known type (like function) is still there,
        # and that no new error occurred.
        assert type(lambda: ...) in yaml.SafeDumper.yaml_representers

    finally:
        builtins.__import__ = original_import
        yaml.SafeDumper.yaml_representers = original_representers
        yaml.SafeDumper.yaml_multi_representers = original_multi_representers
