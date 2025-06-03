#!/usr/bin/env python3
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

"""Tests for omegaconf_utils module."""

import dataclasses
import functools
from typing import Any, Dict
from unittest.mock import Mock

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from nemo_lm.utils.omegaconf_utils import (
    OverridesError,
    apply_overrides_recursively,
    apply_overrides_with_preservation,
    dataclass_to_dict_for_omegaconf,
    is_problematic_callable,
    parse_hydra_overrides,
    safe_create_omegaconf,
    safe_create_omegaconf_with_preservation,
    _track_excluded_fields,
    _restore_excluded_fields,
    _verify_no_callables,
)


# Test dataclasses for testing
@dataclasses.dataclass
class SimpleConfig:
    """Simple config for testing."""

    name: str = "test"
    value: int = 42


@dataclasses.dataclass
class ConfigWithCallable:
    """Config with callable fields for testing."""

    name: str = "test"
    activation_func: Any = torch.nn.functional.relu
    dtype: torch.dtype = torch.float32


@dataclasses.dataclass
class NestedConfig:
    """Nested config for testing."""

    simple: SimpleConfig = dataclasses.field(default_factory=SimpleConfig)
    with_callable: ConfigWithCallable = dataclasses.field(default_factory=ConfigWithCallable)


def dummy_function():
    """Dummy function for testing callable detection."""
    return "dummy"


def another_function(x: int) -> int:
    """Another dummy function."""
    return x * 2


class TestIsProblematicCallable:
    """Test is_problematic_callable function."""

    def test_non_callable(self):
        """Test with non-callable values."""
        assert not is_problematic_callable(42)
        assert not is_problematic_callable("string")
        assert not is_problematic_callable([1, 2, 3])
        assert not is_problematic_callable({"key": "value"})
        assert not is_problematic_callable(None)

    def test_class_types_allowed(self):
        """Test that class types are allowed (not problematic)."""
        assert not is_problematic_callable(str)
        assert not is_problematic_callable(int)
        assert not is_problematic_callable(SimpleConfig)
        assert not is_problematic_callable(torch.nn.ReLU)

    def test_function_objects_problematic(self):
        """Test that function objects are problematic."""
        assert is_problematic_callable(dummy_function)
        assert is_problematic_callable(another_function)
        assert is_problematic_callable(torch.nn.functional.relu)

    def test_partial_functions_problematic(self):
        """Test that partial functions are problematic."""
        partial_func = functools.partial(another_function, x=5)
        assert is_problematic_callable(partial_func)

    def test_lambda_functions_problematic(self):
        """Test that lambda functions are problematic."""
        lambda_func = lambda x: x * 2
        assert is_problematic_callable(lambda_func)

    def test_methods_problematic(self):
        """Test that instance methods are problematic."""

        class TestClass:
            def instance_method(self):
                return "instance"

            @classmethod
            def class_method(cls):
                return "class"

            @staticmethod
            def static_method():
                return "static"

        obj = TestClass()

        # Instance methods should be problematic
        assert is_problematic_callable(obj.instance_method)

        # Class methods should be problematic (they're bound methods)
        assert is_problematic_callable(obj.class_method)

        # Static methods should be problematic (they're function objects)
        assert is_problematic_callable(obj.static_method)

        # But accessing them from the class should behave differently
        assert is_problematic_callable(TestClass.instance_method)  # unbound method
        assert is_problematic_callable(TestClass.class_method)  # bound class method
        assert is_problematic_callable(TestClass.static_method)  # function object


class TestDataclassToDict:
    """Test dataclass_to_dict_for_omegaconf function."""

    def test_simple_dataclass(self):
        """Test conversion of simple dataclass."""
        config = SimpleConfig(name="test", value=100)
        result = dataclass_to_dict_for_omegaconf(config)

        expected = {"name": "test", "value": 100}
        assert result == expected

    def test_torch_dtype_conversion(self):
        """Test that torch.dtype is converted to string."""
        config = ConfigWithCallable(dtype=torch.float16)
        result = dataclass_to_dict_for_omegaconf(config)

        assert result["dtype"] == "torch.float16"
        assert result["name"] == "test"
        # activation_func should be excluded (None not included)
        assert "activation_func" not in result

    def test_callable_exclusion(self):
        """Test that callable fields are excluded."""
        config = ConfigWithCallable(activation_func=torch.nn.functional.gelu)
        result = dataclass_to_dict_for_omegaconf(config)

        assert "activation_func" not in result
        assert "name" in result
        assert "dtype" in result

    def test_nested_dataclass(self):
        """Test conversion of nested dataclasses."""
        config = NestedConfig()
        result = dataclass_to_dict_for_omegaconf(config)

        assert "simple" in result
        assert "with_callable" in result
        assert result["simple"]["name"] == "test"
        assert result["simple"]["value"] == 42
        assert "activation_func" not in result["with_callable"]

    def test_list_handling(self):
        """Test handling of lists."""
        test_list = [SimpleConfig(name="item1"), SimpleConfig(name="item2")]
        result = dataclass_to_dict_for_omegaconf(test_list)

        assert len(result) == 2
        assert result[0]["name"] == "item1"
        assert result[1]["name"] == "item2"

    def test_tuple_handling(self):
        """Test handling of tuples."""
        test_tuple = (SimpleConfig(name="item1"), "string", 42)
        result = dataclass_to_dict_for_omegaconf(test_tuple)

        assert len(result) == 3
        assert result[0]["name"] == "item1"
        assert result[1] == "string"
        assert result[2] == 42

    def test_dict_handling(self):
        """Test handling of dictionaries."""
        test_dict = {"config": SimpleConfig(name="test"), "value": 42, "func": dummy_function}  # Should be excluded
        result = dataclass_to_dict_for_omegaconf(test_dict)

        assert "config" in result
        assert "value" in result
        assert "func" not in result  # Excluded callable
        assert result["config"]["name"] == "test"

    def test_primitive_types(self):
        """Test handling of primitive types."""
        assert dataclass_to_dict_for_omegaconf(42) == 42
        assert dataclass_to_dict_for_omegaconf("string") == "string"
        assert dataclass_to_dict_for_omegaconf(True) is True
        assert dataclass_to_dict_for_omegaconf(None) is None


class TestDataclassToDict_ErrorHandling:
    """Test error handling in dataclass_to_dict_for_omegaconf function."""

    def test_error_handling_specific_exceptions(self):
        """Test that specific exceptions are caught gracefully."""

        @dataclasses.dataclass
        class ProblematicConfig:
            name: str = "test"

            @property
            def problematic_property(self):
                raise AttributeError("Intentional error")

        config = ProblematicConfig()

        # Should handle AttributeError gracefully and continue processing
        result = dataclass_to_dict_for_omegaconf(config)

        # Should still get the valid field
        assert result["name"] == "test"
        # Should not include the problematic field
        assert "problematic_property" not in result

    def test_error_handling_unexpected_exceptions_are_raised(self):
        """Test that unexpected exceptions are not swallowed."""

        @dataclasses.dataclass
        class BadConfig:
            name: str = "test"

            @property
            def explosive_property(self):
                raise ValueError("This should not be caught!")

        config = BadConfig()

        # ValueError should not be caught and should bubble up
        with pytest.raises(ValueError, match="This should not be caught!"):
            dataclass_to_dict_for_omegaconf(config)


class TestTrackExcludedFields:
    """Test track_excluded_fields function."""

    def test_simple_tracking(self):
        """Test tracking callable fields in simple config."""
        config = ConfigWithCallable()
        excluded = _track_excluded_fields(config)

        assert "activation_func" in excluded
        assert excluded["activation_func"] == torch.nn.functional.relu

    def test_nested_tracking(self):
        """Test tracking callable fields in nested config."""
        config = NestedConfig()
        excluded = _track_excluded_fields(config, "root")

        assert "root.with_callable.activation_func" in excluded

    def test_no_callables(self):
        """Test tracking when no callables exist."""
        config = SimpleConfig()
        excluded = _track_excluded_fields(config)

        assert len(excluded) == 0

    def test_dict_with_callables(self):
        """Test tracking callables in dictionary fields."""

        @dataclasses.dataclass
        class ConfigWithDict:
            funcs: Dict[str, Any] = dataclasses.field(
                default_factory=lambda: {"relu": torch.nn.functional.relu, "value": 42}
            )

        config = ConfigWithDict()
        excluded = _track_excluded_fields(config)

        assert "funcs.relu" in excluded


class TestRestoreExcludedFields:
    """Test restore_excluded_fields function."""

    def test_simple_restoration(self):
        """Test restoring excluded fields."""
        config = ConfigWithCallable()
        original_func = config.activation_func

        # Simulate exclusion by setting to None
        config.activation_func = None

        excluded = {"activation_func": original_func}
        _restore_excluded_fields(config, excluded)

        assert config.activation_func == original_func

    def test_nested_restoration(self):
        """Test restoring nested excluded fields."""
        config = NestedConfig()
        original_func = config.with_callable.activation_func

        # Simulate exclusion
        config.with_callable.activation_func = None

        excluded = {"root.with_callable.activation_func": original_func}
        _restore_excluded_fields(config, excluded)

        assert config.with_callable.activation_func == original_func

    def test_invalid_path_handling(self):
        """Test handling of invalid field paths."""
        config = SimpleConfig()
        excluded = {"nonexistent.field": dummy_function}

        # Should not raise exception, just log warning
        _restore_excluded_fields(config, excluded)


class TestVerifyNoCallables:
    """Test verify_no_callables function."""

    def test_clean_dict(self):
        """Test verification of dictionary without callables."""
        test_dict = {"name": "test", "value": 42, "nested": {"key": "value"}}
        assert _verify_no_callables(test_dict)

    def test_dict_with_callables(self):
        """Test verification fails with callables present."""
        test_dict = {"name": "test", "func": dummy_function}
        assert not _verify_no_callables(test_dict)

    def test_clean_list(self):
        """Test verification of list without callables."""
        test_list = [1, 2, "string", {"key": "value"}]
        assert _verify_no_callables(test_list)

    def test_list_with_callables(self):
        """Test verification fails with callables in list."""
        test_list = [1, 2, dummy_function]
        assert not _verify_no_callables(test_list)


class TestSafeCreateOmegaconf:
    """Test safe_create_omegaconf function."""

    def test_simple_config_creation(self):
        """Test creating OmegaConf from simple dataclass."""
        config = SimpleConfig()
        omega_conf = safe_create_omegaconf(config)

        assert isinstance(omega_conf, DictConfig)
        assert omega_conf.name == "test"
        assert omega_conf.value == 42

    def test_config_with_exclusions(self):
        """Test creating OmegaConf with callable exclusions."""
        config = ConfigWithCallable()
        omega_conf = safe_create_omegaconf(config)

        assert isinstance(omega_conf, DictConfig)
        assert omega_conf.name == "test"
        assert "activation_func" not in omega_conf

    def test_nested_config_creation(self):
        """Test creating OmegaConf from nested dataclass."""
        config = NestedConfig()
        omega_conf = safe_create_omegaconf(config)

        assert isinstance(omega_conf, DictConfig)
        assert omega_conf.simple.name == "test"
        assert omega_conf.with_callable.name == "test"
        assert "activation_func" not in omega_conf.with_callable


class TestSafeCreateOmegaconfWithPreservation:
    """Test safe_create_omegaconf_with_preservation function."""

    def test_preservation_tracking(self):
        """Test that callable preservation tracking works."""
        config = ConfigWithCallable()
        omega_conf, excluded = safe_create_omegaconf_with_preservation(config)

        assert isinstance(omega_conf, DictConfig)
        assert len(excluded) > 0
        assert "root.activation_func" in excluded
        assert excluded["root.activation_func"] == torch.nn.functional.relu

    def test_nested_preservation(self):
        """Test preservation with nested configs."""
        config = NestedConfig()
        omega_conf, excluded = safe_create_omegaconf_with_preservation(config)

        assert isinstance(omega_conf, DictConfig)
        assert "root.with_callable.activation_func" in excluded


class TestApplyOverridesRecursively:
    """Test apply_overrides_recursively function."""

    def test_simple_override(self):
        """Test applying simple overrides."""
        config = SimpleConfig()
        overrides = {"name": "updated", "value": 100}

        apply_overrides_recursively(config, overrides)

        assert config.name == "updated"
        assert config.value == 100

    def test_nested_override(self):
        """Test applying nested overrides."""
        config = NestedConfig()
        overrides = {"simple": {"name": "nested_updated", "value": 200}, "with_callable": {"name": "callable_updated"}}

        apply_overrides_recursively(config, overrides)

        assert config.simple.name == "nested_updated"
        assert config.simple.value == 200
        assert config.with_callable.name == "callable_updated"

    def test_torch_dtype_conversion(self):
        """Test torch.dtype string conversion."""
        config = ConfigWithCallable()
        overrides = {"dtype": "torch.float16"}

        apply_overrides_recursively(config, overrides)

        assert config.dtype == torch.float16

    def test_invalid_key_handling(self):
        """Test handling of invalid override keys."""
        config = SimpleConfig()
        overrides = {"nonexistent": "value"}

        # Should not raise exception, just log warning
        apply_overrides_recursively(config, overrides)

        # Original values should be unchanged
        assert config.name == "test"
        assert config.value == 42


class TestApplyOverridesWithPreservation:
    """Test apply_overrides_with_preservation function."""

    def test_preservation_workflow(self):
        """Test complete override workflow with preservation."""
        config = ConfigWithCallable()
        original_func = config.activation_func

        # Track excluded fields
        excluded = {"root.activation_func": original_func}

        # Apply overrides
        overrides = {"name": "preserved_test"}
        apply_overrides_with_preservation(config, overrides, excluded)

        # Check that overrides were applied and callables restored
        assert config.name == "preserved_test"
        assert config.activation_func == original_func


class TestParseHydraOverrides:
    """Test parse_hydra_overrides function."""

    def test_simple_override(self):
        """Test parsing simple Hydra overrides."""
        cfg = OmegaConf.create({"name": "test", "value": 42})
        overrides = ["name=updated", "value=100"]

        result = parse_hydra_overrides(cfg, overrides)

        assert result.name == "updated"
        assert result.value == 100

    def test_addition_override(self):
        """Test Hydra addition syntax."""
        cfg = OmegaConf.create({"name": "test"})
        overrides = ["+new_param=added_value"]

        result = parse_hydra_overrides(cfg, overrides)

        assert result.name == "test"
        assert result.new_param == "added_value"

    def test_nested_override(self):
        """Test nested override syntax."""
        cfg = OmegaConf.create({"section": {"name": "test", "value": 42}})
        overrides = ["section.name=updated", "section.value=200"]

        result = parse_hydra_overrides(cfg, overrides)

        assert result.section.name == "updated"
        assert result.section.value == 200

    def test_invalid_override(self):
        """Test handling of invalid overrides."""
        cfg = OmegaConf.create({"name": "test"})
        invalid_overrides = ["invalid syntax"]

        with pytest.raises(OverridesError):
            parse_hydra_overrides(cfg, invalid_overrides)


class TestOverridesError:
    """Test OverridesError exception."""

    def test_exception_creation(self):
        """Test creating OverridesError."""
        error = OverridesError("Test error message")
        assert str(error) == "Test error message"

    def test_exception_inheritance(self):
        """Test that OverridesError inherits from Exception."""
        error = OverridesError("Test")
        assert isinstance(error, Exception)


# Integration tests
class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_workflow(self):
        """Test the complete workflow from dataclass to overrides."""
        # 1. Create config with callables
        config = NestedConfig()
        original_func = config.with_callable.activation_func
        original_dtype = config.with_callable.dtype

        # 2. Convert to OmegaConf with preservation
        omega_conf, excluded = safe_create_omegaconf_with_preservation(config)

        # Verify initial OmegaConf state
        assert omega_conf.simple.name == "test"
        assert omega_conf.simple.value == 42
        assert omega_conf.with_callable.name == "test"
        assert "activation_func" not in omega_conf.with_callable  # Excluded
        assert omega_conf.with_callable.dtype == "torch.float32"  # Converted to string
        assert len(excluded) > 0  # Should have excluded callables

        # 3. Apply YAML-style overrides
        yaml_overrides = OmegaConf.create(
            {"simple": {"name": "yaml_updated"}, "with_callable": {"name": "yaml_callable", "dtype": "torch.float16"}}
        )
        merged_conf = OmegaConf.merge(omega_conf, yaml_overrides)

        # Verify YAML overrides applied
        assert merged_conf.simple.name == "yaml_updated"
        assert merged_conf.with_callable.name == "yaml_callable"
        assert merged_conf.with_callable.dtype == "torch.float16"

        # 4. Apply Hydra-style CLI overrides (only to existing fields)
        cli_overrides = ["simple.value=999", "with_callable.name=cli_updated"]
        final_conf = parse_hydra_overrides(merged_conf, cli_overrides)

        # Verify CLI overrides applied
        assert final_conf.simple.value == 999
        assert final_conf.with_callable.name == "cli_updated"  # CLI wins over YAML

        # 5. Convert back to dict and apply to original config
        final_dict = OmegaConf.to_container(final_conf, resolve=True)
        assert isinstance(final_dict, dict)
        apply_overrides_with_preservation(config, final_dict, excluded)

        # 6. Verify final results
        assert config.simple.name == "yaml_updated"  # From YAML
        assert config.simple.value == 999  # From CLI
        assert config.with_callable.name == "cli_updated"  # CLI override wins over YAML
        assert config.with_callable.activation_func == original_func  # Preserved
        assert config.with_callable.dtype == torch.float16  # Type converted back correctly

        # 7. Verify that excluded fields were properly tracked and restored
        assert "root.with_callable.activation_func" in excluded
        assert excluded["root.with_callable.activation_func"] == original_func

        # Note: New fields are not added to dataclasses (this is expected behavior)
        # The dataclass structure remains strongly typed

    def test_torch_dtype_roundtrip(self):
        """Test torch.dtype conversion roundtrip."""
        config = ConfigWithCallable(dtype=torch.float16)
        original_dtype = config.dtype

        # Convert to OmegaConf
        omega_conf = safe_create_omegaconf(config)

        # Verify dtype was converted to string
        assert omega_conf.dtype == "torch.float16"
        assert isinstance(omega_conf.dtype, str)

        # Convert back and apply
        config_dict = OmegaConf.to_container(omega_conf, resolve=True)
        apply_overrides_recursively(config, config_dict)

        # Verify dtype was converted back correctly
        assert config.dtype == torch.float16
        assert isinstance(config.dtype, torch.dtype)
        assert config.dtype == original_dtype

    def test_hydra_addition_vs_dataclass_limitation(self):
        """Test that Hydra addition syntax works in OmegaConf but dataclass application is limited."""
        # 1. Create config and convert to OmegaConf
        config = NestedConfig()
        original_func = config.with_callable.activation_func
        omega_conf, excluded = safe_create_omegaconf_with_preservation(config)

        # 2. Apply Hydra overrides with addition syntax
        cli_overrides = ["simple.value=999", "+new_section.param=added_value", "+new_section.nested.deep=deep_value"]
        final_conf = parse_hydra_overrides(omega_conf, cli_overrides)

        # 3. Verify that OmegaConf contains the new sections
        assert final_conf.simple.value == 999
        assert final_conf.new_section.param == "added_value"
        assert final_conf.new_section.nested.deep == "deep_value"

        # Verify the structure is as expected
        assert isinstance(final_conf.new_section, DictConfig)
        assert isinstance(final_conf.new_section.nested, DictConfig)

        # 4. Convert back to dict and apply to dataclass
        final_dict = OmegaConf.to_container(final_conf, resolve=True)

        # Verify the dict contains the new sections
        assert "new_section" in final_dict
        assert final_dict["new_section"]["param"] == "added_value"
        assert final_dict["new_section"]["nested"]["deep"] == "deep_value"

        apply_overrides_with_preservation(config, final_dict, excluded)

        # 5. Verify dataclass behavior:
        # - Existing fields are updated
        assert config.simple.value == 999
        # - Callable fields are preserved
        assert config.with_callable.activation_func == original_func
        # - New fields are not added to the dataclass (logged as warning)
        assert not hasattr(config, "new_section")

        # 6. Verify the original dataclass structure is intact
        assert hasattr(config, "simple")
        assert hasattr(config, "with_callable")
        assert isinstance(config.simple, SimpleConfig)
        assert isinstance(config.with_callable, ConfigWithCallable)
