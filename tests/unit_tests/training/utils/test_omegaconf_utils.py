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
from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from megatron.bridge.training.utils.omegaconf_utils import (
    OverridesError,
    _apply_overrides,
    _dataclass_to_omegaconf_dict,
    _is_omegaconf_problematic,
    _restore_excluded_fields,
    _track_excluded_fields,
    _verify_no_callables,
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
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


@dataclasses.dataclass
class ConfigWithOptionalFields:
    """Config with optional fields for testing None handling."""

    name: str = "test"
    optional_str: str | None = None  # Should be preserved
    optional_int: int | None = None  # Should be preserved
    activation_func: Any = torch.nn.functional.relu  # Should be excluded
    value: int = 42


@dataclasses.dataclass
class ProfilingConfig:
    """Mock ProfilingConfig for testing _target_ functionality."""

    use_pytorch_profiler: bool = False
    profile_ranks: list = dataclasses.field(default_factory=list)
    pytorch_profiler_config: dict | None = None


@dataclasses.dataclass
class ConfigWithOptionalProfilingConfig:
    """Config with optional profiling config for testing _target_ mechanism."""

    name: str = "test"
    profiling: ProfilingConfig | None = None


def dummy_function():
    """Dummy function for testing callable detection."""
    return "dummy"


def another_function(x: int) -> int:
    """Another dummy function."""
    return x * 2


class TestIsOmegaconfProblematic:
    """Test _is_omegaconf_problematic function."""

    def test_non_callable(self):
        """Test with non-callable values."""
        assert not _is_omegaconf_problematic(42)
        assert not _is_omegaconf_problematic("string")
        assert not _is_omegaconf_problematic([1, 2, 3])
        assert not _is_omegaconf_problematic({"key": "value"})
        assert not _is_omegaconf_problematic(None)

    def test_class_types_allowed(self):
        """Test that class types are allowed (not problematic)."""
        assert not _is_omegaconf_problematic(str)
        assert not _is_omegaconf_problematic(int)
        assert not _is_omegaconf_problematic(SimpleConfig)
        assert not _is_omegaconf_problematic(torch.nn.ReLU)

    def test_function_objects_problematic(self):
        """Test that function objects are problematic."""
        assert _is_omegaconf_problematic(dummy_function)
        assert _is_omegaconf_problematic(another_function)
        assert _is_omegaconf_problematic(torch.nn.functional.relu)

    def test_partial_functions_problematic(self):
        """Test that partial functions are problematic."""
        partial_func = functools.partial(another_function, x=5)
        assert _is_omegaconf_problematic(partial_func)

    def test_lambda_functions_problematic(self):
        """Test that lambda functions are problematic."""
        lambda_func = lambda x: x * 2
        assert _is_omegaconf_problematic(lambda_func)

    def test_methods_problematic(self):
        """Test that instance methods are problematic."""

        class TestClass:
            """Test class for method testing."""

            def instance_method(self):
                """Test instance method."""
                return "instance"

            @classmethod
            def class_method(cls):
                """Test class method."""
                return "class"

            @staticmethod
            def static_method():
                """Test static method."""
                return "static"

        obj = TestClass()

        # Instance methods should be problematic
        assert _is_omegaconf_problematic(obj.instance_method)

        # Class methods should be problematic (they're bound methods)
        assert _is_omegaconf_problematic(obj.class_method)

        # Static methods should be problematic (they're function objects)
        assert _is_omegaconf_problematic(obj.static_method)

        # But accessing them from the class should behave differently
        assert _is_omegaconf_problematic(TestClass.instance_method)  # unbound method
        assert _is_omegaconf_problematic(TestClass.class_method)  # bound class method
        assert _is_omegaconf_problematic(TestClass.static_method)  # function object


class TestDataclassToOmegaconfDict:
    """Test _dataclass_to_omegaconf_dict function."""

    def test_simple_dataclass(self):
        """Test conversion of simple dataclass."""
        config = SimpleConfig(name="test", value=100)
        result = _dataclass_to_omegaconf_dict(config)

        expected = {"name": "test", "value": 100}
        assert result == expected

    def test_torch_dtype_conversion(self):
        """Test that torch.dtype is converted to string."""
        config = ConfigWithCallable(dtype=torch.float16)
        result = _dataclass_to_omegaconf_dict(config)

        assert result["dtype"] == "torch.float16"
        assert result["name"] == "test"
        # activation_func should be excluded (None not included)
        assert "activation_func" not in result

    def test_callable_exclusion(self):
        """Test that callable fields are excluded."""
        config = ConfigWithCallable(activation_func=torch.nn.functional.gelu)
        result = _dataclass_to_omegaconf_dict(config)

        assert "activation_func" not in result
        assert "name" in result
        assert "dtype" in result

    def test_nested_dataclass(self):
        """Test conversion of nested dataclasses."""
        config = NestedConfig()
        result = _dataclass_to_omegaconf_dict(config)

        assert "simple" in result
        assert "with_callable" in result
        assert result["simple"]["name"] == "test"
        assert result["simple"]["value"] == 42
        assert "activation_func" not in result["with_callable"]

    def test_list_handling(self):
        """Test handling of lists."""
        test_list = [SimpleConfig(name="item1"), SimpleConfig(name="item2")]
        result = _dataclass_to_omegaconf_dict(test_list)

        assert len(result) == 2
        assert result[0]["name"] == "item1"
        assert result[1]["name"] == "item2"

    def test_tuple_handling(self):
        """Test handling of tuples."""
        test_tuple = (SimpleConfig(name="item1"), "string", 42)
        result = _dataclass_to_omegaconf_dict(test_tuple)

        assert len(result) == 3
        assert result[0]["name"] == "item1"
        assert result[1] == "string"
        assert result[2] == 42

    def test_dict_handling(self):
        """Test handling of dictionaries."""
        test_dict = {"config": SimpleConfig(name="test"), "value": 42, "func": dummy_function}  # Should be excluded
        result = _dataclass_to_omegaconf_dict(test_dict)

        assert "config" in result
        assert "value" in result
        assert "func" not in result  # Excluded callable
        assert result["config"]["name"] == "test"

    def test_primitive_types(self):
        """Test handling of primitive types."""
        assert _dataclass_to_omegaconf_dict(42) == 42
        assert _dataclass_to_omegaconf_dict("string") == "string"
        assert _dataclass_to_omegaconf_dict(True) is True
        assert _dataclass_to_omegaconf_dict(None) is None


class TestDataclassToOmegaconfDict_ErrorHandling:
    """Test error handling in _dataclass_to_omegaconf_dict function."""

    def test_error_handling_specific_exceptions(self):
        """Test that specific exceptions are caught gracefully."""

        @dataclasses.dataclass
        class ProblematicConfig:
            """Problematic config for testing."""

            name: str = "test"

            @property
            def problematic_property(self):
                """Problematic property for testing."""
                raise AttributeError("Intentional error")

        config = ProblematicConfig()

        # Should handle AttributeError gracefully and continue processing
        result = _dataclass_to_omegaconf_dict(config)

        # Should still get the valid field
        assert result["name"] == "test"
        # Should not include the problematic field
        assert "problematic_property" not in result

    def test_error_handling_unexpected_exceptions_are_raised(self):
        """Test that unexpected exceptions are not swallowed."""

        @dataclasses.dataclass
        class BadConfig:
            """Bad config for testing."""

            name: str = "test"
            explosive_field: str = "boom"

            def __getattribute__(self, name):
                """Get attribute for testing."""
                if name == "explosive_field":
                    raise ValueError("This should not be caught!")
                return super().__getattribute__(name)

        config = BadConfig()

        # Now when getattr(config, "explosive_field") is called, it will raise ValueError
        # and this should bubble up through _dataclass_to_omegaconf_dict
        with pytest.raises(ValueError, match="This should not be caught!"):
            _dataclass_to_omegaconf_dict(config)


class TestTrackExcludedFields:
    """Test _track_excluded_fields function."""

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
            """Config with dictionary fields for testing."""

            funcs: dict[str, Any] = dataclasses.field(
                default_factory=lambda: {"relu": torch.nn.functional.relu, "value": 42}
            )

        config = ConfigWithDict()
        excluded = _track_excluded_fields(config)

        assert "funcs.relu" in excluded


class TestRestoreExcludedFields:
    """Test _restore_excluded_fields function."""

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


class TestSafeCreateOmegaconfWithPreservation:
    """Test create_omegaconf_dict_config function."""

    def test_preservation_tracking(self):
        """Test that callable preservation tracking works."""
        config = ConfigWithCallable()
        omega_conf, excluded = create_omegaconf_dict_config(config)

        assert isinstance(omega_conf, DictConfig)
        assert len(excluded) > 0
        assert "root.activation_func" in excluded
        assert excluded["root.activation_func"] == torch.nn.functional.relu

    def test_nested_preservation(self):
        """Test preservation with nested configs."""
        config = NestedConfig()
        omega_conf, excluded = create_omegaconf_dict_config(config)

        assert isinstance(omega_conf, DictConfig)
        assert "root.with_callable.activation_func" in excluded


class TestApplyOverrides:
    """Test _apply_overrides function."""

    def test_simple_override(self):
        """Test applying simple overrides."""
        config = SimpleConfig()
        overrides = {"name": "updated", "value": 100}

        _apply_overrides(config, overrides)

        assert config.name == "updated"
        assert config.value == 100

    def test_nested_override(self):
        """Test applying nested overrides."""
        config = NestedConfig()
        overrides = {"simple": {"name": "nested_updated", "value": 200}, "with_callable": {"name": "callable_updated"}}

        _apply_overrides(config, overrides)

        assert config.simple.name == "nested_updated"
        assert config.simple.value == 200
        assert config.with_callable.name == "callable_updated"

    def test_torch_dtype_conversion(self):
        """Test torch.dtype string conversion."""
        config = ConfigWithCallable()
        overrides = {"dtype": "torch.float16"}

        _apply_overrides(config, overrides)

        assert config.dtype == torch.float16

    def test_invalid_key_handling(self):
        """Test handling of invalid override keys."""
        config = SimpleConfig()
        overrides = {"nonexistent": "value"}

        # Should not raise exception, just log warning
        _apply_overrides(config, overrides)

        # Original values should be unchanged
        assert config.name == "test"
        assert config.value == 42

    def test_target_mechanism_instantiation(self):
        """Test that _target_ fields are properly instantiated."""
        config = ConfigWithOptionalProfilingConfig()
        assert config.profiling is None

        # Create override with _target_ using the full module path
        overrides = {
            "profiling": {
                "_target_": "tests.unit_tests.training.utils.test_omegaconf_utils.ProfilingConfig",
                "use_pytorch_profiler": True,
                "profile_ranks": [0, 1, 2],
                "pytorch_profiler_config": {"record_shapes": True},
            }
        }

        _apply_overrides(config, overrides)

        # Verify the object was properly instantiated
        assert config.profiling is not None
        assert isinstance(config.profiling, ProfilingConfig)
        assert config.profiling.use_pytorch_profiler is True
        assert config.profiling.profile_ranks == [0, 1, 2]
        assert config.profiling.pytorch_profiler_config == {"record_shapes": True}

    def test_target_mechanism_with_failed_instantiation(self):
        """Test that _target_ mechanism handles failed instantiation gracefully."""
        config = ConfigWithOptionalProfilingConfig()

        # Create override with invalid _target_
        overrides = {"profiling": {"_target_": "nonexistent.InvalidClass", "use_pytorch_profiler": True}}

        # Should not raise exception, just log warning and fall through
        _apply_overrides(config, overrides)

        # Since instantiation failed, should fall back to setting the dict directly
        assert config.profiling is not None
        assert isinstance(config.profiling, dict)
        assert config.profiling["_target_"] == "nonexistent.InvalidClass"
        assert config.profiling["use_pytorch_profiler"] is True

    def test_target_mechanism_with_existing_dataclass(self):
        """Test that _target_ mechanism works with existing dataclass instances."""
        config = ConfigWithOptionalProfilingConfig()
        # Pre-populate with an existing instance
        config.profiling = ProfilingConfig(use_pytorch_profiler=False, profile_ranks=[])

        # Create override with _target_ for the same field
        overrides = {
            "profiling": {
                "_target_": "tests.unit_tests.training.utils.test_omegaconf_utils.ProfilingConfig",
                "use_pytorch_profiler": True,
                "profile_ranks": [0, 1],
            }
        }

        _apply_overrides(config, overrides)

        # Should replace the existing instance with new one
        assert isinstance(config.profiling, ProfilingConfig)
        assert config.profiling.use_pytorch_profiler is True
        assert config.profiling.profile_ranks == [0, 1]


class TestApplyOverridesWithPreservation:
    """Test apply_overrides function."""

    def test_preservation_workflow(self):
        """Test complete override workflow with preservation."""
        config = ConfigWithCallable()
        original_func = config.activation_func

        # Track excluded fields
        excluded = {"root.activation_func": original_func}

        # Apply overrides
        overrides = {"name": "preserved_test"}
        apply_overrides(config, overrides, excluded)

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

        # 2. Convert to OmegaConf with preservation
        omega_conf, excluded = create_omegaconf_dict_config(config)

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
        apply_overrides(config, final_dict, excluded)

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
        omega_conf, _ = create_omegaconf_dict_config(config)

        # Verify dtype was converted to string
        assert omega_conf.dtype == "torch.float16"
        assert isinstance(omega_conf.dtype, str)

        # Convert back and apply
        config_dict = OmegaConf.to_container(omega_conf, resolve=True)
        _apply_overrides(config, config_dict)

        # Verify dtype was converted back correctly
        assert config.dtype == torch.float16
        assert isinstance(config.dtype, torch.dtype)
        assert config.dtype == original_dtype

    def test_hydra_addition_vs_dataclass_limitation(self):
        """Test that Hydra addition syntax works in OmegaConf but dataclass application is limited."""
        # 1. Create config and convert to OmegaConf
        config = NestedConfig()
        original_func = config.with_callable.activation_func
        omega_conf, excluded = create_omegaconf_dict_config(config)

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

        apply_overrides(config, final_dict, excluded)

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


class TestNoneHandling:
    """Test proper handling of None values vs callable exclusions."""

    def test_legitimate_none_values_preserved(self):
        """Test that legitimate None values are preserved while callables are excluded."""
        config = ConfigWithOptionalFields()

        # Verify initial state
        assert config.optional_str is None
        assert config.optional_int is None
        assert config.activation_func == torch.nn.functional.relu

        # Convert to OmegaConf
        omega_conf, excluded = create_omegaconf_dict_config(config)

        # Verify legitimate None values are preserved in OmegaConf
        assert omega_conf.optional_str is None
        assert omega_conf.optional_int is None
        assert omega_conf.name == "test"
        assert omega_conf.value == 42

        # Verify callable was excluded from OmegaConf
        assert "activation_func" not in omega_conf

        # Verify callable was tracked for restoration
        assert "root.activation_func" in excluded
        assert excluded["root.activation_func"] == torch.nn.functional.relu

    def test_none_values_roundtrip_correctly(self):
        """Test that None values survive the full conversion roundtrip."""
        config = ConfigWithOptionalFields()
        original_func = config.activation_func

        # Convert to OmegaConf with preservation
        omega_conf, excluded = create_omegaconf_dict_config(config)

        # Apply some overrides but leave None values alone
        overrides = {"name": "updated", "value": 100}
        merged_conf = OmegaConf.merge(omega_conf, OmegaConf.create(overrides))

        # Convert back to dict and apply to original config
        final_dict = OmegaConf.to_container(merged_conf, resolve=True)
        apply_overrides(config, final_dict, excluded)

        # Verify None values were preserved
        assert config.optional_str is None
        assert config.optional_int is None

        # Verify overrides were applied
        assert config.name == "updated"
        assert config.value == 100

        # Verify callable was restored
        assert config.activation_func == original_func

    def test_setting_none_explicitly_works(self):
        """Test that explicitly setting fields to None works correctly."""
        config = ConfigWithOptionalFields(name="test", optional_str="initially_set", optional_int=42, value=100)

        # Now set them to None explicitly
        config.optional_str = None
        config.optional_int = None

        # Convert and verify None values are preserved
        omega_conf, excluded = create_omegaconf_dict_config(config)

        assert omega_conf.optional_str is None
        assert omega_conf.optional_int is None
        assert omega_conf.name == "test"
        assert omega_conf.value == 100
