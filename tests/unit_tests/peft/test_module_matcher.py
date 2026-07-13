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

from collections import defaultdict
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from megatron.bridge.peft.module_matcher import ModuleMatcher


class MockColumnParallelLinear(nn.Module):
    """Mock ColumnParallelLinear for testing."""

    def __init__(self, in_features: int = 128, out_features: int = 256):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return x


class MockRowParallelLinear(nn.Module):
    """Mock RowParallelLinear for testing."""

    def __init__(self, in_features: int = 128, out_features: int = 256):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return x


class MockTELinear(nn.Module):
    """Mock Transformer Engine Linear for testing."""

    def __init__(self, in_features: int = 128, out_features: int = 256):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return x


class TestModuleMatcher:
    """Test suite for ModuleMatcher class."""

    def test_default_initialization(self):
        """Test ModuleMatcher with default parameters."""
        matcher = ModuleMatcher()
        assert matcher.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
        assert matcher.exclude_modules == []
        assert len(matcher.canonical_mapping) == 0

    def test_custom_target_modules(self):
        """Test ModuleMatcher with custom target modules."""
        target_modules = ["linear_qkv", "linear_proj"]
        matcher = ModuleMatcher(target_modules=target_modules)
        assert matcher.target_modules == target_modules

    def test_custom_exclude_modules(self):
        """Test ModuleMatcher with custom exclude modules."""
        exclude_modules = ["linear_fc1", "linear_fc2"]
        matcher = ModuleMatcher(exclude_modules=exclude_modules)
        assert matcher.exclude_modules == exclude_modules

    def test_custom_canonical_mapping(self):
        """Test ModuleMatcher with custom canonical mapping."""
        canonical_mapping = defaultdict(set)
        canonical_mapping["test_pattern"] = {"TestModule"}
        matcher = ModuleMatcher(canonical_mapping=canonical_mapping)
        assert "test_pattern" in matcher.canonical_mapping

    @pytest.mark.parametrize(
        "module_type,expected_match",
        [
            (nn.Linear, True),
            (nn.Conv2d, True),  # Should match because name matches, regardless of module type
            (nn.BatchNorm1d, True),  # Should match because name matches, regardless of module type
        ],
    )
    def test_target_modules_exact_match(self, module_type, expected_match):
        """Test exact matching with target modules."""
        matcher = ModuleMatcher(target_modules=["linear_qkv"])

        # Create modules with proper parameters for each type
        if module_type == nn.Linear:
            module = module_type(128, 256)
        elif module_type == nn.Conv2d:
            module = module_type(in_channels=3, out_channels=64, kernel_size=3)
        elif module_type == nn.BatchNorm1d:
            module = module_type(128)
        else:
            module = module_type(128)

        result = matcher.match(module, name="linear_qkv")
        if expected_match:
            assert result is not None
            pattern, full_name = result
            assert pattern == "linear_qkv"
            assert full_name == "linear_qkv"
        else:
            assert result is None

    def test_target_modules_no_match(self):
        """Test no match with target modules."""
        matcher = ModuleMatcher(target_modules=["linear_qkv"])
        module = nn.Linear(128, 256)

        result = matcher.match(module, name="linear_proj")
        assert result is None

    def test_target_vs_exclude_mode_difference(self):
        """Test the difference between target_modules (name-based) and exclude mode (type-based)."""
        # In target_modules mode, matching is based on name patterns only
        target_matcher = ModuleMatcher(target_modules=["linear_qkv"])
        conv_module = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

        # Should match because name matches, regardless of module type
        result = target_matcher.match(conv_module, name="linear_qkv")
        assert result is not None

        # In exclude mode, matching is based on both name and module type
        with (
            patch("megatron.bridge.peft.module_matcher.ColumnParallelLinear", MockColumnParallelLinear),
            patch("megatron.bridge.peft.module_matcher.RowParallelLinear", MockRowParallelLinear),
        ):
            exclude_matcher = ModuleMatcher(target_modules=[], exclude_modules=[])

            # Should NOT match because it's not a linear layer type
            result = exclude_matcher.match(conv_module, name="linear_qkv")
            assert result is None

            # Should match because it IS a linear layer type
            linear_module = nn.Linear(128, 256)
            result = exclude_matcher.match(linear_module, name="linear_qkv")
            assert result is not None

    def test_target_modules_wildcard_match(self):
        """Test wildcard matching with target modules."""
        matcher = ModuleMatcher(target_modules=["*.layers.0.*.linear_qkv"])
        module = nn.Linear(128, 256)

        result = matcher.match(module, name="linear_qkv", prefix="model.layers.0.self_attention")
        assert result is not None
        pattern, full_name = result
        assert pattern == "*.layers.0.*.linear_qkv"
        assert full_name == "model.layers.0.self_attention.linear_qkv"

    def test_target_modules_wildcard_no_match(self):
        """Test wildcard pattern that doesn't match."""
        matcher = ModuleMatcher(target_modules=["*.layers.0.*.linear_qkv"])
        module = nn.Linear(128, 256)

        result = matcher.match(module, name="linear_qkv", prefix="model.layers.1.self_attention")
        assert result is None

    def test_canonical_mapping_exact_match(self):
        """Test exact matching with canonical mapping."""
        canonical_mapping = defaultdict(set)
        canonical_mapping["linear_qkv"] = {nn.Linear}
        matcher = ModuleMatcher(canonical_mapping=canonical_mapping)
        module = nn.Linear(128, 256)

        result = matcher.match(module, name="linear_qkv")
        assert result is not None
        pattern, full_name = result
        assert pattern == "linear_qkv"
        assert full_name == "linear_qkv"

    def test_canonical_mapping_wildcard_match(self):
        """Test wildcard matching with canonical mapping."""
        canonical_mapping = defaultdict(set)
        canonical_mapping["*.linear_qkv"] = {nn.Linear}
        matcher = ModuleMatcher(canonical_mapping=canonical_mapping)
        module = nn.Linear(128, 256)

        result = matcher.match(module, name="linear_qkv", prefix="model.layers.0.self_attention")
        assert result is not None
        pattern, full_name = result
        assert pattern == "*.linear_qkv"
        assert full_name == "model.layers.0.self_attention.linear_qkv"

    @patch("megatron.bridge.peft.module_matcher.ColumnParallelLinear", MockColumnParallelLinear)
    @patch("megatron.bridge.peft.module_matcher.RowParallelLinear", MockRowParallelLinear)
    def test_exclude_modules_mode_match(self):
        """Test exclude modules mode with matching linear layer."""
        matcher = ModuleMatcher(target_modules=[], exclude_modules=["linear_fc1"])
        module = nn.Linear(128, 256)

        result = matcher.match(module, name="linear_qkv")
        assert result is not None
        pattern, full_name = result
        assert pattern == "linear_qkv"
        assert full_name == "linear_qkv"

    @patch("megatron.bridge.peft.module_matcher.ColumnParallelLinear", MockColumnParallelLinear)
    @patch("megatron.bridge.peft.module_matcher.RowParallelLinear", MockRowParallelLinear)
    def test_exclude_modules_mode_exclude(self):
        """Test exclude modules mode with excluded module."""
        matcher = ModuleMatcher(target_modules=[], exclude_modules=["linear_fc1"])
        module = nn.Linear(128, 256)

        result = matcher.match(module, name="linear_fc1")
        assert result is None

    @patch("megatron.bridge.peft.module_matcher.ColumnParallelLinear", MockColumnParallelLinear)
    @patch("megatron.bridge.peft.module_matcher.RowParallelLinear", MockRowParallelLinear)
    def test_exclude_modules_wildcard_exclude(self):
        """Test exclude modules mode with wildcard exclusion."""
        matcher = ModuleMatcher(target_modules=[], exclude_modules=["*.linear_fc*"])
        module = nn.Linear(128, 256)

        result = matcher.match(module, name="linear_fc1", prefix="model.layers.0")
        assert result is None

    @patch("megatron.bridge.peft.module_matcher.ColumnParallelLinear", MockColumnParallelLinear)
    @patch("megatron.bridge.peft.module_matcher.RowParallelLinear", MockRowParallelLinear)
    def test_exclude_modules_non_linear_no_match(self):
        """Test exclude modules mode with non-linear module."""
        matcher = ModuleMatcher(target_modules=[], exclude_modules=[])
        module = nn.Conv2d(3, 64, 3)

        result = matcher.match(module, name="conv1")
        assert result is None

    @patch("megatron.bridge.peft.module_matcher.ColumnParallelLinear", MockColumnParallelLinear)
    @patch("megatron.bridge.peft.module_matcher.RowParallelLinear", MockRowParallelLinear)
    def test_parallel_linear_types(self):
        """Test matching with parallel linear types."""
        matcher = ModuleMatcher(target_modules=[], exclude_modules=[])

        # Test ColumnParallelLinear
        col_module = MockColumnParallelLinear()
        result = matcher.match(col_module, name="linear_qkv")
        assert result is not None

        # Test RowParallelLinear
        row_module = MockRowParallelLinear()
        result = matcher.match(row_module, name="linear_proj")
        assert result is not None

    @patch("megatron.bridge.peft.module_matcher.HAVE_TE_COL_LINEAR", True)
    @patch("megatron.bridge.peft.module_matcher.TEColumnParallelLinear", MockTELinear)
    def test_transformer_engine_support(self):
        """Test matching with Transformer Engine modules when available."""
        matcher = ModuleMatcher(target_modules=[], exclude_modules=[])
        te_module = MockTELinear()

        result = matcher.match(te_module, name="linear_qkv")
        assert result is not None

    def test_full_name_construction(self):
        """Test proper construction of full names with prefixes."""
        matcher = ModuleMatcher(target_modules=["linear_qkv"])
        module = nn.Linear(128, 256)

        # Test with prefix
        result = matcher.match(module, name="linear_qkv", prefix="model.layers.0.self_attention")
        assert result is not None
        pattern, full_name = result
        assert full_name == "model.layers.0.self_attention.linear_qkv"

        # Test without prefix
        result = matcher.match(module, name="linear_qkv")
        assert result is not None
        pattern, full_name = result
        assert full_name == "linear_qkv"

    def test_assertion_exclude_modules_with_canonical_mapping(self):
        """Test assertion when exclude_modules is used with canonical_mapping."""
        canonical_mapping = defaultdict(set)
        canonical_mapping["linear_qkv"] = {nn.Linear}
        matcher = ModuleMatcher(canonical_mapping=canonical_mapping, exclude_modules=["linear_fc1"])
        module = nn.Linear(128, 256)

        with pytest.raises(AssertionError, match="exclude_modules should be empty when using canonical_mapping"):
            matcher.match(module, name="linear_qkv")

    def test_assertion_exclude_modules_with_target_modules(self):
        """Test assertion when exclude_modules is used with target_modules."""
        matcher = ModuleMatcher(target_modules=["linear_qkv"], exclude_modules=["linear_fc1"])
        module = nn.Linear(128, 256)

        with pytest.raises(AssertionError, match="exclude_modules should be empty when using target_modules"):
            matcher.match(module, name="linear_qkv")

    def test_none_name_handling(self):
        """Test handling of None name parameter."""
        matcher = ModuleMatcher(target_modules=["linear_qkv"])
        module = nn.Linear(128, 256)

        result = matcher.match(module, name=None)
        assert result is None

    def test_none_prefix_handling(self):
        """Test handling of None prefix parameter."""
        matcher = ModuleMatcher(target_modules=["linear_qkv"])
        module = nn.Linear(128, 256)

        result = matcher.match(module, name="linear_qkv", prefix=None)
        assert result is not None
        pattern, full_name = result
        assert full_name == "linear_qkv"

    def test_empty_target_modules_with_exclude(self):
        """Test behavior with empty target_modules and exclude_modules."""
        matcher = ModuleMatcher(target_modules=[], exclude_modules=[])
        module = nn.Linear(128, 256)

        # Should fall back to exclude mode
        with (
            patch("megatron.bridge.peft.module_matcher.ColumnParallelLinear", MockColumnParallelLinear),
            patch("megatron.bridge.peft.module_matcher.RowParallelLinear", MockRowParallelLinear),
        ):
            result = matcher.match(module, name="linear_qkv")
            assert result is not None

    def test_complex_wildcard_patterns(self):
        """Test complex wildcard patterns."""
        matcher = ModuleMatcher(target_modules=["model.layers.*.self_attention.linear_*"])
        module = nn.Linear(128, 256)

        # Should match
        result = matcher.match(module, name="linear_qkv", prefix="model.layers.5.self_attention")
        assert result is not None

        # Should not match - wrong layer structure
        result = matcher.match(module, name="linear_qkv", prefix="model.decoder.layers.5.self_attention")
        assert result is None

    def test_multiple_target_patterns(self):
        """Test matching with multiple target patterns."""
        matcher = ModuleMatcher(target_modules=["linear_qkv", "*.linear_proj", "model.*.linear_fc1"])
        module = nn.Linear(128, 256)

        # Test exact match
        result = matcher.match(module, name="linear_qkv")
        assert result is not None
        assert result[0] == "linear_qkv"

        # Test wildcard match 1
        result = matcher.match(module, name="linear_proj", prefix="model.layers.0")
        assert result is not None
        assert result[0] == "*.linear_proj"

        # Test wildcard match 2
        result = matcher.match(module, name="linear_fc1", prefix="model.layers.0")
        assert result is not None
        assert result[0] == "model.*.linear_fc1"

    def test_edge_case_empty_strings(self):
        """Test edge cases with empty strings."""
        matcher = ModuleMatcher(target_modules=[""])
        module = nn.Linear(128, 256)

        result = matcher.match(module, name="", prefix="")
        assert result is not None
        pattern, full_name = result
        assert pattern == ""
        assert full_name == ""

    @pytest.mark.parametrize(
        "target_modules,canonical_mapping,exclude_modules",
        [
            (["linear_qkv"], None, []),
            ([], {"pattern": {nn.Linear}}, []),
            ([], None, ["linear_fc1"]),
        ],
    )
    def test_different_mode_configurations(self, target_modules, canonical_mapping, exclude_modules):
        """Test different valid configurations of the matcher."""
        if canonical_mapping is None:
            canonical_mapping = defaultdict(set)

        matcher = ModuleMatcher(
            target_modules=target_modules, canonical_mapping=canonical_mapping, exclude_modules=exclude_modules
        )

        # Should not raise any exceptions during initialization
        assert isinstance(matcher, ModuleMatcher)
