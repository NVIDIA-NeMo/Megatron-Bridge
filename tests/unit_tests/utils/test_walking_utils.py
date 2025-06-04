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

"""
Unit tests for walking utilities module.

Tests the module transformation utilities, ensuring they work
correctly for various PyTorch module hierarchies and transformation patterns.
"""

import pytest
import torch
import torch.nn as nn

from nemo_lm.utils import walking_utils as fn


class CustomMLP(nn.Module):
    """Custom MLP for testing module walking functionality."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x):
        """Forward pass of the CustomMLP."""
        return x + self.linear2(self.linear1(x))


class SharedMLP(nn.Module):
    """MLP with shared modules to test transformation tracking."""

    def __init__(self, shared: nn.Module):
        super().__init__()
        self.linear1 = shared
        self.linear2 = shared

    def forward(self, x):
        """Forward pass of the SharedMLP."""
        return x + self.linear2(self.linear1(x))


def add_relu(x):
    """Transform function that adds ReLU after Linear layers."""
    if isinstance(x, nn.Linear):
        return nn.Sequential(x, nn.ReLU())
    return x


def add_relu_named(x, name=None, to_replace="linear1"):
    """Transform function that adds ReLU to specific named modules."""
    if name == to_replace and isinstance(x, nn.Linear):
        return nn.Sequential(x, nn.ReLU())
    return x


def add_relu_first(x, i=None):
    """Transform function that adds ReLU to the first module."""
    if i == 0 and isinstance(x, nn.Linear):
        return nn.Sequential(x, nn.ReLU())
    return x


def add_custom_attribute(module, custom_id=42):
    """Transform function that adds a custom attribute."""
    module.custom_id = custom_id
    return module


def freeze_linear(module):
    """Transform function that freezes Linear layer parameters."""
    if isinstance(module, nn.Linear):
        for param in module.parameters():
            param.requires_grad = False
    return module


class TestMapFunction:
    """Test the map function for module transformations."""

    def test_map_identity(self):
        """Test mapping an identity function."""
        module = nn.Linear(10, 10)
        identity = lambda x: x
        result = fn.map(module, identity)
        assert result is module

    def test_map_transform(self):
        """Test mapping a transform function."""
        module = nn.Linear(10, 10)
        transformed_module = fn.map(module, add_relu)
        assert isinstance(transformed_module, nn.Sequential)
        assert isinstance(transformed_module[0], nn.Linear)
        assert isinstance(transformed_module[1], nn.ReLU)

    def test_map_with_kwargs(self):
        """Test mapping with keyword arguments."""
        module = nn.Linear(10, 10)
        transformed = fn.map(module, add_custom_attribute, custom_id=123)
        assert hasattr(transformed, 'custom_id')
        assert transformed.custom_id == 123

    def test_map_leaf_only(self):
        """Test mapping with leaf_only parameter."""

        def count_parameters(module):
            if list(module.parameters(recurse=False)):
                module.param_count = len(list(module.parameters(recurse=False)))
            return module

        model = CustomMLP()
        # leaf_only=True should apply function only to modules with parameters
        fn.map(model, count_parameters, leaf_only=True)

        # Linear layers should have param_count attribute
        assert hasattr(model.linear1, 'param_count')
        assert hasattr(model.linear2, 'param_count')


class TestWalkFunction:
    """Test the walk function for recursive module transformations."""

    def test_walk_custom_module(self):
        """Test walking through a custom module."""
        mlp = CustomMLP()
        with_relu = fn.walk(mlp, add_relu)
        assert isinstance(with_relu.linear1, nn.Sequential)
        assert isinstance(with_relu.linear2, nn.Sequential)

        # Test named transformation
        with_relu_first = fn.walk(CustomMLP(), add_relu_named)
        assert isinstance(with_relu_first.linear1, nn.Sequential)
        assert isinstance(with_relu_first.linear2, nn.Linear)

        # Test indexed transformation
        with_relu_indexed = fn.walk(CustomMLP(), add_relu_first)
        assert isinstance(with_relu_indexed.linear1, nn.Sequential)
        assert isinstance(with_relu_indexed.linear2, nn.Linear)

    def test_walk_shared_module(self):
        """Test walking through modules with shared references."""

        def double_linear(module: nn.Module):
            if isinstance(module, nn.Linear):
                module.weight.data *= 2
                module.bias.data *= 2
            return module

        shared_linear = nn.Linear(10, 10)
        mlp = SharedMLP(shared_linear)

        # Get initial weight and bias values
        initial_weight = shared_linear.weight.data.clone()
        initial_bias = shared_linear.bias.data.clone()

        # Apply the doubling function using walk
        transformed_mlp = fn.walk(mlp, double_linear)

        # Check that the shared linear module was only transformed once
        assert torch.allclose(transformed_mlp.linear1.weight.data, initial_weight * 2)
        assert torch.allclose(transformed_mlp.linear1.bias.data, initial_bias * 2)
        assert torch.allclose(transformed_mlp.linear2.weight.data, initial_weight * 2)
        assert torch.allclose(transformed_mlp.linear2.bias.data, initial_bias * 2)
        assert transformed_mlp.linear1 is transformed_mlp.linear2

    def test_walk_leaf_only(self):
        """Test walking with leaf_only parameter."""

        def validate_linear(module: nn.Module):
            # Should only be called on Linear modules when leaf_only=True
            assert isinstance(module, nn.Linear)
            return module

        fn.walk(CustomMLP(), validate_linear, leaf_only=True)

    def test_walk_with_transformation_tracking(self):
        """Test that transformation tracking prevents double transformations."""
        transform_count = 0

        def counting_transform(module):
            nonlocal transform_count
            if isinstance(module, nn.Linear):
                transform_count += 1
                module.transformed = True
            return module

        # Create a model with shared linear layer
        shared_linear = nn.Linear(10, 10)
        model = SharedMLP(shared_linear)

        fn.walk(model, counting_transform)

        # Should only transform the shared linear once
        assert transform_count == 1
        assert hasattr(shared_linear, 'transformed')

    def test_walk_parameter_freezing(self):
        """Test walking to freeze parameters."""
        model = CustomMLP()

        # Initially all parameters should be trainable
        assert all(p.requires_grad for p in model.parameters())

        # Freeze linear layers
        frozen_model = fn.walk(model, freeze_linear)

        # All parameters should now be frozen
        assert all(not p.requires_grad for p in frozen_model.parameters())


class TestWalkListModule:
    """Test walking through list-based module containers."""

    @pytest.mark.parametrize("module_container", [nn.ModuleList, nn.Sequential])
    def test_walk_module_container(self, module_container):
        """Test walking through ModuleList and Sequential containers."""
        modules = [nn.Linear(10, 10), nn.Linear(10, 10)]
        module = module_container(modules) if module_container is nn.ModuleList else nn.Sequential(*modules)

        def fill_weights(module):
            """Fill the weights of the module with 1.0."""
            if isinstance(module, nn.Linear):
                module.weight.data.fill_(1.0)
            return module

        walked_module = fn.walk(module, fill_weights)

        assert isinstance(walked_module, module_container)
        assert len(walked_module) == 2
        assert torch.allclose(walked_module[0].weight, torch.ones_like(walked_module[0].weight))
        assert torch.allclose(walked_module[1].weight, torch.ones_like(walked_module[1].weight))

    @pytest.mark.parametrize("module_container", [nn.ModuleList, nn.Sequential])
    def test_walk_module_container_with_kwargs(self, module_container):
        """Test walking with keyword arguments through containers."""
        modules = [nn.Linear(10, 10), nn.Linear(10, 10)]
        module = module_container(modules) if module_container is nn.ModuleList else nn.Sequential(*modules)

        def fill_weights_value(module, value):
            """Fill the weights of the module with the given value."""
            if isinstance(module, nn.Linear):
                module.weight.data.fill_(value)
            return module

        walked_module = fn.walk(module, fill_weights_value, value=2.0)

        assert isinstance(walked_module, module_container)
        assert len(walked_module) == 2
        assert torch.allclose(walked_module[0].weight, 2.0 * torch.ones_like(walked_module[0].weight))
        assert torch.allclose(walked_module[1].weight, 2.0 * torch.ones_like(walked_module[1].weight))

    @pytest.mark.parametrize("module_container", [nn.ModuleList, nn.Sequential])
    def test_walk_module_container_with_recursion(self, module_container):
        """Test walking through nested container structures."""
        modules = [
            nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10)),
            nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10)),
        ]
        module = module_container(modules) if module_container is nn.ModuleList else nn.Sequential(*modules)

        def fill_weights(module):
            """Fill the weights of the linear module with 1.0."""
            if isinstance(module, nn.Linear):
                module.weight.data.fill_(1.0)
            return module

        walked_module = fn.walk(module, fill_weights)

        assert isinstance(walked_module, module_container)
        assert len(walked_module) == 2
        for seq in walked_module:
            assert isinstance(seq, nn.Sequential)
            assert len(seq) == 2
            assert torch.allclose(seq[0].weight, torch.ones_like(seq[0].weight))
            assert torch.allclose(seq[1].weight, torch.ones_like(seq[1].weight))


class TestWalkDictModule:
    """Test walking through dictionary-based module containers."""

    def test_walk_module_dict_identity(self):
        """Test walking through an nn.ModuleDict without transformations."""
        modules = nn.ModuleDict({"linear": nn.Linear(10, 10), "conv": nn.Conv2d(1, 20, 5)})
        identity = lambda x: x

        walked_modules = fn.walk(modules, identity)

        assert isinstance(walked_modules, nn.ModuleDict)
        assert "linear" in walked_modules and isinstance(walked_modules["linear"], nn.Linear)
        assert "conv" in walked_modules and isinstance(walked_modules["conv"], nn.Conv2d)

    def test_walk_module_dict_transform(self):
        """Test walking through an nn.ModuleDict with transformations."""
        modules = nn.ModuleDict({"linear": nn.Linear(10, 10), "conv": nn.Conv2d(1, 20, 5)})

        def add_relu_to_named(module: nn.Module, name=None):
            """Add ReLU to the named module."""
            if name in ["linear", "conv"]:
                return nn.Sequential(module, nn.ReLU())
            return module

        walked_modules = fn.walk(modules, add_relu_to_named)

        assert isinstance(walked_modules, nn.ModuleDict)
        for module in walked_modules.values():
            assert isinstance(module, nn.Sequential)
            assert isinstance(module[1], nn.ReLU)

    def test_walk_module_dict_with_kwargs(self):
        """Test walking ModuleDict with keyword arguments."""
        modules = nn.ModuleDict({"linear1": nn.Linear(10, 10), "linear2": nn.Linear(5, 5)})

        def add_tag(module, tag="test"):
            """Add a tag to the module."""
            module.tag = tag
            return module

        walked_modules = fn.walk(modules, add_tag, tag="custom")

        assert isinstance(walked_modules, nn.ModuleDict)
        for module in walked_modules.values():
            assert hasattr(module, 'tag')
            assert module.tag == "custom"


class TestForallFunction:
    """Test the forall predicate function."""

    def test_forall_all_linear(self):
        """Test forall with a predicate that should return True."""
        modules = nn.ModuleList([nn.Linear(10, 10), nn.Linear(5, 5)])

        def is_linear(module):
            """Check if the module is a linear module or a module list."""
            return isinstance(module, (nn.Linear, nn.ModuleList))

        assert fn.forall(modules, is_linear, recurse=True)

    def test_forall_mixed_modules(self):
        """Test forall with a predicate that should return False."""
        modules = nn.ModuleList([nn.Linear(10, 10), nn.ReLU()])

        def is_linear(module):
            """Check if the module is a linear module."""
            return isinstance(module, nn.Linear)

        assert not fn.forall(modules, is_linear, recurse=True)

    def test_forall_non_recursive(self):
        """Test forall without recursion."""
        model = CustomMLP()

        def is_custom_mlp(module):
            """Check if the module is a CustomMLP module."""
            return isinstance(module, CustomMLP)

        # Should return True for the top-level module only
        assert fn.forall(model, is_custom_mlp, recurse=False)

        # Should return False when recursing (contains Linear modules)
        assert not fn.forall(model, is_custom_mlp, recurse=True)

    def test_forall_with_bool_protocol(self):
        """Test forall with objects that implement bool protocol."""

        class BoolResult:
            """Helper class for testing bool protocol."""

            def __init__(self, value):
                self.value = value

            def __bool__(self):
                return self.value

        def bool_predicate(module):
            """Check if the module is a linear module."""
            return BoolResult(isinstance(module, nn.Linear))

        linear = nn.Linear(10, 10)
        assert fn.forall(linear, bool_predicate)

        relu = nn.ReLU()
        assert not fn.forall(relu, bool_predicate)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_module_list(self):
        """Test walking through empty containers."""
        empty_list = nn.ModuleList([])
        result = fn.walk(empty_list, lambda x: x)
        assert isinstance(result, nn.ModuleList)
        assert len(result) == 0

    def test_empty_module_dict(self):
        """Test walking through empty ModuleDict."""
        empty_dict = nn.ModuleDict({})
        result = fn.walk(empty_dict, lambda x: x)
        assert isinstance(result, nn.ModuleDict)
        assert len(result) == 0

    def test_function_signature_matching(self):
        """Test that function signature matching works correctly."""

        def transform_with_extra_kwargs(module, used_kwarg, unused_kwarg=None):
            """Transform the module with the given keyword arguments."""
            module.used_kwarg = used_kwarg
            return module

        model = nn.Linear(10, 10)
        result = fn.walk(model, transform_with_extra_kwargs, used_kwarg="test", extra_kwarg="ignored")

        assert hasattr(result, 'used_kwarg')
        assert result.used_kwarg == "test"

    def test_module_with_builtin_map(self):
        """Test handling modules that have their own map method."""

        class ModuleWithMap(nn.Module):
            """Module with custom map method for testing."""

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
                self.map_called = False

            def map(self, func, **kwargs):
                """Map the function to the module."""
                self.map_called = True
                return self

        module = ModuleWithMap()
        result = fn.map(module, lambda x: x)

        assert result.map_called
        assert result is module

    def test_skip_map_flag(self):
        """Test the _skip_map flag functionality."""

        class ModuleWithMap(nn.Module):
            """Module with custom map method for testing skip functionality."""

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
                self.map_called = False

            def map(self, func, **kwargs):
                """Map the function to the module."""
                self.map_called = True
                return self

        module = ModuleWithMap()

        # With _skip_map=False (default), should use module's map
        result1 = fn.map(module, lambda x: x)
        assert result1.map_called

        # With _skip_map=True, should skip module's map
        module.map_called = False
        result2 = fn.map(module, lambda x: x, _skip_map=True)
        assert not result2.map_called


class TestParameterCounting:
    """Test parameter counting functionality."""

    def test_parameter_counting_accuracy(self):
        """Test that parameter counting works correctly."""
        model = CustomMLP()

        def count_params(module):
            """Count the parameters of the module."""
            if hasattr(module, 'weight'):
                if not hasattr(module, 'param_count'):
                    module.param_count = 0
                module.param_count += module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    module.param_count += module.bias.numel()
            return module

        walked_model = fn.walk(model, count_params)

        # Check that linear layers got param counts
        assert hasattr(walked_model.linear1, 'param_count')
        assert hasattr(walked_model.linear2, 'param_count')

        # Verify counts are reasonable
        assert walked_model.linear1.param_count > 0
        assert walked_model.linear2.param_count > 0
