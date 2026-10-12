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


import torch
import torch.nn as nn

from megatron.bridge.models.gemma.modules import extend_instance


class TestExtendInstance:
    """Test suite for the extend_instance function."""

    def test_extend_instance_basic_functionality(self):
        """Test basic functionality of extend_instance."""

        # Create a simple base class
        class BaseClass:
            def method(self):
                return "base"

        # Create a mixin class
        class Mixin:
            def method(self):
                return f"mixin -> {super().method()}"

            def new_method(self):
                return "new_method"

        # Create an instance and extend it
        obj = BaseClass()
        original_class = obj.__class__
        extend_instance(obj, Mixin)

        # Test that the class has changed
        assert obj.__class__ != original_class
        assert obj.__class__.__name__ == "BaseClass"

        # Test that the mixin method is called first
        assert obj.method() == "mixin -> base"

        # Test that new methods are available
        assert obj.new_method() == "new_method"

    def test_extend_instance_preserves_attributes(self):
        """Test that extend_instance preserves object attributes."""

        class BaseClass:
            def __init__(self, value):
                self.value = value

        class Mixin:
            def get_doubled_value(self):
                return self.value * 2

        # Create an instance with attributes
        obj = BaseClass(42)
        extend_instance(obj, Mixin)

        # Test that attributes are preserved
        assert obj.value == 42
        assert obj.get_doubled_value() == 84

    def test_extend_instance_method_resolution_order(self):
        """Test that extend_instance correctly sets the method resolution order."""

        class BaseClass:
            def identify(self):
                return "base"

        class Mixin:
            def identify(self):
                return "mixin"

        obj = BaseClass()
        extend_instance(obj, Mixin)

        # Mixin should be first in MRO, so its method should be called
        assert obj.identify() == "mixin"

        # Check MRO
        mro = obj.__class__.__mro__
        assert len(mro) >= 3  # NewClass, Mixin, BaseClass, object
        assert mro[1] == Mixin

    def test_extend_instance_multiple_extensions(self):
        """Test applying multiple mixins in sequence."""

        class BaseClass:
            def value(self):
                return 1

        class FirstMixin:
            def value(self):
                return super().value() + 10

        class SecondMixin:
            def value(self):
                return super().value() + 100

        obj = BaseClass()
        extend_instance(obj, FirstMixin)
        extend_instance(obj, SecondMixin)

        # Should be 1 + 10 + 100 = 111
        assert obj.value() == 111

    def test_extend_instance_with_torch_module(self):
        """Test extend_instance with PyTorch modules."""

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        class ModuleMixin:
            def forward(self, x):
                result = super().forward(x)
                return result * 2  # Scale output by 2

        module = SimpleModule()
        x = torch.randn(3, 10)

        # Get original output
        original_output = module(x)

        # Extend the module
        extend_instance(module, ModuleMixin)

        # Get new output
        new_output = module(x)

        # Should be doubled
        assert torch.allclose(new_output, original_output * 2)
