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

import pytest
import torch

from megatron.bridge.models.activations import openai_gelu


class TestOpenAIGELU:
    """Test OpenAI GELU activation function."""

    @pytest.fixture
    def sample_input_1d(self):
        """Create a 1D sample input tensor for testing."""
        return torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=torch.float32)

    def test_openai_gelu_basic_values(self, sample_input_1d):
        """Test openai_gelu with basic known values."""
        result = openai_gelu(sample_input_1d)

        # Check that the result has the same shape as input
        assert result.shape == sample_input_1d.shape

        # Check that the result is a tensor
        assert isinstance(result, torch.Tensor)

        # Check that the dtype is preserved
        assert result.dtype == sample_input_1d.dtype

    def test_openai_gelu_zero(self):
        """Test that openai_gelu(0) = 0."""
        zero_input = torch.tensor(0.0)
        result = openai_gelu(zero_input)
        assert torch.allclose(result, torch.tensor(0.0), atol=1e-6)

    def test_openai_gelu_mathematical_correctness(self):
        """Test the mathematical correctness of openai_gelu implementation."""
        x = torch.tensor(1.0)
        expected = 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
        result = openai_gelu(x)
        assert torch.allclose(result, expected, atol=1e-6)
