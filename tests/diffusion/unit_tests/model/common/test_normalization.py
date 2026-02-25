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

from megatron.bridge.diffusion.models.common.normalization import RMSNorm


def test_rmsnorm_initialization():
    """Test RMSNorm initialization with different hidden sizes."""
    hidden_size = 768
    norm = RMSNorm(hidden_size)

    # Check weight parameter exists and has correct shape
    assert hasattr(norm, "weight")
    assert norm.weight.shape == (hidden_size,)
    assert isinstance(norm.weight, torch.nn.Parameter)

    # Check weight is initialized to ones
    assert torch.allclose(norm.weight, torch.ones(hidden_size))

    # Check epsilon value
    assert norm.eps == 1e-6


def test_rmsnorm_initialization_with_custom_eps():
    """Test RMSNorm initialization with custom epsilon."""
    hidden_size = 512
    custom_eps = 1e-8
    norm = RMSNorm(hidden_size, eps=custom_eps)

    assert norm.eps == custom_eps


def test_rmsnorm_initialization_with_config():
    """Test RMSNorm initialization with config parameter (for compatibility)."""
    hidden_size = 1024
    mock_config = {"dummy": "config"}

    # Should not raise an error even with config parameter
    norm = RMSNorm(hidden_size, config=mock_config)
    assert norm.weight.shape == (hidden_size,)


def test_rmsnorm_forward_2d_input():
    """Test RMSNorm forward pass with 2D input [batch, hidden]."""
    batch_size = 4
    hidden_size = 256

    norm = RMSNorm(hidden_size)
    x = torch.randn(batch_size, hidden_size)

    output = norm(x)

    # Check output shape
    assert output.shape == (batch_size, hidden_size)

    # Check output dtype matches input
    assert output.dtype == x.dtype


def test_rmsnorm_forward_3d_input():
    """Test RMSNorm forward pass with 3D input [batch, seq_len, hidden]."""
    batch_size = 2
    seq_len = 128
    hidden_size = 512

    norm = RMSNorm(hidden_size)
    x = torch.randn(batch_size, seq_len, hidden_size)

    output = norm(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, hidden_size)

    # Check output dtype matches input
    assert output.dtype == x.dtype


def test_rmsnorm_numerical_correctness():
    """Test that RMSNorm produces numerically correct results."""
    hidden_size = 64
    norm = RMSNorm(hidden_size, eps=1e-6)

    # Create a simple input
    x = torch.randn(2, hidden_size)

    # Manually compute expected RMS normalization
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + norm.eps)
    expected = x / rms

    # Get actual output (without weight scaling for this test)
    with torch.no_grad():
        norm.weight.fill_(1.0)  # Set weights to 1 to isolate normalization

    output = norm(x)

    # Compare (allow small numerical differences)
    assert torch.allclose(output, expected, rtol=1e-4, atol=1e-6)


def test_rmsnorm_weight_scaling():
    """Test that RMSNorm correctly applies weight scaling."""
    hidden_size = 32
    norm = RMSNorm(hidden_size)

    # Set custom weights
    with torch.no_grad():
        norm.weight.fill_(2.0)

    x = torch.randn(4, hidden_size)

    # Get normalized output
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + norm.eps)
    normalized = x / rms
    expected = normalized * 2.0

    output = norm(x)

    assert torch.allclose(output, expected, rtol=1e-4, atol=1e-6)


def test_rmsnorm_different_dtypes():
    """Test RMSNorm with different input dtypes."""
    hidden_size = 128
    norm = RMSNorm(hidden_size)

    # Test with float32
    x_float32 = torch.randn(2, hidden_size, dtype=torch.float32)
    output_float32 = norm(x_float32)
    assert output_float32.dtype == torch.float32

    # Test with float16
    x_float16 = torch.randn(2, hidden_size, dtype=torch.float16)
    output_float16 = norm(x_float16)
    assert output_float16.dtype == torch.float16

    # Test with bfloat16
    x_bfloat16 = torch.randn(2, hidden_size, dtype=torch.bfloat16)
    output_bfloat16 = norm(x_bfloat16)
    assert output_bfloat16.dtype == torch.bfloat16


def test_rmsnorm_preserves_dtype():
    """Test that RMSNorm preserves input dtype even though internal computation is in float32."""
    hidden_size = 256
    norm = RMSNorm(hidden_size)

    # Test with bfloat16 (common in training)
    x = torch.randn(3, 10, hidden_size, dtype=torch.bfloat16)
    output = norm(x)

    # Output should have same dtype as input
    assert output.dtype == torch.bfloat16

    # But should have been normalized correctly (internal computation in float)
    # Verify by checking the RMS is approximately 1
    with torch.no_grad():
        norm.weight.fill_(1.0)
        output_normalized = norm(x.float())
        rms = torch.sqrt(torch.mean(output_normalized**2, dim=-1))
        # RMS should be close to 1 after normalization
        assert torch.allclose(rms, torch.ones_like(rms), rtol=0.1)


def test_rmsnorm_zero_input():
    """Test RMSNorm behavior with zero input (edge case)."""
    hidden_size = 64
    norm = RMSNorm(hidden_size, eps=1e-6)

    # Create zero input
    x = torch.zeros(2, hidden_size)

    # Should not crash and should produce zero output (scaled by weights)
    output = norm(x)

    # With zero input and epsilon, the norm is sqrt(0 + eps)
    # So output should be 0 / sqrt(eps) * weight = 0
    assert torch.allclose(output, torch.zeros_like(output))


def test_rmsnorm_gradient_flow():
    """Test that gradients flow properly through RMSNorm."""
    hidden_size = 128
    norm = RMSNorm(hidden_size)

    x = torch.randn(4, hidden_size, requires_grad=True)
    output = norm(x)

    # Compute loss and backprop
    loss = output.sum()
    loss.backward()

    # Check that gradients exist for both input and weight
    assert x.grad is not None
    assert norm.weight.grad is not None

    # Check that gradients are not all zeros
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    assert not torch.allclose(norm.weight.grad, torch.zeros_like(norm.weight.grad))


def test_rmsnorm_batch_independence():
    """Test that normalization is independent across batch dimension."""
    hidden_size = 64
    norm = RMSNorm(hidden_size)

    # Create batched input
    x1 = torch.randn(1, hidden_size)
    x2 = torch.randn(1, hidden_size)
    x_batched = torch.cat([x1, x2], dim=0)

    # Process individually and batched
    with torch.no_grad():
        out1 = norm(x1)
        out2 = norm(x2)
        out_batched = norm(x_batched)

    # Results should be identical
    assert torch.allclose(out_batched[0], out1[0], rtol=1e-5)
    assert torch.allclose(out_batched[1], out2[0], rtol=1e-5)


def test_rmsnorm_sequence_independence():
    """Test that normalization is independent across sequence dimension."""
    hidden_size = 64
    seq_len = 10
    norm = RMSNorm(hidden_size)

    # Create 3D input [batch, seq, hidden]
    x = torch.randn(2, seq_len, hidden_size)

    with torch.no_grad():
        # Process full sequence
        output_full = norm(x)

        # Process each position separately
        for i in range(seq_len):
            output_single = norm(x[:, i : i + 1, :])
            assert torch.allclose(output_full[:, i : i + 1, :], output_single, rtol=1e-5)


def test_rmsnorm_epsilon_effect():
    """Test that epsilon parameter affects numerical stability."""
    hidden_size = 64

    # Create very small input values
    x = torch.randn(2, hidden_size) * 1e-8

    # Test with different epsilon values
    norm_large_eps = RMSNorm(hidden_size, eps=1e-3)
    norm_small_eps = RMSNorm(hidden_size, eps=1e-10)

    with torch.no_grad():
        norm_large_eps.weight.fill_(1.0)
        norm_small_eps.weight.fill_(1.0)

        output_large_eps = norm_large_eps(x)
        output_small_eps = norm_small_eps(x)

    # Outputs should be different due to epsilon
    assert not torch.allclose(output_large_eps, output_small_eps, rtol=1e-2)


def test_rmsnorm_large_values():
    """Test RMSNorm with large input values."""
    hidden_size = 128
    norm = RMSNorm(hidden_size)

    # Create input with large values
    x = torch.randn(2, hidden_size) * 1000

    # Should not produce NaN or Inf
    output = norm(x)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_rmsnorm_different_hidden_sizes():
    """Test RMSNorm with various hidden sizes."""
    hidden_sizes = [64, 128, 256, 512, 768, 1024, 2048, 4096]

    for hidden_size in hidden_sizes:
        norm = RMSNorm(hidden_size)
        x = torch.randn(2, 10, hidden_size)
        output = norm(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
