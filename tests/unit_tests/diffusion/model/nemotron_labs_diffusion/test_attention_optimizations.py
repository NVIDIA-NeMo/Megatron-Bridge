"""Numerical coverage for grouped-query attention."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from megatron.bridge.diffusion.models.common import nemotron_labs_diffusion_attention as attention


pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_gqa_output_and_gradients_match_repeated_kv(native: bool, kv_heads: int) -> None:
    if native and not attention._FLEX_SUPPORTS_GQA:
        pytest.skip("This PyTorch version has no native FlexAttention GQA")
    torch.manual_seed(42)
    inputs = [torch.randn(2, heads, 16, 8, requires_grad=True) for heads in (4, kv_heads, kv_heads)]
    reference = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    q, k, v = reference
    expected = F.scaled_dot_product_attention(
        q, attention.repeat_kv(k, 4 // kv_heads), attention.repeat_kv(v, 4 // kv_heads)
    )
    with (
        patch.object(attention, "_FLEX_SUPPORTS_GQA", native),
        patch.object(attention, "flex_attention", wraps=flex_attention) as kernel,
    ):
        # Exercise the production wrapper body; GPU Inductor compilation is a
        # separate check from CPU numerical and API-compatibility parity.
        actual = attention.fused_flex_attention.__wrapped__(*inputs)
    call = kernel.call_args
    assert call.args[1].shape[1] == (kv_heads if native else 4)
    if native:
        assert call.kwargs["enable_gqa"] == (kv_heads != 4)
    else:
        assert "enable_gqa" not in call.kwargs
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for actual_input, expected_input in zip(inputs, reference):
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-5, atol=1e-6)
