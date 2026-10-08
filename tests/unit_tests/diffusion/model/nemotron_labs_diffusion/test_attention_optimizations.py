"""Numerical coverage for grouped-query attention and compiled block masks."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from megatron.bridge.diffusion.common.dllm import (
    asymmetric_semi_ar_mask_mod,
    build_block_mask,
    compute_block_mask,
)
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


def assert_same_blocks(actual: BlockMask, expected: BlockMask) -> None:
    for field in (
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
        "q_num_blocks",
        "q_indices",
        "full_q_num_blocks",
        "full_q_indices",
    ):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)


@pytest.mark.parametrize("half_length", [64, 96])
def test_compiled_symmetric_mask_matches_eager(half_length: int) -> None:
    actual = compute_block_mask(16, half_length, device="cpu")
    expected = create_block_mask(
        actual.mask_mod, B=None, H=None, Q_LEN=2 * half_length, KV_LEN=2 * half_length, device="cpu"
    )
    assert_same_blocks(actual, expected)


@pytest.mark.parametrize("noisy_length,clean_length", [(64, 96), (96, 160)])
def test_compiled_asymmetric_mask_matches_eager(noisy_length: int, clean_length: int) -> None:
    predicate = asymmetric_semi_ar_mask_mod(
        block_size=16,
        noisy_length=noisy_length,
        noisy_response_offset=0,
        prompt_lengths=torch.tensor([7, 19]),
        noisy_valid_lengths=torch.tensor([32, noisy_length]),
        clean_lengths=torch.tensor([39, clean_length]),
    )
    length = noisy_length + clean_length
    actual = build_block_mask(predicate, batch_size=2, query_length=length, key_value_length=length, device="cpu")
    expected = create_block_mask(predicate, B=2, H=None, Q_LEN=length, KV_LEN=length, device="cpu")
    assert_same_blocks(actual, expected)
