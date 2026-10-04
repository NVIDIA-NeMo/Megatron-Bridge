"""Numerical contracts for the model-local optimizations, independent of CUDA kernels."""

import math

import pytest
import torch

from megatron.bridge.models.qwen.modeling_qwen4_exp.gated_residual_math import (
    activate_residual_gate,
    mix_residual_streams,
)
from megatron.bridge.models.qwen.modeling_qwen4_exp.norms import grouped_rmsnorm, norm_chunk_rows
from megatron.bridge.models.qwen.modeling_qwen4_exp.qsa_selection import select_document_blocks


def _reference_norm(x, weight, groups, eps):
    values = x.float().unflatten(-1, (groups, -1))
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (normalized.flatten(-2) * (1.0 + weight.float())).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("groups", [1, 4])
@pytest.mark.parametrize("rows", [0, 1, 17])
def test_chunked_norm_forward_and_gradients(dtype, groups, rows):
    """Compare every input and weight gradient with eager autograd across chunk boundaries."""
    torch.manual_seed(11)
    x = torch.randn(rows, 2, groups * 16, dtype=dtype).requires_grad_()
    weight = (torch.randn(groups * 16, dtype=dtype) * 0.1).requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    ref_w = weight.detach().clone().requires_grad_()
    actual = grouped_rmsnorm(x, weight, groups, 1e-6, workspace_bytes=1)
    expected = _reference_norm(ref_x, ref_w, groups, 1e-6)
    gradient = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, weight), gradient)
    reference_grads = torch.autograd.grad(expected, (ref_x, ref_w), gradient)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    tolerance = {torch.float32: 3e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    for actual_grad, ref_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, ref_grad, rtol=tolerance, atol=tolerance)


def test_noncontiguous_norm_inputs():
    """Views preserve both gradients and the original output shape."""
    torch.manual_seed(23)
    source = torch.randn(7, 3, 32, requires_grad=True)
    x = source.transpose(0, 1)
    weight = torch.randn(32, requires_grad=True)
    actual = grouped_rmsnorm(x, weight, 4, 1e-6, workspace_bytes=384)
    expected = _reference_norm(x, weight, 4, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(
        torch.autograd.grad(actual.sum(), (source, weight)),
        torch.autograd.grad(expected.sum(), (source, weight)),
    ):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)


def test_norm_rejects_double_backward():
    """Reject unsupported higher derivatives instead of treating saved scales as constants."""
    x = torch.randn(3, 8, requires_grad=True)
    weight = torch.zeros(8, requires_grad=True)
    output = grouped_rmsnorm(x, weight, 2, 1e-6)
    gradient = torch.autograd.grad(output.square().sum(), x, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), x)


def test_norm_saved_state_does_not_keep_token_sized_fp32_products():
    """The backward state contains raw activations and one statistic per stream."""
    x = torch.randn(257, 4, 64, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.zeros(64, dtype=torch.bfloat16, requires_grad=True)
    saved = []

    def record(tensor):
        saved.append((tensor.shape, tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        grouped_rmsnorm(x, weight, 4, 1e-6)
    assert saved == [(x.shape, x.dtype), (weight.shape, weight.dtype), (torch.Size([1028, 4, 1]), torch.float32)]
    assert norm_chunk_rows(10240, 64 << 20) * 8 * 10240 * 4 <= 64 << 20


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 17, 257])
def test_residual_gate_rounding_matches_hf(dtype, rows):
    """Check eager operation boundaries rather than a widened FP32 expression."""
    torch.manual_seed(37)
    down = torch.randn(rows, 32, dtype=dtype).requires_grad_()
    reference_down = down.detach().clone().requires_grad_()
    actual = activate_residual_gate(down, 4)
    expected = torch.nn.functional.silu(reference_down / 4)
    gradient = torch.randn_like(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, down, gradient)[0],
        torch.autograd.grad(expected, reference_down, gradient)[0],
        rtol=0,
        atol=0,
    )
    inputs = [torch.randn(rows, 4 * 64, dtype=dtype).requires_grad_() for _ in range(2)]
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    actual = mix_residual_streams(*inputs, 4)
    expected = (refs[0].sigmoid().reshape(rows, 4, 64) * refs[1].reshape(rows, 4, 64)).mean(-2)
    gradient = torch.randn_like(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(torch.autograd.grad(actual, inputs, gradient), torch.autograd.grad(expected, refs, gradient)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def _replicated_key_selection(q, keys, valid_blocks, docs, positions, ratio, topk):
    blocks = keys.shape[1]
    scores = torch.einsum("thd,tnd->thn", q.float(), keys[docs.long()].float()).relu().sum(1) / math.sqrt(q.shape[-1])
    valid = torch.arange(blocks).unsqueeze(0) < ((positions.long() + 1) // ratio).unsqueeze(1)
    valid &= valid_blocks[docs.long()]
    scores.masked_fill_(~valid, float("-inf"))
    values, selected = scores.topk(min(topk, blocks), dim=-1)
    kept = torch.isfinite(values)
    selected = selected.masked_fill(~kept, 0)
    bits = torch.zeros(q.shape[0], (blocks + 8) // 8, dtype=torch.int32)
    bits.scatter_add_(1, selected >> 3, ((1 << (selected & 7)) * kept).to(torch.int32))
    return bits.to(torch.uint8)


@pytest.mark.parametrize("lengths", [[1, 3, 33], [37, 19, 2], [40, 40]])
@pytest.mark.parametrize("workspace", [1, 1 << 20])
@pytest.mark.parametrize("ties", [False, True])
def test_document_local_qsa_preserves_selected_blocks(lengths, workspace, ties):
    """Packed boundaries, partial tails and top-k ties retain the actual selected IDs."""
    torch.manual_seed(51)
    ratio, topk, heads, dim = 4, 2, 3, 8
    blocks = max(lengths) // ratio
    docs = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    positions = torch.cat([torch.arange(length) for length in lengths])
    q = torch.randn(sum(lengths), heads, dim)
    keys = torch.randn(len(lengths), blocks, dim)
    if ties:
        q.zero_()
    valid = torch.arange(blocks).unsqueeze(0) < (torch.tensor(lengths) // ratio).unsqueeze(1)
    actual, _ = select_document_blocks(
        q,
        keys,
        valid,
        docs,
        positions,
        compress_ratio=ratio,
        block_topk=topk,
        workspace_bytes=workspace,
    )
    expected = _replicated_key_selection(q, keys, valid, docs, positions, ratio, topk)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qsa_documents_without_complete_blocks():
    """The current partial block needs no completed-block bit."""
    bits, all_selected = select_document_blocks(
        torch.randn(3, 2, 8),
        torch.empty(1, 0, 8),
        torch.empty(1, 0, dtype=torch.bool),
        torch.zeros(3, dtype=torch.long),
        torch.arange(3),
        compress_ratio=4,
        block_topk=2,
    )
    assert all_selected and bits.shape == (3, 1) and not bits.any()


@pytest.mark.parametrize("groups,width", [(0, 4), (3, 4), (1, 0)])
def test_invalid_normalization_shape_fails(groups, width):
    """Reject malformed group layouts before any output allocation."""
    with pytest.raises(ValueError):
        grouped_rmsnorm(torch.empty(2, width), torch.empty(width), groups, 1e-6)
