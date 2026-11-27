# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Numerical contracts for the model-local optimizations, independent of CUDA kernels."""

import math
from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.hyper_connection import (
    HyperConnectionExecutionContext,
    HyperConnectionResidualConnection,
)

from megatron.bridge.models.qwen.modeling_qwen4_exp.gated_residual import GatedResidualHyperConnection
from megatron.bridge.models.qwen.modeling_qwen4_exp.gated_residual_math import (
    activate_residual_gate,
    mix_residual_streams,
)
from megatron.bridge.models.qwen.modeling_qwen4_exp.norms import grouped_rmsnorm
from megatron.bridge.models.qwen.modeling_qwen4_exp.qsa_selection import select_document_blocks


def _reference_norm(x, weight, groups, eps):
    values = x.float().unflatten(-1, (groups, -1))
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (normalized.flatten(-2) * (1.0 + weight.float())).to(x.dtype)


@pytest.mark.parametrize("use_input_residual", [False, True])
def test_gated_residual_core_read_write_and_gradients(use_input_residual):
    torch.manual_seed(29)
    config = SimpleNamespace(
        mhc_num_residual_streams=2,
        hidden_size=4,
        mhc_gated_residual_rank=2,
        layernorm_epsilon=1e-6,
        params_dtype=torch.float32,
        perform_initialization=True,
        init_method_std=0.1,
        sequence_parallel=False,
    )
    owner = GatedResidualHyperConnection(config)
    context = HyperConnectionExecutionContext(owner)
    read = HyperConnectionResidualConnection(config, operation="read", use_input_residual=use_input_residual)
    write = HyperConnectionResidualConnection(config, operation="write")
    x = torch.randn(3, 1, 8, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)
    branch, state = read(x, execution_context=context)
    assert state[0] is x and state[1].numel() == 0
    actual = write(
        (branch.square(), bias), state=state, dropout_probability=0.0, training=True, execution_context=context
    )

    normalized = _reference_norm(x, owner.hc_norm.weight, 2, config.layernorm_epsilon)
    gate = owner.input_mix_weight_up(torch.nn.functional.silu(owner.input_mix_weight_down(normalized) / 2)).sigmoid()
    reference_branch = (gate * normalized).unflatten(-1, (2, 4)).mean(-2)
    inject = 2 * (owner.block_inject_weight(normalized) / 2).sigmoid()
    expected = (
        x.unflatten(-1, (2, 4)) + inject[..., None] * (reference_branch.square() + bias).unsqueeze(-2)
    ).flatten(-2)
    torch.testing.assert_close(actual, expected)
    inputs = (x, bias, *owner.parameters())
    for a, b in zip(torch.autograd.grad(actual.sum(), inputs), torch.autograd.grad(expected.sum(), inputs)):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
    with pytest.raises(NotImplementedError, match="recomputation"):
        owner(x, checkpoint_fn=lambda function, *args: function(*args))
    with pytest.raises(NotImplementedError, match="recomputation"):
        owner.fused_h_res_h_post_bda(
            state[1],
            x,
            state[2],
            (branch, bias),
            0.0,
            True,
            False,
            checkpoint_fn=lambda function, *args: function(*args),
        )


@pytest.mark.parametrize("dtype,groups,rows", [(torch.float32, 1, 0), (torch.float16, 4, 1), (torch.bfloat16, 4, 17)])
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_residual_gate_rounding_matches_hf(dtype):
    """Check eager operation boundaries rather than a widened FP32 expression."""
    torch.manual_seed(37)
    rows = 17
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


@pytest.mark.parametrize("lengths,workspace,ties", [([1, 3, 33], 1, False), ([37, 19, 2], 1 << 20, True)])
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


@pytest.mark.parametrize("batch,seq,workspace,ties", [(1, 1, 1, False), (2, 9, 1, False), (3, 12, 1 << 20, True)])
def test_unpacked_qsa_metadata_preserves_selection(batch, seq, workspace, ties):
    """Known batch rows retain exact bits at top-k boundaries, tails and ties."""
    torch.manual_seed(67)
    ratio, topk = 4, 2
    # Include a strided input: advanced indexing must keep the same GEMM layout.
    q = torch.randn(batch * seq, 3, 16)[..., ::2]
    if ties:
        q.zero_()
    keys = torch.randn(batch, seq // ratio, 8)
    valid = torch.ones(batch, seq // ratio, dtype=torch.bool)
    docs = torch.arange(batch).repeat_interleave(seq)
    positions = torch.arange(seq).repeat(batch)
    args = (q, keys, valid, docs, positions)
    kwargs = dict(compress_ratio=ratio, block_topk=topk, workspace_bytes=workspace)
    expected, expected_all = select_document_blocks(*args, **kwargs)
    actual, actual_all = select_document_blocks(*args, **kwargs, unpacked_sequence_length=seq)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_all == expected_all == (seq // ratio <= topk)
    if not actual_all:
        torch.testing.assert_close(actual, _replicated_key_selection(*args, ratio, topk), rtol=0, atol=0)
