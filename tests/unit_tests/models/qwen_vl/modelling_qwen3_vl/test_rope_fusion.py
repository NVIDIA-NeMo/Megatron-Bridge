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

"""Fused RoPE for Qwen's per-token (MRoPE and vision 2D) rotary embedding."""

import types

import pytest
import torch

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import rope
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import apply_rotary_pos_emb_absolute


pytestmark = pytest.mark.unit

requires_fused_rope = pytest.mark.skipif(
    not torch.cuda.is_available() or rope.fused_apply_rotary_pos_emb is None,
    reason="requires CUDA and Transformer Engine's fused RoPE",
)

SEQ, HEADS, HEAD_DIM = 256, 4, 64


def _config(*, fusion: bool, fp32: bool = False, interleaved: bool = False) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        apply_rope_fusion=fusion, apply_rotary_pos_emb_in_fp32=fp32, rotary_interleaved=interleaved
    )


def _per_token_freqs(rows: int, batch: int, rotary_dim: int, device: str = "cuda") -> torch.Tensor:
    """[rows, batch, 1, rotary_dim] angles at arbitrary per-sample positions, like MRoPE."""
    positions = torch.randint(0, 8 * rows, (rows, batch), device=device).float()
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    angles = positions[..., None] * inv_freq
    return torch.cat((angles, angles), dim=-1)[:, :, None, :]


def _rope_fp64(t: torch.Tensor, freqs: torch.Tensor, *, interleaved: bool) -> torch.Tensor:
    x, f = t.double(), freqs.double()
    if x.dim() == 3:
        f = f[:, 0]
    rotary_dim = f.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    if interleaved:
        rotated = torch.stack((-x_rot[..., 1::2], x_rot[..., ::2]), dim=-1).flatten(-2)
    else:
        x1, x2 = x_rot.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat((x_rot * f.cos() + rotated * f.sin(), x_pass), dim=-1)


Q_OF_QKV = (Ellipsis, slice(0, HEAD_DIM))  # q as a strided view into a fused QKV projection, as attention passes it


def _fwd_bwd(base, freqs, config, grad, cu_seqlens=None, index=Q_OF_QKV):
    """RoPE on ``base[index]`` with its real strides; returns (out, input gradient)."""
    leaf = base.detach().clone().requires_grad_(True)
    out = apply_rotary_pos_emb_absolute(leaf[index], freqs, config, cu_seqlens)
    (dbase,) = torch.autograd.grad(out, leaf, grad)
    return out.detach(), dbase[index]


def _fp64_fwd_bwd(base, freqs, grad, *, interleaved, index=Q_OF_QKV):
    x = base[index].detach().double().requires_grad_(True)
    out = _rope_fp64(x, freqs, interleaved=interleaved)
    (dx,) = torch.autograd.grad(out, x, grad.double())
    return out.detach(), dx


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


@pytest.fixture
def fused_calls(monkeypatch):
    calls = []
    kernel = rope.fused_apply_rotary_pos_emb

    def spy(t, freqs, **kwargs):
        calls.append((tuple(t.shape), tuple(freqs.shape)))
        return kernel(t, freqs, **kwargs)

    monkeypatch.setattr(rope, "fused_apply_rotary_pos_emb", spy)
    monkeypatch.setattr(rope, "_FUSED_ROPE_FALLBACK_REASONS", set())
    return calls


def _check_fused_against_unfused(base, freqs, grad, *, cu_seqlens=None, fp32=False, interleaved=False, index=Q_OF_QKV):
    fused, unfused = (_config(fusion=f, fp32=fp32, interleaved=interleaved) for f in (True, False))
    out_f, dq_f = _fwd_bwd(base, freqs, fused, grad, cu_seqlens, index)
    out_u, dq_u = _fwd_bwd(base, freqs, unfused, grad, cu_seqlens, index)
    out_64, dq_64 = _fp64_fwd_bwd(base, freqs, grad, interleaved=interleaved, index=index)
    q = base[index]
    assert out_f.shape == q.shape and out_f.dtype == q.dtype and dq_f.dtype == q.dtype
    # The kernel computes in FP32 and rounds once; the unfused BF16 chain rounds after every op. Fused stays within
    # a few BF16 ulps of unfused and is at least as close to FP64.
    assert _max_rel(out_f, out_u) <= 2**-6 and _max_rel(dq_f, dq_u) <= 2**-6
    assert _max_rel(out_f, out_64) <= max(1.05 * _max_rel(out_u, out_64), 2**-20)
    assert _max_rel(dq_f, dq_64) <= max(1.05 * _max_rel(dq_u, dq_64), 2**-20)
    rotary_dim = freqs.shape[-1]
    assert torch.equal(out_f[..., rotary_dim:], q[..., rotary_dim:])


@requires_fused_rope
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("fp32", [False, True])
def test_fused_sbhd_per_token_freqs(fused_calls, batch, interleaved, fp32):
    torch.manual_seed(0)
    qkv = torch.randn(SEQ, batch, HEADS, 3 * HEAD_DIM, device="cuda").to(torch.bfloat16)
    freqs = _per_token_freqs(SEQ, batch, rotary_dim=16)  # partial rotary: 16 of 64
    grad = torch.randn(SEQ, batch, HEADS, HEAD_DIM, device="cuda").to(torch.bfloat16)
    _check_fused_against_unfused(qkv, freqs, grad, fp32=fp32, interleaved=interleaved)
    # one launch per call; the micro batch is presented as one [s * b, 1] sequence of per-(position, sample) rows
    assert fused_calls == [((SEQ * batch, 1, HEADS, HEAD_DIM), (SEQ * batch, 1, 1, 16))]


@requires_fused_rope
def test_fused_sbhd_shared_freqs_broadcast_over_batch(fused_calls):
    torch.manual_seed(1)
    qkv = torch.randn(SEQ, 3, HEADS, 3 * HEAD_DIM, device="cuda").to(torch.bfloat16)
    freqs = _per_token_freqs(SEQ, 1, rotary_dim=HEAD_DIM)
    grad = torch.randn(SEQ, 3, HEADS, HEAD_DIM, device="cuda").to(torch.bfloat16)
    _check_fused_against_unfused(qkv, freqs, grad)
    assert fused_calls == [((SEQ, 3, HEADS, HEAD_DIM), (SEQ, 1, 1, HEAD_DIM))]


@requires_fused_rope
def test_fused_sbhd_unmergeable_batch_slice(fused_calls):
    torch.manual_seed(2)
    # A batch slice of a larger buffer (as a KV-cache slice is): sequence and batch strides do not merge, so the
    # input is copied before the kernel.
    base = torch.randn(SEQ, 4, HEADS, 3 * HEAD_DIM, device="cuda").to(torch.bfloat16)
    index = (slice(None), slice(1, 3), Ellipsis, slice(0, HEAD_DIM))
    assert base[index].stride(0) != 2 * base[index].stride(1)
    freqs = _per_token_freqs(SEQ, 2, rotary_dim=32)
    grad = torch.randn(SEQ, 2, HEADS, HEAD_DIM, device="cuda").to(torch.bfloat16)
    _check_fused_against_unfused(base, freqs, grad, index=index)
    assert len(fused_calls) == 1


@requires_fused_rope
def test_fused_thd_pairs_token_i_with_row_i(fused_calls):
    torch.manual_seed(3)
    tokens = 3 * 96
    qkv = torch.randn(tokens, HEADS, 3 * HEAD_DIM, device="cuda").to(torch.bfloat16)
    freqs = _per_token_freqs(tokens, 1, rotary_dim=HEAD_DIM)
    grad = torch.randn(tokens, HEADS, HEAD_DIM, device="cuda").to(torch.bfloat16)
    cu_seqlens = torch.tensor([0, 64, 160, tokens], dtype=torch.int32, device="cuda")
    _check_fused_against_unfused(qkv, freqs, grad, cu_seqlens=cu_seqlens, fp32=True)
    # Like the unfused code, rows are not looked up by position within a sequence: boundaries do not matter.
    other = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    out_a, _ = _fwd_bwd(qkv, freqs, _config(fusion=True), grad, cu_seqlens)
    out_b, _ = _fwd_bwd(qkv, freqs, _config(fusion=True), grad, other)
    assert torch.equal(out_a, out_b)
    assert all(shape == (tokens, 1, HEADS, HEAD_DIM) for shape, _ in fused_calls)


@requires_fused_rope
def test_fused_skips_fp32_casts_bitwise(fused_calls):
    torch.manual_seed(4)
    q = torch.randn(SEQ, 2, HEADS, HEAD_DIM, device="cuda").to(torch.bfloat16)
    freqs = _per_token_freqs(SEQ, 2, rotary_dim=HEAD_DIM)
    fused = apply_rotary_pos_emb_absolute(q, freqs, _config(fusion=True, fp32=True))
    cast = rope.fused_apply_rotary_pos_emb(
        q.float().reshape(-1, 1, HEADS, HEAD_DIM), freqs.reshape(-1, 1, 1, HEAD_DIM)
    )
    assert torch.equal(fused, cast.to(torch.bfloat16).view_as(q))


@requires_fused_rope
def test_fused_rope_cuda_graph_replay_with_new_freqs(fused_calls):
    torch.manual_seed(5)
    config = _config(fusion=True)
    static_x = torch.randn(SEQ, 2, HEADS, HEAD_DIM, device="cuda").to(torch.bfloat16).requires_grad_(True)
    static_freqs = _per_token_freqs(SEQ, 2, rotary_dim=32).contiguous()
    static_grad = torch.randn(SEQ, 2, HEADS, HEAD_DIM, device="cuda").to(torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            torch.autograd.grad(apply_rotary_pos_emb_absolute(static_x, static_freqs, config), static_x, static_grad)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = apply_rotary_pos_emb_absolute(static_x, static_freqs, config)
        (static_dx,) = torch.autograd.grad(static_out, static_x, static_grad)
    for _ in range(2):
        x = torch.randn_like(static_x)
        freqs = _per_token_freqs(SEQ, 2, rotary_dim=32)
        grad = torch.randn_like(static_grad)
        with torch.no_grad():
            static_x.copy_(x)
            static_freqs.copy_(freqs)
            static_grad.copy_(grad)
        graph.replay()
        eager_x = x.detach().clone().requires_grad_(True)
        eager_out = apply_rotary_pos_emb_absolute(eager_x, freqs, config)
        (eager_dx,) = torch.autograd.grad(eager_out, eager_x, grad)
        assert torch.equal(static_out, eager_out) and torch.equal(static_dx, eager_dx)


@requires_fused_rope
def test_fused_rope_falls_back_once_when_freqs_require_grad(fused_calls):
    torch.manual_seed(6)
    q = torch.randn(SEQ, 2, HEADS, HEAD_DIM, device="cuda").to(torch.bfloat16)
    freqs = _per_token_freqs(SEQ, 2, rotary_dim=HEAD_DIM).requires_grad_(True)
    with pytest.warns(UserWarning, match="require gradients"):
        out = apply_rotary_pos_emb_absolute(q, freqs, _config(fusion=True))
    assert torch.equal(out, apply_rotary_pos_emb_absolute(q, freqs, _config(fusion=False)))
    assert fused_calls == []
    assert len(rope._FUSED_ROPE_FALLBACK_REASONS) == 1


def test_fused_rope_falls_back_on_cpu(monkeypatch):
    monkeypatch.setattr(rope, "_FUSED_ROPE_FALLBACK_REASONS", set())
    torch.manual_seed(7)
    q = torch.randn(16, 2, 2, 8)
    freqs = _per_token_freqs(16, 2, rotary_dim=8, device="cpu")
    with pytest.warns(UserWarning, match="not on a CUDA device|not available"):
        out = apply_rotary_pos_emb_absolute(q, freqs, _config(fusion=True))
    assert torch.equal(out, apply_rotary_pos_emb_absolute(q, freqs, _config(fusion=False)))


class _TensorMetadata:
    """Metadata-only stand-in for a CUDA tensor: the eligibility check never reads data."""

    is_cuda = True

    def __init__(self, shape, stride=None, storage_offset=0):
        self.shape = torch.Size(shape)
        contiguous, step = [], 1
        for size in reversed(shape):
            contiguous.insert(0, step)
            step *= size
        self._stride = tuple(stride) if stride is not None else tuple(contiguous)
        self._storage_offset = storage_offset

    def dim(self):
        return len(self.shape)

    def size(self, dim):
        return self.shape[dim]

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self._storage_offset

    def numel(self):
        return self.shape.numel()


@pytest.mark.parametrize(
    ("shape", "freqs_shape", "cu_seqlens", "reason"),
    [
        ((8, 2, 2, 8), (8, 2, 8), None, "not shaped"),
        ((8, 2, 2, 8), (8, 2, 1, 7), None, "odd"),
        ((8, 2, 2, 8), (8, 2, 1, 16), None, "larger than the head dimension"),
        ((8, 2, 2, 8), (4, 2, 1, 8), None, "one row per sequence position"),
        ((8, 2, 2, 8), (8, 3, 1, 8), None, "one row per sequence position"),
        ((8, 2, 8), (6, 1, 1, 8), torch.tensor([0, 8]), "one row per token"),
        ((0, 2, 2, 8), (0, 2, 1, 8), None, "empty"),
    ],
)
def test_fused_rope_unsupported_reasons(monkeypatch, shape, freqs_shape, cu_seqlens, reason):
    monkeypatch.setattr(rope, "fused_apply_rotary_pos_emb", object())
    freqs = torch.zeros(freqs_shape)
    assert reason in rope._fused_rope_unsupported_reason(_TensorMetadata(shape), freqs, cu_seqlens)


def test_fused_rope_rejects_inputs_beyond_32_bit_offsets(monkeypatch):
    monkeypatch.setattr(rope, "fused_apply_rotary_pos_emb", object())
    # A view whose last element sits 2**31 elements past its first: the kernel's 32-bit offsets would overflow.
    t = _TensorMetadata((2, 1, 1, 8), stride=(2**31 - 7, 8, 8, 1))
    assert "32-bit" in rope._fused_rope_unsupported_reason(t, torch.zeros(2, 1, 1, 8), None)
    assert rope._fused_rope_unsupported_reason(_TensorMetadata((2, 1, 1, 8)), torch.zeros(2, 1, 1, 8), None) is None
