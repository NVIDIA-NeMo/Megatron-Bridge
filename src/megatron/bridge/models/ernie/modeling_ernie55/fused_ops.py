# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for Megatron-Bridge from the ERNIE 5.5 reference implementation's fused Triton ops.
# Changed from the original: Triton is imported and the kernels are defined on the first launch (_load_kernels), and
# the public functions got type hints and longer docstrings. The kernels and the autograd functions are otherwise the
# original's.
"""Triton SiTU-GLU, multimax lm_head and THD-CP local RoPE kernels.

The SiTU-GLU and multimax kernels run the eager fp32 chain in registers and round once to the input dtype. They
keep the eager operation order (``mul_rn`` / ``add_rn`` stop FMA contraction), so the SiTU forward / backward and
the multimax forward / logits gradient are bit-exact against it; only the multimax ranges / ts gradients sum in a
different order. The SiTU backward saves only the input and recomputes. The THD-CP RoPE kernel rounds every
product and the sum to the input dtype, like eager, and is bit-exact against it in forward and backward.

These are the reference implementation's kernels, which it dispatches on every CUDA tensor. The statements above
are the original's; Bridge does not retest them.

Bridge notes:

* Importing this module imports only PyTorch. :func:`_load_kernels` imports Triton and defines the kernels, as
  module globals, the first time a kernel launches, so CPU-only machines, machines without Triton and
  documentation builds can import it.
* Bridge calls these functions through
  :mod:`~megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch`, which keeps the eager path for CPU tensors,
  unsupported layouts and ``ernie_fused_ops=False``.
"""

import torch


__all__ = ["fused_multimax", "fused_situ_glu", "fused_thd_cp_local_rope"]

# Bound by _load_kernels() on the first launch. The kernels are module globals so that a kernel finds the helper
# kernels it calls: Triton resolves the names a kernel uses in the kernel's globals.
triton = None
tl = None
libdevice = None
_sigmoid_precise = None
_situ_glu_fwd_kernel = None
_situ_glu_bwd_kernel = None
_multimax_terms = None
_multimax_fwd_kernel = None
_multimax_bwd_kernel = None
_mul_round = None
_mla_rope_kernel = None
_KERNELS_LOADED = False


def _load_kernels() -> None:
    """Import Triton and define the kernels as module globals (once per process)."""
    global triton, tl, libdevice, _KERNELS_LOADED
    global _sigmoid_precise, _situ_glu_fwd_kernel, _situ_glu_bwd_kernel
    global _multimax_terms, _multimax_fwd_kernel, _multimax_bwd_kernel
    global _mul_round, _mla_rope_kernel
    if _KERNELS_LOADED:
        return
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def _sigmoid_precise(x):
        # torch's float sigmoid: 1 / (1 + expf(-x)) with an IEEE divide (tl.sigmoid uses ex2.approx).
        return libdevice.div_rn(1.0, 1.0 + libdevice.exp(-x))

    @triton.jit
    def _situ_glu_fwd_kernel(
        x_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        beta: tl.constexpr,
        inv_beta: tl.constexpr,
        linear_beta: tl.constexpr,
        inv_linear_beta: tl.constexpr,
        has_linear_tanh: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
        mask = cols < n_cols
        base = row * (2 * n_cols)
        gate = tl.load(x_ptr + base + cols, mask=mask).to(tl.float32)
        up = tl.load(x_ptr + base + n_cols + cols, mask=mask).to(tl.float32)
        # torch divides by a scalar as a multiply by its fp32 reciprocal.
        gate_act = (beta * libdevice.tanh(gate * inv_beta)) * _sigmoid_precise(gate)
        if has_linear_tanh:
            up = linear_beta * libdevice.tanh(up * inv_linear_beta)
        tl.store(out_ptr + row * n_cols + cols, gate_act * up, mask=mask)

    @triton.jit
    def _situ_glu_bwd_kernel(
        x_ptr,
        out_grad_ptr,
        x_grad_ptr,
        n_cols: tl.constexpr,
        beta: tl.constexpr,
        inv_beta: tl.constexpr,
        linear_beta: tl.constexpr,
        inv_linear_beta: tl.constexpr,
        has_linear_tanh: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
        mask = cols < n_cols
        base = row * (2 * n_cols)
        gate = tl.load(x_ptr + base + cols, mask=mask).to(tl.float32)
        up = tl.load(x_ptr + base + n_cols + cols, mask=mask).to(tl.float32)
        out_grad = tl.load(out_grad_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        gate_tanh = libdevice.tanh(gate * inv_beta)
        gate_scaled = beta * gate_tanh
        gate_sigmoid = _sigmoid_precise(gate)
        gate_act = gate_scaled * gate_sigmoid
        if has_linear_tanh:
            up_tanh = libdevice.tanh(up * inv_linear_beta)
            up_act = linear_beta * up_tanh
        else:
            up_act = up
        # Autograd's chain op by op: mul backward is grad * other, tanh_backward is grad * fma(-y, y, 1),
        # sigmoid_backward is (grad * (1 - y)) * y, and the scalar divide's backward multiplies by the reciprocal.
        gate_act_grad = libdevice.mul_rn(out_grad, up_act)
        up_act_grad = libdevice.mul_rn(out_grad, gate_act)
        tanh_grad = libdevice.mul_rn(libdevice.mul_rn(gate_act_grad, gate_sigmoid), beta)
        tanh_grad = libdevice.mul_rn(
            libdevice.mul_rn(tanh_grad, libdevice.fma_rn(-gate_tanh, gate_tanh, 1.0)), inv_beta
        )
        sigmoid_grad = libdevice.mul_rn(gate_act_grad, gate_scaled)
        sigmoid_grad = libdevice.mul_rn(
            libdevice.mul_rn(sigmoid_grad, libdevice.sub_rn(1.0, gate_sigmoid)), gate_sigmoid
        )
        if has_linear_tanh:
            up_act_grad = libdevice.mul_rn(up_act_grad, linear_beta)
            up_act_grad = libdevice.mul_rn(
                libdevice.mul_rn(up_act_grad, libdevice.fma_rn(-up_tanh, up_tanh, 1.0)), inv_linear_beta
            )
        tl.store(x_grad_ptr + base + cols, libdevice.add_rn(tanh_grad, sigmoid_grad), mask=mask)
        tl.store(x_grad_ptr + base + n_cols + cols, up_act_grad, mask=mask)

    @triton.jit
    def _multimax_terms(x, rt_ptr):
        r0, r1, r2, r3 = tl.load(rt_ptr), tl.load(rt_ptr + 1), tl.load(rt_ptr + 2), tl.load(rt_ptr + 3)
        t0, t1, t2, t3 = tl.load(rt_ptr + 4), tl.load(rt_ptr + 5), tl.load(rt_ptr + 6), tl.load(rt_ptr + 7)
        a = tl.maximum(r0 - x, 0.0)
        b = tl.maximum(x - r1, 0.0)
        c = tl.maximum(r2 - x, 0.0)
        d = tl.maximum(x - r3, 0.0)
        return a, b, c, d, t0, t1, t2, t3

    @triton.jit
    def _multimax_fwd_kernel(x_ptr, out_ptr, rt_ptr, numel, block: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        a, b, c, d, t0, t1, t2, t3 = _multimax_terms(x, rt_ptr)
        out = libdevice.add_rn(x, libdevice.mul_rn(t0, a))
        out = libdevice.add_rn(out, libdevice.mul_rn(t1, b))
        out = libdevice.add_rn(out, libdevice.mul_rn(t2, libdevice.mul_rn(c, c)))
        out = libdevice.add_rn(out, libdevice.mul_rn(t3, libdevice.mul_rn(d, d)))
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _multimax_bwd_kernel(x_ptr, grad_ptr, x_grad_ptr, rt_ptr, partial_ptr, numel, block: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * block + tl.arange(0, block)
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        a, b, c, d, t0, t1, t2, t3 = _multimax_terms(x, rt_ptr)
        # relu' is 0 at 0, as torch's threshold_backward.
        ga = tl.where(a > 0, g * t0, 0.0)
        gb = tl.where(b > 0, g * t1, 0.0)
        gc = g * t2 * 2.0 * c
        gd = g * t3 * 2.0 * d
        tl.store(x_grad_ptr + offs, g - ga + gb - gc + gd, mask=mask)
        # Per-block partial sums of d(r0..r3), d(t0..t3); the host sums the blocks in a fixed order.
        p = partial_ptr + pid.to(tl.int64) * 8
        tl.store(p, tl.sum(ga, 0))
        tl.store(p + 1, -tl.sum(gb, 0))
        tl.store(p + 2, tl.sum(gc, 0))
        tl.store(p + 3, -tl.sum(gd, 0))
        tl.store(p + 4, tl.sum(g * a, 0))
        tl.store(p + 5, tl.sum(g * b, 0))
        tl.store(p + 6, tl.sum(g * c * c, 0))
        tl.store(p + 7, tl.sum(g * d * d, 0))

    @triton.jit
    def _mul_round(a, b, dtype: tl.constexpr):
        # torch multiplies two half-precision tensors in fp32 and rounds the product to their dtype.
        return libdevice.mul_rn(a, b).to(dtype).to(tl.float32)

    @triton.jit
    def _mla_rope_kernel(
        x_ptr,
        out_ptr,
        cos_ptr,
        sin_ptr,
        n_heads,
        stride_t,
        stride_h,
        sign,
        nope_dim: tl.constexpr,
        half: tl.constexpr,
        swap_sin: tl.constexpr,
        block_h: tl.constexpr,
        block_nope: tl.constexpr,
        block_half: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        heads = tl.program_id(1) * block_h + tl.arange(0, block_h)
        head_mask = heads[:, None] < n_heads
        src = x_ptr + t * stride_t + heads[:, None].to(tl.int64) * stride_h
        dst = out_ptr + (t * n_heads + heads[:, None].to(tl.int64)) * (nope_dim + 2 * half)
        for start in tl.static_range(0, nope_dim, block_nope):
            cols = start + tl.arange(0, block_nope)[None, :]
            mask = head_mask & (cols < nope_dim)
            tl.store(dst + cols, tl.load(src + cols, mask=mask), mask=mask)
        # The rotary part is MLA-interleaved: (x[2i], x[2i+1]) rotate by angle i and stay interleaved in the output.
        i = tl.arange(0, block_half)
        i_mask = i < half
        mask = head_mask & i_mask[None, :]
        even = tl.load(src + nope_dim + 2 * i[None, :], mask=mask, other=0.0)
        dtype = even.dtype
        even = even.to(tl.float32)
        odd = tl.load(src + nope_dim + 2 * i[None, :] + 1, mask=mask, other=0.0).to(tl.float32)
        cos_l = tl.load(cos_ptr + t * 2 * half + i, mask=i_mask, other=0.0).to(tl.float32)[None, :]
        cos_r = tl.load(cos_ptr + t * 2 * half + half + i, mask=i_mask, other=0.0).to(tl.float32)[None, :]
        sin_l = sign * tl.load(sin_ptr + t * 2 * half + i, mask=i_mask, other=0.0).to(tl.float32)[None, :]
        sin_r = sign * tl.load(sin_ptr + t * 2 * half + half + i, mask=i_mask, other=0.0).to(tl.float32)[None, :]
        if swap_sin:
            sin_l, sin_r = sin_r, sin_l
        # Eager: t * cos + rotate_half(t) * sin with every product and the sum rounded to the input dtype.
        out_even = libdevice.add_rn(_mul_round(even, cos_l, dtype), _mul_round(-odd, sin_l, dtype))
        out_odd = libdevice.add_rn(_mul_round(odd, cos_r, dtype), _mul_round(even, sin_r, dtype))
        tl.store(dst + nope_dim + 2 * i[None, :], out_even, mask=mask)
        tl.store(dst + nope_dim + 2 * i[None, :] + 1, out_odd, mask=mask)

    _KERNELS_LOADED = True


def _situ_glu_launch(x, beta, linear_beta):
    n_cols = x.shape[-1] // 2
    block_n = min(1024, triton.next_power_of_2(n_cols))
    grid = (x.numel() // (2 * n_cols), triton.cdiv(n_cols, block_n))
    kwargs = dict(
        n_cols=n_cols,
        beta=float(beta),
        inv_beta=1.0 / float(beta),
        linear_beta=1.0 if linear_beta is None else float(linear_beta),
        inv_linear_beta=1.0 if linear_beta is None else 1.0 / float(linear_beta),
        has_linear_tanh=linear_beta is not None,
        block_n=block_n,
        num_warps=8 if block_n >= 512 else 4,
    )
    return grid, kwargs


class _FusedSituGlu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, linear_beta):
        out = x.new_empty(*x.shape[:-1], x.shape[-1] // 2)
        if out.numel():
            _load_kernels()
            grid, kwargs = _situ_glu_launch(x, beta, linear_beta)
            _situ_glu_fwd_kernel[grid](x, out, **kwargs)
        ctx.save_for_backward(x)
        ctx.beta, ctx.linear_beta = beta, linear_beta
        return out

    @staticmethod
    def backward(ctx, out_grad):
        (x,) = ctx.saved_tensors
        x_grad = torch.empty_like(x)
        if x.numel():
            _load_kernels()
            grid, kwargs = _situ_glu_launch(x, ctx.beta, ctx.linear_beta)
            _situ_glu_bwd_kernel[grid](x, out_grad.contiguous(), x_grad, **kwargs)
        return x_grad, None, None


def fused_situ_glu(x: torch.Tensor, beta: float, linear_beta: float | None = None) -> torch.Tensor:
    """``beta*tanh(g/beta)*sigmoid(g) * (linear_beta*tanh(u/linear_beta) or u)`` of ``x = [g, u]``, fp32 inside.

    Args:
        x: Contiguous CUDA tensor ``[..., 2 * n]``: the gate half, then the up half.
        beta: Scale of the gate's tanh.
        linear_beta: Scale of the up half's tanh; ``None`` leaves the up half linear.

    Returns:
        ``[..., n]`` in the dtype of ``x``.
    """
    return _FusedSituGlu.apply(x, beta, linear_beta)


_MULTIMAX_BLOCK = 4096


class _FusedMultimax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, ranges, ts):
        rt = torch.cat([ranges.detach().reshape(-1), ts.detach().reshape(-1)]).float()
        out = torch.empty_like(logits)
        numel = logits.numel()
        if numel:
            _load_kernels()
            _multimax_fwd_kernel[(triton.cdiv(numel, _MULTIMAX_BLOCK),)](
                logits, out, rt, numel, block=_MULTIMAX_BLOCK, num_warps=8
            )
        ctx.save_for_backward(logits, rt)
        ctx.meta = (ranges.shape, ranges.dtype), (ts.shape, ts.dtype)
        return out

    @staticmethod
    def backward(ctx, grad):
        logits, rt = ctx.saved_tensors
        numel = logits.numel()
        blocks = -(-numel // _MULTIMAX_BLOCK)
        x_grad = torch.empty_like(logits)
        partial = rt.new_zeros(blocks, 8)
        if numel:
            _load_kernels()
            _multimax_bwd_kernel[(blocks,)](
                logits, grad.contiguous(), x_grad, rt, partial, numel, block=_MULTIMAX_BLOCK, num_warps=8
            )
        rt_grad = partial.sum(0)
        (r_shape, r_dtype), (t_shape, t_dtype) = ctx.meta
        return x_grad, rt_grad[:4].view(r_shape).to(r_dtype), rt_grad[4:].view(t_shape).to(t_dtype)


def fused_multimax(logits: torch.Tensor, ranges: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """``apply_multimax(logits, ranges, ts, compute_in_fp32=True)`` for contiguous CUDA logits.

    Args:
        logits: Contiguous CUDA logits of any shape.
        ranges: The four ranges ``r``.
        ts: The four scales ``t``.

    Returns:
        ``x + t0 relu(r0 - x) + t1 relu(x - r1) + t2 relu(r2 - x)^2 + t3 relu(x - r3)^2`` computed in fp32, in a new
        tensor with the shape and dtype of ``logits``. The gradients of ``ranges`` and ``ts`` come back in their dtypes.
    """
    return _FusedMultimax.apply(logits, ranges, ts)


def _mla_rope_launch(x, cos, sin, nope_dim, inverse, transpose=False):
    n_tokens, n_heads, dim = x.shape
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    half = (dim - nope_dim) // 2
    if out.numel():
        _load_kernels()
        block_h = min(16, triton.next_power_of_2(n_heads))
        _mla_rope_kernel[(n_tokens, triton.cdiv(n_heads, block_h))](
            x,
            out,
            cos,
            sin,
            n_heads,
            x.stride(0),
            x.stride(1),
            -1.0 if inverse != transpose else 1.0,
            nope_dim=nope_dim,
            half=half,
            swap_sin=transpose,
            block_h=block_h,
            block_nope=min(256, triton.next_power_of_2(max(nope_dim, 1))),
            block_half=triton.next_power_of_2(max(half, 1)),
            num_warps=4,
        )
    return out


class _FusedMlaRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, nope_dim, inverse):
        ctx.save_for_backward(cos, sin)
        ctx.nope_dim, ctx.inverse = nope_dim, inverse
        return _mla_rope_launch(x, cos, sin, nope_dim, inverse)

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        # The transpose: the opposite rotation with the two sin halves swapped, rounded per product as autograd.
        grad = grad if grad.stride(-1) == 1 else grad.contiguous()
        return _mla_rope_launch(grad, cos, sin, ctx.nope_dim, ctx.inverse, transpose=True), None, None, None, None


def fused_thd_cp_local_rope(
    x: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: torch.Tensor,
    global_start: int,
    inverse: bool = False,
) -> torch.Tensor:
    """``cp_utils.apply_thd_cp_local_rope_unfused`` (non-interleaved table, MLA-interleaved rotary part) in one
    kernel per direction; bit-exact against it in forward and backward.

    Args:
        x: CUDA rows ``[t, heads, dim]``, ``[t, 1, heads, dim]`` or ``[t, dim]``; the last ``pos_dim`` channels rotate.
        rotary_pos_emb: The RoPE angle table, ``[max_seqlen, 1, 1, pos_dim]``.
        nope_dim: Leading channels that pass through.
        pos_dim: Rotary channels.
        cu_seqlens_padded: Packed document boundaries of the whole (global) THD row.
        global_start: Global packed row of ``x[0]`` (contiguous context-parallel partition).
        inverse: Rotate by the negative angles.

    Returns:
        A new tensor with the shape and dtype of ``x``.
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    position_ids = cp_utils._thd_cp_position_ids(cu_seqlens_padded, global_start, x.shape[0])
    freqs = torch.index_select(rotary_pos_emb, 0, position_ids.long()).reshape(x.shape[0], pos_dim)
    cos, sin = torch.cos(freqs).to(x.dtype), torch.sin(freqs).to(x.dtype)
    squeezed_batch = x.ndim == 4 and x.shape[1] == 1
    squeezed_head = x.ndim == 2
    rope_input = x.squeeze(1) if squeezed_batch else x
    rope_input = rope_input.unsqueeze(1) if squeezed_head else rope_input
    if rope_input.stride(-1) != 1:
        rope_input = rope_input.contiguous()
    output = _FusedMlaRope.apply(rope_input, cos, sin, nope_dim, inverse)
    if squeezed_batch:
        return output.unsqueeze(1)
    if squeezed_head:
        return output.squeeze(1)
    return output
