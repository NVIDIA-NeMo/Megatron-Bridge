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

"""Fused fp32 Triton kernels for the ERNIE 5.5 hyper-connections (``ernie_fused_mhc``).

The eager ERNIE 5.5 mHC (:mod:`~megatron.bridge.models.ernie.modeling_ernie55.hyper_connection`) launches
about 140 small kernels per hyper-connection unit in the forward pass and about 250 in the backward pass, most
of them fp32 elementwise operations and reductions over a few values per token. This module computes the same
fp32 contract with a few kernels per unit:

* **Mappings** (:func:`fused_mhc_mappings`). The projection stays one cuBLAS GEMM, ``fp32(X) @ W^T``, on the
  transposed view of ``W`` (the eager path's contiguous ``W^T`` copy would be one more kernel per call), so it
  matches the eager projection to fp32 rounding under the same TF32 setting. One Triton epilogue computes the
  stream norm, ``r = 1 / (||X|| / sqrt(K) + 1e-6)``, ``h = r * proj * alpha + bias``, ``h_pre = sigmoid + 1e-6``,
  ``h_post = 2 sigmoid`` and the Sinkhorn projection of the residual logits, all in fp32 registers. The backward
  pass recomputes the epilogue from the saved projection and norm (the Sinkhorn states are re-derived per reverse
  step, O(iterations^2) small normalizations per token), writes the projection gradient, the per-token norm
  coefficient and per-program partial sums of the ``alpha`` and ``bias`` gradients, and leaves the two
  weight-side products to cuBLAS.
* **Aggregate** (:func:`fused_mhc_aggregate`): ``sum_j h_pre_j x_j`` in fp32, one kernel per direction.
* **Residual update** (:func:`fused_mhc_post_bda`): ``x_i <- sum_j H_res[j, i] x_j + h_post_i * y`` in fp32,
  rounded once to the stream dtype, one kernel per direction. The layer output bias is not supported (ERNIE 5.5
  has ``add_bias_linear=False``).

Numerics. The kernels keep the eager operation order where it is cheap (IEEE divides via ``div_rn``, the
precise ``expf`` of torch's softmax and sigmoid, unfused ``r * proj * alpha``) but sum in their own order, so
they are not bitwise equal to the eager path: the mappings agree to about 1e-6 relative and the mixing to fp32
rounding. bf16 stream outputs are both a single rounding of an fp32 sum, so they differ on a small fraction of
elements, by one bf16 ulp of the larger value plus the fp32 difference of the sums (which exceeds one ulp only
where the update nearly cancels to zero).

Determinism. Every launch configuration is a fixed module constant (no autotuning: the backward reduction order
depends on the block sizes), no kernel uses atomics, and no code path depends on the grad mode, so the forward
of a full-recompute replay reproduces the first pass bitwise, and repeated runs give the same bits.

Importing this module imports only PyTorch; :func:`_load_kernels` imports Triton and defines the kernels on the
first launch, as :mod:`~megatron.bridge.models.ernie.modeling_ernie55.fused_ops` does.
"""

from __future__ import annotations

import torch
from torch import Tensor


__all__ = [
    "fused_mhc_aggregate",
    "fused_mhc_mappings",
    "fused_mhc_post_bda",
    "fused_mhc_supports",
    "fused_mhc_unsupported_reason",
]

STREAM_DTYPES = (torch.float32, torch.bfloat16)
"""Residual-stream dtypes the kernels take: fp32 (decoder) and bf16 (MTP)."""

LAYER_OUTPUT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
"""Layer-output dtypes the residual update takes."""

MAX_STREAMS = 8
"""Largest number of residual streams the kernels take (a power of two, at least 2)."""

# Launch constants, fixed so that every reduction order is fixed (see the module docstring).
_MAP_FWD_BLOCK_M = 4
_MAP_FWD_BLOCK_K = 1024
_MAP_FWD_WARPS = 4
_MAP_BWD_BLOCK_M = 16
_MAP_BWD_WARPS = 4
_NORM_GRAD_BLOCK = 1024
_NORM_GRAD_WARPS = 4
_ROW_FWD_BLOCK_S = 1
_ROW_FWD_BLOCK_C = 1024
_ROW_FWD_WARPS = 4
_ROW_BWD_BLOCK_S = 1
_ROW_BWD_BLOCK_C = 512
_ROW_BWD_WARPS = 4

# Bound by _load_kernels() on the first launch. The kernels are module globals so that a kernel finds the helper
# functions it calls: Triton resolves the names a kernel uses in the kernel's globals.
triton = None
tl = None
libdevice = None
_mhc_sigmoid = None
_mhc_softmax = None
_mhc_col_normalize = None
_mhc_row_normalize = None
_mhc_map_fwd = None
_mhc_map_bwd = None
_mhc_norm_grad_add = None
_mhc_agg_fwd = None
_mhc_agg_bwd = None
_mhc_post_bda_fwd = None
_mhc_post_bda_bwd = None
_KERNELS_LOADED = False


def _load_kernels() -> None:
    """Import Triton and define the kernels as module globals (once per process)."""
    global triton, tl, libdevice, _KERNELS_LOADED
    global _mhc_sigmoid, _mhc_softmax, _mhc_col_normalize, _mhc_row_normalize
    global _mhc_map_fwd, _mhc_map_bwd, _mhc_norm_grad_add
    global _mhc_agg_fwd, _mhc_agg_bwd, _mhc_post_bda_fwd, _mhc_post_bda_bwd
    if _KERNELS_LOADED:
        return
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def _mhc_sigmoid(x):
        # torch's float sigmoid: 1 / (1 + expf(-x)) with an IEEE divide (tl.sigmoid uses ex2.approx).
        return libdevice.div_rn(1.0, 1.0 + libdevice.exp(-x))

    @triton.jit
    def _mhc_softmax(logits):
        # torch's softmax over the last axis of [rows, n, n]: exp(x - max) / sum.
        e = libdevice.exp(logits - tl.max(logits, axis=2)[:, :, None])
        return libdevice.div_rn(e, tl.sum(e, axis=2)[:, :, None])

    @triton.jit
    def _mhc_col_normalize(mat, eps):
        # [rows, n, n] / (column sum + eps): a column sum runs over the rows (axis 1).
        return libdevice.div_rn(mat, tl.sum(mat, axis=1)[:, None, :] + eps)

    @triton.jit
    def _mhc_row_normalize(mat, eps):
        # [rows, n, n] / (row sum + eps): a row sum runs over the columns (axis 2).
        return libdevice.div_rn(mat, tl.sum(mat, axis=2)[:, :, None] + eps)

    @triton.jit
    def _mhc_map_fwd(
        x_ptr,
        proj_ptr,
        alpha_pre_ptr,
        alpha_post_ptr,
        alpha_res_ptr,
        bias_ptr,
        h_pre_ptr,
        h_post_ptr,
        h_res_ptr,
        nrm_ptr,
        rows_total,
        width,
        inv_sqrt_width,
        norm_eps,
        compute_h_eps,
        sinkhorn_eps,
        N: tl.constexpr,
        NP: tl.constexpr,
        COLUMN_STEPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < rows_total
        rows64 = rows.to(tl.int64)

        # ||X||_2 over the K = n*C values of each token, in a fixed chunk order.
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for k0 in range(0, width, BLOCK_K):
            cols = k0 + tl.arange(0, BLOCK_K)
            mask = row_mask[:, None] & (cols[None, :] < width)
            xv = tl.load(x_ptr + rows64[:, None] * width + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            acc += xv * xv
        nrm = libdevice.sqrt_rn(tl.sum(acc, axis=1))
        tl.store(nrm_ptr + rows64, nrm, mask=row_mask)
        # Eager: 1.0 / (norm / sqrt(K) + eps); torch divides by a Python scalar as a multiply by its reciprocal.
        r = libdevice.div_rn(1.0, libdevice.mul_rn(nrm, inv_sqrt_width) + norm_eps)

        idx = tl.arange(0, N)
        proj_row = proj_ptr + rows64 * NP
        res_off = 2 * N + idx[None, :, None] * N + idx[None, None, :]
        p_pre = tl.load(proj_row[:, None] + idx[None, :], mask=row_mask[:, None], other=0.0)
        p_post = tl.load(proj_row[:, None] + N + idx[None, :], mask=row_mask[:, None], other=0.0)
        p_res = tl.load(proj_row[:, None, None] + res_off, mask=row_mask[:, None, None], other=0.0)
        a_pre = tl.load(alpha_pre_ptr).to(tl.float32)
        a_post = tl.load(alpha_post_ptr).to(tl.float32)
        a_res = tl.load(alpha_res_ptr).to(tl.float32)
        b_pre = tl.load(bias_ptr + idx).to(tl.float32)
        b_post = tl.load(bias_ptr + N + idx).to(tl.float32)
        b_res = tl.load(bias_ptr + res_off).to(tl.float32)

        # h = r * proj * alpha + bias, multiplied left to right as the eager expression.
        l_pre = libdevice.mul_rn(libdevice.mul_rn(r[:, None], p_pre), a_pre) + b_pre[None, :]
        l_post = libdevice.mul_rn(libdevice.mul_rn(r[:, None], p_post), a_post) + b_post[None, :]
        l_res = libdevice.mul_rn(libdevice.mul_rn(r[:, None, None], p_res), a_res) + b_res

        h_pre = _mhc_sigmoid(l_pre) + compute_h_eps
        h_post = _mhc_sigmoid(l_post) * 2.0
        # Sinkhorn: softmax + eps, one column normalization, then (iterations - 1) row and column rounds.
        mat = _mhc_col_normalize(_mhc_softmax(l_res) + sinkhorn_eps, sinkhorn_eps)
        for _ in range(1, COLUMN_STEPS):
            mat = _mhc_col_normalize(_mhc_row_normalize(mat, sinkhorn_eps), sinkhorn_eps)

        out2 = rows64[:, None] * N + idx[None, :]
        tl.store(h_pre_ptr + out2, h_pre, mask=row_mask[:, None])
        tl.store(h_post_ptr + out2, h_post, mask=row_mask[:, None])
        out3 = rows64[:, None, None] * (N * N) + idx[None, :, None] * N + idx[None, None, :]
        tl.store(h_res_ptr + out3, mat, mask=row_mask[:, None, None])

    @triton.jit
    def _mhc_map_bwd(
        g_pre_ptr,
        g_post_ptr,
        g_res_ptr,
        proj_ptr,
        nrm_ptr,
        alpha_pre_ptr,
        alpha_post_ptr,
        alpha_res_ptr,
        bias_ptr,
        g_proj_ptr,
        coef_ptr,
        partial_ptr,
        rows_total,
        inv_sqrt_width,
        norm_eps,
        sinkhorn_eps,
        N: tl.constexpr,
        NP: tl.constexpr,
        COLUMN_STEPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < rows_total
        rows64 = rows.to(tl.int64)
        idx = tl.arange(0, N)
        m2 = row_mask[:, None]
        m3 = row_mask[:, None, None]

        # Recompute the forward epilogue. Masked rows load zero gradients, so they add nothing to the partials.
        nrm = tl.load(nrm_ptr + rows64, mask=row_mask, other=0.0)
        r = libdevice.div_rn(1.0, libdevice.mul_rn(nrm, inv_sqrt_width) + norm_eps)
        proj_row = proj_ptr + rows64 * NP
        res_off = 2 * N + idx[None, :, None] * N + idx[None, None, :]
        p_pre = tl.load(proj_row[:, None] + idx[None, :], mask=m2, other=0.0)
        p_post = tl.load(proj_row[:, None] + N + idx[None, :], mask=m2, other=0.0)
        p_res = tl.load(proj_row[:, None, None] + res_off, mask=m3, other=0.0)
        a_pre = tl.load(alpha_pre_ptr).to(tl.float32)
        a_post = tl.load(alpha_post_ptr).to(tl.float32)
        a_res = tl.load(alpha_res_ptr).to(tl.float32)
        b_pre = tl.load(bias_ptr + idx).to(tl.float32)
        b_post = tl.load(bias_ptr + N + idx).to(tl.float32)
        b_res = tl.load(bias_ptr + res_off).to(tl.float32)
        rp_pre = libdevice.mul_rn(r[:, None], p_pre)
        rp_post = libdevice.mul_rn(r[:, None], p_post)
        rp_res = libdevice.mul_rn(r[:, None, None], p_res)
        s_pre = _mhc_sigmoid(libdevice.mul_rn(rp_pre, a_pre) + b_pre[None, :])
        s_post = _mhc_sigmoid(libdevice.mul_rn(rp_post, a_post) + b_post[None, :])
        l_res = libdevice.mul_rn(rp_res, a_res) + b_res

        out2 = rows64[:, None] * N + idx[None, :]
        out3 = rows64[:, None, None] * (N * N) + idx[None, :, None] * N + idx[None, None, :]
        g_pre = tl.load(g_pre_ptr + out2, mask=m2, other=0.0)
        g_post = tl.load(g_post_ptr + out2, mask=m2, other=0.0)
        g = tl.load(g_res_ptr + out3, mask=m3, other=0.0)

        # sigmoid backward as torch: (grad * (1 - y)) * y; h_post = 2 * sigmoid scales the gradient first.
        gl_pre = libdevice.mul_rn(libdevice.mul_rn(g_pre, 1.0 - s_pre), s_pre)
        gl_post = libdevice.mul_rn(libdevice.mul_rn(g_post * 2.0, 1.0 - s_post), s_post)

        # Sinkhorn VJP, last normalization first. With A_1 = col(M0), B_{k+1} = row(A_k), A_{k+1} = col(B_{k+1}),
        # the reverse step through round t (t = iterations-1 .. 1) needs A_t (input of the row normalization),
        # B_{t+1} and A_{t+1}; they are re-derived from M0 in registers. y = v / (sum(v) + eps) has the VJP
        # g' = g/(s+eps) - sum(g/(s+eps) * y).
        # The inner loop carries exactly one value and never copies one name into another: Triton makes a name
        # loop-carried only if its value after a dry run of the body differs from its live-in value, so a copy
        # such as `a_prev = a_t` (with a_prev == a_t on entry) is not carried and keeps its pre-loop value after
        # the loop.
        p = _mhc_softmax(l_res)
        m0 = p + sinkhorn_eps
        for step in range(0, COLUMN_STEPS - 1):
            t = COLUMN_STEPS - 1 - step
            a_prev = _mhc_col_normalize(m0, sinkhorn_eps)
            for _ in range(1, t):
                a_prev = _mhc_col_normalize(_mhc_row_normalize(a_prev, sinkhorn_eps), sinkhorn_eps)
            b_t = _mhc_row_normalize(a_prev, sinkhorn_eps)
            a_t = _mhc_col_normalize(b_t, sinkhorn_eps)
            g = libdevice.div_rn(g, tl.sum(b_t, axis=1)[:, None, :] + sinkhorn_eps)
            g = g - tl.sum(g * a_t, axis=1)[:, None, :]
            g = libdevice.div_rn(g, tl.sum(a_prev, axis=2)[:, :, None] + sinkhorn_eps)
            g = g - tl.sum(g * b_t, axis=2)[:, :, None]
        g = libdevice.div_rn(g, tl.sum(m0, axis=1)[:, None, :] + sinkhorn_eps)
        g = g - tl.sum(g * _mhc_col_normalize(m0, sinkhorn_eps), axis=1)[:, None, :]
        # softmax backward as torch: y * (g - sum(g * y)); the + eps does not change the gradient.
        gl_res = p * (g - tl.sum(g * p, axis=2)[:, :, None])

        # h = (r * proj) * alpha + bias: d(r*proj) = g * alpha, d(proj) = d(r*proj) * r.
        grp_pre = libdevice.mul_rn(gl_pre, a_pre)
        grp_post = libdevice.mul_rn(gl_post, a_post)
        grp_res = libdevice.mul_rn(gl_res, a_res)
        g_proj_row = g_proj_ptr + rows64 * NP
        tl.store(g_proj_row[:, None] + idx[None, :], libdevice.mul_rn(grp_pre, r[:, None]), mask=m2)
        tl.store(g_proj_row[:, None] + N + idx[None, :], libdevice.mul_rn(grp_post, r[:, None]), mask=m2)
        tl.store(g_proj_row[:, None, None] + res_off, libdevice.mul_rn(grp_res, r[:, None, None]), mask=m3)

        # dX from the norm: r = 1/D, D = ||X|| / sqrt(K) + eps, d||X||/dX = X / ||X|| (0 where ||X|| = 0, as torch).
        g_r = (
            tl.sum(grp_pre * p_pre, axis=1)
            + tl.sum(grp_post * p_post, axis=1)
            + tl.sum(tl.sum(grp_res * p_res, axis=2), axis=1)
        )
        g_nrm = libdevice.mul_rn(libdevice.mul_rn(g_r, -libdevice.mul_rn(r, r)), inv_sqrt_width)
        coef = tl.where(nrm > 0.0, libdevice.div_rn(g_nrm, nrm), 0.0)
        tl.store(coef_ptr + rows64, coef, mask=row_mask)

        # Per-program partial sums of the alpha (d alpha = sum g * r * proj) and bias gradients; the host sums
        # the programs in a fixed order.
        part = partial_ptr + pid.to(tl.int64) * (3 + NP)
        tl.store(part, tl.sum(tl.sum(gl_pre * rp_pre, axis=1), axis=0))
        tl.store(part + 1, tl.sum(tl.sum(gl_post * rp_post, axis=1), axis=0))
        tl.store(part + 2, tl.sum(tl.sum(tl.sum(gl_res * rp_res, axis=2), axis=1), axis=0))
        tl.store(part + 3 + idx, tl.sum(gl_pre, axis=0))
        tl.store(part + 3 + N + idx, tl.sum(gl_post, axis=0))
        tl.store(part + 3 + 2 * N + idx[:, None] * N + idx[None, :], tl.sum(gl_res, axis=0))

    @triton.jit
    def _mhc_norm_grad_add(lin_ptr, x_ptr, coef_ptr, out_ptr, width, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < width
        offs = row * width + cols
        lin = tl.load(lin_ptr + offs, mask=mask, other=0.0)
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        coef = tl.load(coef_ptr + row)
        tl.store(out_ptr + offs, (lin + libdevice.mul_rn(xv, coef)).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _mhc_agg_fwd(
        x_ptr, h_ptr, out_ptr, rows_total, channels, N: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr
    ):
        rows = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        row_mask = rows < rows_total
        rows64 = rows.to(tl.int64)
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < channels
        idx = tl.arange(0, N)
        h = tl.load(h_ptr + rows64[:, None] * N + idx[None, :], mask=row_mask[:, None], other=0.0).to(tl.float32)
        mask3 = row_mask[:, None, None] & col_mask[None, None, :]
        offs3 = rows64[:, None, None] * (N * channels) + idx[None, :, None] * channels + cols[None, None, :]
        xv = tl.load(x_ptr + offs3, mask=mask3, other=0.0).to(tl.float32)
        out = tl.sum(xv * h[:, :, None], axis=1)
        tl.store(
            out_ptr + rows64[:, None] * channels + cols[None, :],
            out,
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _mhc_agg_bwd(
        go_ptr,
        x_ptr,
        h_ptr,
        g_x_ptr,
        g_h_ptr,
        rows_total,
        channels,
        N: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        row_mask = rows < rows_total
        rows64 = rows.to(tl.int64)
        idx = tl.arange(0, N)
        h = tl.load(h_ptr + rows64[:, None] * N + idx[None, :], mask=row_mask[:, None], other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK_S, N), dtype=tl.float32)
        for c0 in range(0, channels, BLOCK_C):
            cols = c0 + tl.arange(0, BLOCK_C)
            col_mask = cols < channels
            mask2 = row_mask[:, None] & col_mask[None, :]
            go = tl.load(go_ptr + rows64[:, None] * channels + cols[None, :], mask=mask2, other=0.0).to(tl.float32)
            mask3 = row_mask[:, None, None] & col_mask[None, None, :]
            offs3 = rows64[:, None, None] * (N * channels) + idx[None, :, None] * channels + cols[None, None, :]
            xv = tl.load(x_ptr + offs3, mask=mask3, other=0.0).to(tl.float32)
            tl.store(g_x_ptr + offs3, (go[:, None, :] * h[:, :, None]).to(g_x_ptr.dtype.element_ty), mask=mask3)
            acc += tl.sum(go[:, None, :] * xv, axis=2)
        tl.store(g_h_ptr + rows64[:, None] * N + idx[None, :], acc, mask=row_mask[:, None])

    @triton.jit
    def _mhc_post_bda_fwd(
        h_res_ptr,
        orig_ptr,
        h_post_ptr,
        x_ptr,
        out_ptr,
        rows_total,
        channels,
        N: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        row_mask = rows < rows_total
        rows64 = rows.to(tl.int64)
        cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < channels
        idx = tl.arange(0, N)
        mask2 = row_mask[:, None] & col_mask[None, :]
        mask3 = row_mask[:, None, None] & col_mask[None, None, :]
        xv = tl.load(x_ptr + rows64[:, None] * channels + cols[None, :], mask=mask2, other=0.0).to(tl.float32)
        offs3 = rows64[:, None, None] * (N * channels) + idx[None, :, None] * channels + cols[None, None, :]
        orig = tl.load(orig_ptr + offs3, mask=mask3, other=0.0).to(tl.float32)  # [rows, j, cols]
        for i in tl.static_range(N):
            # Column i of H_res: out_i = sum_j H_res[j, i] x_j + h_post_i * y, summed in fp32 and rounded once.
            hr_col = tl.load(
                h_res_ptr + rows64[:, None] * (N * N) + idx[None, :] * N + i, mask=row_mask[:, None], other=0.0
            )
            mixed = tl.sum(hr_col[:, :, None] * orig, axis=1)
            hp = tl.load(h_post_ptr + rows64 * N + i, mask=row_mask, other=0.0)
            # Eager order: the rounded product h_post_i * y, then the mixed residual (mul_rn stops an FMA).
            out = libdevice.mul_rn(hp[:, None], xv) + mixed
            tl.store(
                out_ptr + rows64[:, None] * (N * channels) + i * channels + cols[None, :],
                out.to(out_ptr.dtype.element_ty),
                mask=mask2,
            )

    @triton.jit
    def _mhc_post_bda_bwd(
        go_ptr,
        h_res_ptr,
        h_post_ptr,
        x_ptr,
        orig_ptr,
        g_x_ptr,
        g_orig_ptr,
        g_h_post_ptr,
        g_h_res_ptr,
        rows_total,
        channels,
        N: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        row_mask = rows < rows_total
        rows64 = rows.to(tl.int64)
        idx = tl.arange(0, N)
        hp = tl.load(h_post_ptr + rows64[:, None] * N + idx[None, :], mask=row_mask[:, None], other=0.0)
        acc_hp = tl.zeros((BLOCK_S, N), dtype=tl.float32)
        acc_hr = tl.zeros((BLOCK_S, N, N), dtype=tl.float32)  # [rows, j, i], the layout of H_res
        for c0 in range(0, channels, BLOCK_C):
            cols = c0 + tl.arange(0, BLOCK_C)
            col_mask = cols < channels
            mask2 = row_mask[:, None] & col_mask[None, :]
            mask3 = row_mask[:, None, None] & col_mask[None, None, :]
            offs3 = rows64[:, None, None] * (N * channels) + idx[None, :, None] * channels + cols[None, None, :]
            go = tl.load(go_ptr + offs3, mask=mask3, other=0.0).to(tl.float32)  # [rows, i, cols]
            xv = tl.load(x_ptr + rows64[:, None] * channels + cols[None, :], mask=mask2, other=0.0).to(tl.float32)
            # d y = sum_i h_post_i * d out_i; d h_post_i = sum_c d out_i * y.
            g_x = tl.sum(hp[:, :, None] * go, axis=1)
            tl.store(
                g_x_ptr + rows64[:, None] * channels + cols[None, :], g_x.to(g_x_ptr.dtype.element_ty), mask=mask2
            )
            acc_hp += tl.sum(go * xv[:, None, :], axis=2)
            for j in tl.static_range(N):
                # d x_j = sum_i H_res[j, i] d out_i; d H_res[j, i] = sum_c x_j * d out_i.
                hr_row = tl.load(
                    h_res_ptr + rows64[:, None] * (N * N) + j * N + idx[None, :], mask=row_mask[:, None], other=0.0
                )
                g_orig = tl.sum(hr_row[:, :, None] * go, axis=1)
                orig_off = rows64[:, None] * (N * channels) + j * channels + cols[None, :]
                tl.store(g_orig_ptr + orig_off, g_orig.to(g_orig_ptr.dtype.element_ty), mask=mask2)
                orig_j = tl.load(orig_ptr + orig_off, mask=mask2, other=0.0).to(tl.float32)
                dots = tl.sum(orig_j[:, None, :] * go, axis=2)  # [rows, i]
                acc_hr += tl.where(idx[None, :, None] == j, dots[:, None, :], 0.0)
        tl.store(g_h_post_ptr + rows64[:, None] * N + idx[None, :], acc_hp, mask=row_mask[:, None])
        tl.store(
            g_h_res_ptr + rows64[:, None, None] * (N * N) + idx[None, :, None] * N + idx[None, None, :],
            acc_hr,
            mask=row_mask[:, None, None],
        )

    _KERNELS_LOADED = True


def fused_mhc_unsupported_reason(
    streams: Tensor,
    n: int,
    *,
    fp32_operands: tuple[Tensor, ...] = (),
    layer_output: Tensor | None = None,
    bias: Tensor | None = None,
) -> str | None:
    """Why the fused kernels cannot take these operands, or ``None`` if they can.

    Args:
        streams: The residual streams (or any stream-shaped operand): CUDA, fp32 or bf16.
        n: Number of residual streams: a power of two from 2 to :data:`MAX_STREAMS`.
        fp32_operands: Operands that must be fp32 CUDA tensors (the mapping weight, ``alpha_*`` and ``bias``
            parameters, or ``h_pre``, ``h_post`` and ``H_res``).
        layer_output: The layer output of the residual update: CUDA, in :data:`LAYER_OUTPUT_DTYPES`.
        bias: The layer output bias of the residual update; the kernels take only ``None``.

    Returns:
        A short reason, or ``None`` when the kernels apply.
    """
    if not streams.is_cuda:
        return "the streams are not on a CUDA device"
    if streams.dtype not in STREAM_DTYPES:
        return f"stream dtype {streams.dtype} (the kernels take fp32 or bf16 streams)"
    if not (2 <= n <= MAX_STREAMS and n & (n - 1) == 0):
        return f"{n} residual streams (the kernels take a power of two from 2 to {MAX_STREAMS})"
    if bias is not None:
        return "a layer output bias"
    for operand in fp32_operands:
        if not operand.is_cuda or operand.dtype != torch.float32:
            return f"a {operand.dtype} operand on {operand.device} (the mappings and their parameters must be fp32)"
    if layer_output is not None and (not layer_output.is_cuda or layer_output.dtype not in LAYER_OUTPUT_DTYPES):
        return f"layer output dtype {layer_output.dtype}"
    from megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch import triton_available

    if not triton_available():
        return "Triton does not import"
    return None


def fused_mhc_supports(
    streams: Tensor,
    n: int,
    *,
    fp32_operands: tuple[Tensor, ...] = (),
    layer_output: Tensor | None = None,
    bias: Tensor | None = None,
) -> bool:
    """Whether the fused kernels take these operands (:func:`fused_mhc_unsupported_reason` is ``None``).

    Args:
        streams: See :func:`fused_mhc_unsupported_reason`.
        n: See :func:`fused_mhc_unsupported_reason`.
        fp32_operands: See :func:`fused_mhc_unsupported_reason`.
        layer_output: See :func:`fused_mhc_unsupported_reason`.
        bias: See :func:`fused_mhc_unsupported_reason`.

    Returns:
        ``True`` if the kernels apply.
    """
    reason = fused_mhc_unsupported_reason(
        streams, n, fp32_operands=fp32_operands, layer_output=layer_output, bias=bias
    )
    return reason is None


def _fp32_reciprocal(value: float) -> float:
    """``1 / value`` rounded as torch rounds the reciprocal of a Python scalar divisor (in fp32)."""
    return torch.tensor(value, dtype=torch.float32).reciprocal().item()


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class _FusedMhcMappings(torch.autograd.Function):
    """``(h_pre, h_post, H_res)`` of flattened streams ``[M, K]``; see :func:`fused_mhc_mappings`."""

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
        norm_eps,
        compute_h_eps,
        sinkhorn_iterations,
        sinkhorn_eps,
    ):
        rows, width = x.shape
        num_proj = n * n + 2 * n
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.contiguous().float()
            # fp32(X) W^T with the transposed view of W as the GEMM operand. The eager path multiplies by a
            # contiguous copy of W^T, which costs one copy kernel per call; caching that copy is unsafe (optimizer
            # updates through ``param.data`` do not bump the version counter), so the fused path skips it. cuBLAS
            # may then pick a different kernel and summation order: the projection agrees with the eager one to
            # fp32 rounding (same TF32 setting), not bitwise.
            proj = torch.matmul(x_fp32, weight.float().t())
        h_pre = torch.empty(rows, n, dtype=torch.float32, device=x.device)
        h_post = torch.empty(rows, n, dtype=torch.float32, device=x.device)
        h_res = torch.empty(rows, n, n, dtype=torch.float32, device=x.device)
        nrm = torch.empty(rows, dtype=torch.float32, device=x.device)
        inv_sqrt_width = _fp32_reciprocal(width**0.5)
        column_steps = max(int(sinkhorn_iterations), 1)
        if rows:
            _load_kernels()
            _mhc_map_fwd[(_cdiv(rows, _MAP_FWD_BLOCK_M),)](
                x_fp32,
                proj,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias,
                h_pre,
                h_post,
                h_res,
                nrm,
                rows,
                width,
                inv_sqrt_width,
                float(norm_eps),
                float(compute_h_eps),
                float(sinkhorn_eps),
                N=n,
                NP=num_proj,
                COLUMN_STEPS=column_steps,
                BLOCK_M=_MAP_FWD_BLOCK_M,
                BLOCK_K=_MAP_FWD_BLOCK_K,
                num_warps=_MAP_FWD_WARPS,
            )
        # Saved unconditionally: the forward does not depend on the grad mode (full-recompute replays).
        ctx.save_for_backward(x_fp32, weight, proj, nrm, alpha_pre, alpha_post, alpha_res, bias)
        ctx.meta = (x.dtype, n, inv_sqrt_width, float(norm_eps), float(sinkhorn_eps), column_steps)
        return h_pre, h_post, h_res

    @staticmethod
    def backward(ctx, g_pre, g_post, g_res):
        x_fp32, weight, proj, nrm, alpha_pre, alpha_post, alpha_res, bias = ctx.saved_tensors
        x_dtype, n, inv_sqrt_width, norm_eps, sinkhorn_eps, column_steps = ctx.meta
        rows, width = x_fp32.shape
        num_proj = n * n + 2 * n
        programs = _cdiv(rows, _MAP_BWD_BLOCK_M)
        device = x_fp32.device
        g_proj = torch.empty(rows, num_proj, dtype=torch.float32, device=device)
        coef = torch.empty(rows, dtype=torch.float32, device=device)
        partials = torch.empty(programs, 3 + num_proj, dtype=torch.float32, device=device)
        if rows:
            _load_kernels()
            _mhc_map_bwd[(programs,)](
                g_pre.float().contiguous(),
                g_post.float().contiguous(),
                g_res.float().contiguous(),
                proj,
                nrm,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias,
                g_proj,
                coef,
                partials,
                rows,
                inv_sqrt_width,
                norm_eps,
                sinkhorn_eps,
                N=n,
                NP=num_proj,
                COLUMN_STEPS=column_steps,
                BLOCK_M=_MAP_BWD_BLOCK_M,
                num_warps=_MAP_BWD_WARPS,
            )
        sums = partials.sum(0)
        with torch.autocast(device_type=device.type, enabled=False):
            g_lin = torch.matmul(g_proj, weight.float())
            g_weight = torch.matmul(g_proj.t(), x_fp32)
        g_x = g_lin if x_dtype == torch.float32 else torch.empty(rows, width, dtype=x_dtype, device=device)
        if rows:
            _mhc_norm_grad_add[(rows, _cdiv(width, _NORM_GRAD_BLOCK))](
                g_lin, x_fp32, coef, g_x, width, BLOCK=_NORM_GRAD_BLOCK, num_warps=_NORM_GRAD_WARPS
            )
        return (
            g_x,
            g_weight.to(weight.dtype),
            sums[0:1].view_as(alpha_pre).to(alpha_pre.dtype),
            sums[1:2].view_as(alpha_post).to(alpha_post.dtype),
            sums[2:3].view_as(alpha_res).to(alpha_res.dtype),
            sums[3:].view_as(bias).to(bias.dtype),
            None,
            None,
            None,
            None,
            None,
        )


class _FusedMhcAggregate(torch.autograd.Function):
    """``sum_j h_j x_j`` of ``[M, n, C]`` streams in fp32; see :func:`fused_mhc_aggregate`."""

    @staticmethod
    def forward(ctx, streams, h_pre):
        rows, n, channels = streams.shape
        out = torch.empty(rows, channels, dtype=torch.float32, device=streams.device)
        if out.numel():
            _load_kernels()
            grid = (_cdiv(rows, _ROW_FWD_BLOCK_S), _cdiv(channels, _ROW_FWD_BLOCK_C))
            _mhc_agg_fwd[grid](
                streams,
                h_pre,
                out,
                rows,
                channels,
                N=n,
                BLOCK_S=_ROW_FWD_BLOCK_S,
                BLOCK_C=_ROW_FWD_BLOCK_C,
                num_warps=_ROW_FWD_WARPS,
            )
        ctx.save_for_backward(streams, h_pre)
        return out

    @staticmethod
    def backward(ctx, grad):
        streams, h_pre = ctx.saved_tensors
        rows, n, channels = streams.shape
        g_streams = torch.empty_like(streams)
        g_h = torch.empty(rows, n, dtype=h_pre.dtype, device=h_pre.device)
        if rows:
            _load_kernels()
            _mhc_agg_bwd[(_cdiv(rows, _ROW_BWD_BLOCK_S),)](
                grad.contiguous(),
                streams,
                h_pre,
                g_streams,
                g_h,
                rows,
                channels,
                N=n,
                BLOCK_S=_ROW_BWD_BLOCK_S,
                BLOCK_C=_ROW_BWD_BLOCK_C,
                num_warps=_ROW_BWD_WARPS,
            )
        return g_streams, g_h


class _FusedMhcPostBda(torch.autograd.Function):
    """``H_res^T @ streams + h_post * y`` of ``[M, n, C]`` streams; see :func:`fused_mhc_post_bda`."""

    @staticmethod
    def forward(ctx, h_res, streams, h_post, x):
        rows, n, channels = streams.shape
        out = torch.empty_like(streams)
        if out.numel():
            _load_kernels()
            grid = (_cdiv(rows, _ROW_FWD_BLOCK_S), _cdiv(channels, _ROW_FWD_BLOCK_C))
            _mhc_post_bda_fwd[grid](
                h_res,
                streams,
                h_post,
                x,
                out,
                rows,
                channels,
                N=n,
                BLOCK_S=_ROW_FWD_BLOCK_S,
                BLOCK_C=_ROW_FWD_BLOCK_C,
                num_warps=_ROW_FWD_WARPS,
            )
        ctx.save_for_backward(h_res, streams, h_post, x)
        return out

    @staticmethod
    def backward(ctx, grad):
        h_res, streams, h_post, x = ctx.saved_tensors
        rows, n, channels = streams.shape
        g_h_res = torch.empty_like(h_res)
        g_streams = torch.empty_like(streams)
        g_h_post = torch.empty_like(h_post)
        g_x = torch.empty_like(x)
        if rows:
            _load_kernels()
            _mhc_post_bda_bwd[(_cdiv(rows, _ROW_BWD_BLOCK_S),)](
                grad.contiguous(),
                h_res,
                h_post,
                x,
                streams,
                g_x,
                g_streams,
                g_h_post,
                g_h_res,
                rows,
                channels,
                N=n,
                BLOCK_S=_ROW_BWD_BLOCK_S,
                BLOCK_C=_ROW_BWD_BLOCK_C,
                num_warps=_ROW_BWD_WARPS,
            )
        return g_h_res, g_streams, g_h_post, g_x


def fused_mhc_mappings(
    x: Tensor,
    weight: Tensor,
    alpha_pre: Tensor,
    alpha_post: Tensor,
    alpha_res: Tensor,
    bias: Tensor,
    n: int,
    norm_eps: float,
    compute_h_eps: float,
    sinkhorn_iterations: int,
    sinkhorn_eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """The ERNIE 5.5 mHC mappings of flattened streams, in fp32, with a fused epilogue and backward.

    Args:
        x: ``[M, K]`` CUDA streams (``K = n*C``), fp32 or bf16.
        weight: ``mapping_proj.weight``, fp32 ``[n^2 + 2n, K]``.
        alpha_pre: fp32 ``[1]`` scale of the ``h_pre`` logits.
        alpha_post: fp32 ``[1]`` scale of the ``h_post`` logits.
        alpha_res: fp32 ``[1]`` scale of the ``H_res`` logits.
        bias: fp32 ``[n^2 + 2n]`` logit bias.
        n: Number of residual streams.
        norm_eps: Epsilon of ``r``, added outside the square root.
        compute_h_eps: Epsilon added to ``h_pre``.
        sinkhorn_iterations: Number of Sinkhorn column normalizations.
        sinkhorn_eps: Sinkhorn epsilon (after the softmax and in every normalizer).

    Returns:
        ``(h_pre [M, n], h_post [M, n], H_res [M, n, n])``, all fp32 and contiguous.
    """
    return _FusedMhcMappings.apply(
        x,
        weight,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        int(n),
        norm_eps,
        compute_h_eps,
        sinkhorn_iterations,
        sinkhorn_eps,
    )


def fused_mhc_aggregate(streams: Tensor, h_pre: Tensor) -> Tensor:
    """``sum_j h_pre_j x_j`` over the stream axis, computed and returned in fp32.

    Args:
        streams: ``[..., n, C]`` CUDA streams, fp32 or bf16.
        h_pre: ``[..., n]`` fp32 aggregation weights.

    Returns:
        ``[..., C]`` in fp32.
    """
    *lead, n, channels = streams.shape
    out = _FusedMhcAggregate.apply(streams.reshape(-1, n, channels).contiguous(), h_pre.reshape(-1, n).contiguous())
    return out.view(*lead, channels)


def fused_mhc_post_bda(h_res: Tensor, original_residual: Tensor, h_post: Tensor, x: Tensor) -> Tensor:
    """``x_i <- sum_j H_res[j, i] x_j + h_post_i * y`` in fp32, rounded once to the stream dtype (no bias).

    Args:
        h_res: ``[..., n, n]`` fp32 residual mixing matrices.
        original_residual: ``[..., n, C]`` CUDA streams, fp32 or bf16.
        h_post: ``[..., n]`` fp32 expansion weights.
        x: ``[..., C]`` layer output ``y``.

    Returns:
        A contiguous ``[..., n, C]`` tensor in ``original_residual.dtype``.
    """
    *lead, n, channels = original_residual.shape
    out = _FusedMhcPostBda.apply(
        h_res.reshape(-1, n, n).contiguous(),
        original_residual.reshape(-1, n, channels).contiguous(),
        h_post.reshape(-1, n).contiguous(),
        x.reshape(-1, channels).contiguous(),
    )
    return out.view(*lead, n, channels)
