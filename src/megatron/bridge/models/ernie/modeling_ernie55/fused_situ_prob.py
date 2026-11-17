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

"""The routed experts' SiTU-GLU and routing-probability multiply in one Triton kernel per direction.

With ``bias_activation_fusion=False`` (every ERNIE 5.5 recipe, as the reference), Megatron-Core's ``TEGroupedMLP``
computes the routed experts' activation in three passes: SiTU-GLU (Bridge's Triton kernel from
:mod:`~megatron.bridge.models.ernie.modeling_ernie55.fused_ops` with ``ernie_fused_ops``), the product with the fp32
routing probability (an fp32 tensor) and its cast back to the activation dtype. Autograd adds five passes to the
SiTU-GLU backward: the cast's backward, the two products of the multiply's backward, the cast of the activation
gradient and the row sum of the probability gradient. :func:`fused_situ_glu_prob` computes the same values in one
kernel per direction; Bridge runs it for the routed experts behind ``ernie_fused_situ_prob`` (see
:class:`~megatron.bridge.models.ernie.modeling_ernie55.moe.Ernie55GroupedMLP`).

Arithmetic. With ``x = [g | u]`` (the fc1 output, rounded to its dtype), the fp32 probability ``p`` of the row, ``a``
the activation and ``round`` the round-to-nearest-even cast to the activation dtype:

* Forward: ``a = round(act(x))`` with ``act`` computed in fp32 in the operation order of Bridge's SiTU-GLU kernel (the
  first rounding), then ``y = round(fl32(a * p))`` (the second rounding), Megatron-Core's ``(a * p).to(a.dtype)``.
* Backward from ``dy``: autograd hands the SiTU-GLU backward ``da = round(fl32(dy * p))`` (the multiply's gradient,
  cast to the dtype of ``a``), and the kernel runs Bridge's SiTU-GLU backward on it operation for operation, so
  ``dx`` is the unfused path's. ``dp = sum_j fl32(dy_j * a_j)``: autograd's ``grad * self`` with every product
  rounded to fp32, summed over the row in fp32 (``a`` is recomputed from ``x``, bitwise the forward's).

The forward output and ``dx`` are therefore bitwise equal to the unfused path's, which the GPU tests check.
``dp`` differs only in its summation order: autograd sums the row with its reduction kernel (``sum_to``), the kernel
adds each lane's columns in order and then the lanes as a tree. That order is fixed, so the kernel is deterministic.

Subnormals. Triton links libdevice with ``nvvm-reflect-ftz`` (its default, ``enable_reflect_ftz=True``), so libdevice
functions, ``mul_rn`` included, flush subnormal inputs and results to zero, while PyTorch's CUDA kernels keep them.
The SiTU-GLU expressions call libdevice as ``fused_ops`` does, so both kernels flush the same values. The three
products that stand in for PyTorch's multiplies (``a * p`` forward, ``dy * p`` and ``dy * a`` backward) use
``_mul_rn_ieee``, a ``mul.rn.f32`` without ``.ftz``: round to nearest even with IEEE subnormals, as PyTorch computes
them, and never contracted into an FMA. With ``libdevice.mul_rn`` there, a subnormal ``a * p``, a bf16-subnormal
``a`` or a subnormal ``dy * p`` that rounds up to the smallest normal bf16 would become zeros that the unfused path
does not have.

Kernels. The forward runs one program per row and block of 1024 columns, like Bridge's SiTU-GLU kernel. The
backward runs one program per row, which holds the row's probability gradient. Backward saves the input and the
probabilities and recomputes the activation; the unfused path also keeps the rounded activation for the multiply's
backward.

Importing this module imports only PyTorch. :func:`_load_kernels` imports Triton and defines the kernels, as module
globals, on the first launch, as :mod:`~megatron.bridge.models.ernie.modeling_ernie55.fused_ops` does.
"""

from __future__ import annotations

import torch


__all__ = ["fused_situ_glu_prob"]

# Bound by _load_kernels() on the first launch. The kernels are module globals so that a kernel finds the helper
# kernel it calls: Triton resolves the names a kernel uses in the kernel's globals.
triton = None
tl = None
libdevice = None
_sigmoid_precise = None
_mul_rn_ieee = None
_situ_glu_prob_fwd_kernel = None
_situ_glu_prob_bwd_kernel = None
_KERNELS_LOADED = False


def _load_kernels() -> None:
    """Import Triton and define the kernels as module globals (once per process)."""
    global triton, tl, libdevice, _KERNELS_LOADED
    global _sigmoid_precise, _mul_rn_ieee, _situ_glu_prob_fwd_kernel, _situ_glu_prob_bwd_kernel
    if _KERNELS_LOADED:
        return
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    # The SiTU-GLU arithmetic below repeats fused_ops' kernels expression for expression, libdevice calls included:
    # the bitwise equality with the unfused path rests on it.

    @triton.jit
    def _sigmoid_precise(x):
        # fused_ops._sigmoid_precise: torch's float sigmoid, 1 / (1 + expf(-x)) with an IEEE divide.
        return libdevice.div_rn(1.0, 1.0 + libdevice.exp(-x))

    @triton.jit
    def _mul_rn_ieee(a, b):
        # PyTorch's fp32 multiply: round to nearest even, subnormal inputs and results kept. libdevice.mul_rn is
        # mul.rn.ftz.f32 under Triton's nvvm-reflect-ftz. The .rn also keeps the product out of an FMA.
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _situ_glu_prob_fwd_kernel(
        x_ptr,
        prob_ptr,
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
        # fused_ops._situ_glu_fwd_kernel.
        gate_act = (beta * libdevice.tanh(gate * inv_beta)) * _sigmoid_precise(gate)
        if has_linear_tanh:
            up = linear_beta * libdevice.tanh(up * inv_linear_beta)
        # First rounding: the activation in its dtype, the value the SiTU-GLU kernel stores.
        act = (gate_act * up).to(out_ptr.dtype.element_ty).to(tl.float32)
        # Second rounding: Megatron-Core's (act * prob).to(act.dtype), an fp32 product cast to the activation dtype.
        prob = tl.load(prob_ptr + row)
        tl.store(out_ptr + row * n_cols + cols, _mul_rn_ieee(act, prob), mask=mask)

    @triton.jit
    def _situ_glu_prob_bwd_kernel(
        x_ptr,
        prob_ptr,
        out_grad_ptr,
        x_grad_ptr,
        prob_grad_ptr,
        n_cols: tl.constexpr,
        beta: tl.constexpr,
        inv_beta: tl.constexpr,
        linear_beta: tl.constexpr,
        inv_linear_beta: tl.constexpr,
        has_linear_tanh: tl.constexpr,
        has_prob_grad: tl.constexpr,
        block_n: tl.constexpr,
    ):
        # One program per row: the row's probability gradient is a sum over its columns.
        row = tl.program_id(0).to(tl.int64)
        base = row * (2 * n_cols)
        prob = tl.load(prob_ptr + row)
        prob_grad = tl.zeros([block_n], dtype=tl.float32)
        for start in range(0, n_cols, block_n):
            cols = start + tl.arange(0, block_n)
            mask = cols < n_cols
            gate = tl.load(x_ptr + base + cols, mask=mask).to(tl.float32)
            up = tl.load(x_ptr + base + n_cols + cols, mask=mask).to(tl.float32)
            out_grad = tl.load(out_grad_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
            # Autograd's backward of (act * prob).to(act.dtype) for act: the fp32 product grad * prob, which the
            # engine casts to the activation dtype before the SiTU-GLU backward reads it.
            act_grad = _mul_rn_ieee(out_grad, prob).to(x_grad_ptr.dtype.element_ty).to(tl.float32)
            # fused_ops._situ_glu_bwd_kernel on act_grad.
            gate_tanh = libdevice.tanh(gate * inv_beta)
            gate_scaled = beta * gate_tanh
            gate_sigmoid = _sigmoid_precise(gate)
            gate_act = gate_scaled * gate_sigmoid
            if has_linear_tanh:
                up_tanh = libdevice.tanh(up * inv_linear_beta)
                up_act = linear_beta * up_tanh
            else:
                up_act = up
            gate_act_grad = libdevice.mul_rn(act_grad, up_act)
            up_act_grad = libdevice.mul_rn(act_grad, gate_act)
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
            if has_prob_grad:
                # Autograd's grad * self for the probability: the output gradient times the rounded activation of
                # the forward, each product rounded to fp32. Masked lanes add nothing.
                act = (gate_act * up_act).to(x_grad_ptr.dtype.element_ty).to(tl.float32)
                prob_grad += tl.where(mask, _mul_rn_ieee(out_grad, act), 0.0)
        if has_prob_grad:
            tl.store(prob_grad_ptr + row, tl.sum(prob_grad, axis=0))

    _KERNELS_LOADED = True


def _launch_config(x: torch.Tensor, beta: float, linear_beta: float | None) -> tuple[int, int, dict]:
    """Rows, column blocks and the constexpr arguments of both kernels; those of fused_ops' SiTU-GLU launcher."""
    n_cols = x.shape[-1] // 2
    block_n = min(1024, triton.next_power_of_2(n_cols))
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
    return x.numel() // (2 * n_cols), triton.cdiv(n_cols, block_n), kwargs


class _FusedSituGluProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probs, beta, linear_beta):
        out = x.new_empty(*x.shape[:-1], x.shape[-1] // 2)
        if out.numel():
            _load_kernels()
            rows, col_blocks, kwargs = _launch_config(x, beta, linear_beta)
            _situ_glu_prob_fwd_kernel[(rows, col_blocks)](x, probs, out, **kwargs)
        ctx.save_for_backward(x, probs)
        ctx.beta, ctx.linear_beta = beta, linear_beta
        return out

    @staticmethod
    def backward(ctx, out_grad):
        x, probs = ctx.saved_tensors
        x_grad = torch.empty_like(x)
        has_prob_grad = ctx.needs_input_grad[1]
        prob_grad = None
        if has_prob_grad:
            # Rows without columns have a zero gradient; every other row is written by its program.
            prob_grad = torch.empty_like(probs) if x.numel() else torch.zeros_like(probs)
        if x.numel():
            _load_kernels()
            rows, _, kwargs = _launch_config(x, ctx.beta, ctx.linear_beta)
            _situ_glu_prob_bwd_kernel[(rows,)](
                x,
                probs,
                out_grad.contiguous(),
                x_grad,
                probs if prob_grad is None else prob_grad,
                has_prob_grad=has_prob_grad,
                **kwargs,
            )
        return x_grad, prob_grad, None, None


def fused_situ_glu_prob(
    x: torch.Tensor, probs: torch.Tensor, beta: float, linear_beta: float | None = None
) -> torch.Tensor:
    """``round(fp32(round(situ_glu(x))) * probs)``: SiTU-GLU of ``x = [g, u]`` times one fp32 probability per row.

    Equal to Megatron-Core's routed-expert activation ``(situ_glu(x) * probs).to(x.dtype)`` with Bridge's SiTU-GLU
    kernel as ``situ_glu``, forward and input gradient bitwise; the probability gradient sums each row in its own
    order (see the module docstring).

    Args:
        x: Contiguous CUDA tensor ``[..., 2 * n]`` in bf16, fp16 or fp32: the gate half, then the up half.
        probs: Contiguous fp32 tensor on the device of ``x`` with one probability per row of ``x``
            (``x.numel() // x.shape[-1]`` elements, e.g. ``[rows]`` or ``[rows, 1]``).
        beta: Scale of the gate's tanh (``activation_func_tanh_clamp_scale``).
        linear_beta: Scale of the up half's tanh (``activation_func_tanh_clamp_scale_linear``); ``None`` leaves the
            up half linear.

    Returns:
        ``[..., n]`` in the dtype of ``x``. Its backward returns the gradient of ``probs`` in their shape and fp32.
    """
    return _FusedSituGluProb.apply(x, probs, beta, linear_beta)
