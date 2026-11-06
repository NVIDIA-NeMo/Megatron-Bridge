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

"""ERNIE 5.5 manifold-constrained hyper-connections (mHC): the eager fp32 mapping and mixing contract.

Megatron-Core's ``HyperConnectionHybridLayer`` wraps every hybrid layer in a
``HyperConnectionModule``. ERNIE 5.5 keeps that structure and Megatron-Core's parameters,
initialization and autograd plumbing, and changes only the arithmetic of four operations. For the
streams ``X = [x_1 .. x_n]`` of one token (stored as ``[s, b, n*C]``, ``K = n*C``):

* ``proj = fp32(X) W^T`` with ``W = mapping_proj.weight`` (``[n^2 + 2n, K]``, fp32), and
  ``r = 1 / (||fp32(X)||_2 / sqrt(K) + 1e-6)`` (the epsilon outside the square root);
* ``h = r * proj * [a_pre]*n ++ [a_post]*n ++ [a_res]*n^2 + bias``;
  ``h_pre = sigmoid(h[:n]) + 1e-6``, ``h_post = 2 sigmoid(h[n:2n])``,
  ``H_res = Sinkhorn(h[2n:].view(n, n))`` with ``M = softmax(L, -1) + 1e-6``, a column
  normalization ``M / (colsum + 1e-6)``, then ``iterations - 1`` rounds of a row and a column
  normalization, plain autograd throughout;
* the layer input ``a = sum_j h_pre_j x_j``, in fp32 (the downcasting pre-norm rounds it once);
* the residual update ``x_i <- sum_j H_res[j, i] x_j + h_post_i * y`` for the layer output ``y``,
  in fp32 and rounded once to the stream dtype.

All four keep ``h_pre``, ``h_post`` and ``H_res`` in fp32 and run eagerly, as the reference
implementation does. Megatron-Core by default casts the mappings to the stream dtype, and it compiles
its mixing operations with ``torch.compile``. Two consequences make that unsuitable for ERNIE 5.5:

* Precision. The decoder's streams are fp32 (see
  :class:`~megatron.bridge.models.ernie.modeling_ernie55.hybrid_stack.Ernie55HybridStack`), so the
  cast is a no-op there, but the MTP layer's streams are bf16: Megatron-Core rounds its mappings to
  bf16 and mixes in bf16, where ERNIE 5.5 mixes in fp32 and rounds once. Megatron-Core's
  ``mhc_keep_mappings_in_fp32`` (off by default) keeps the mappings in fp32, but its mixing
  operations stay compiled (see the next point).
* Recompute consistency. Under full activation recompute the first forward runs under
  ``torch.no_grad()`` and the backward replays it with gradients enabled. Compiled code may
  specialize the two differently and change reduction order (Megatron-Core's own projection is eager
  for this reason; its other mHC operations are still compiled). A replay that differs from the forward
  changes the backward point, and through the MoE router top-k after the mHC aggregate, possibly
  the routing and the expert-token counts of the replay. Eager operations compute the same bits in
  both passes; the GPU test ``test_ernie55_stack_gpu.py`` checks this.

The fp32 GEMMs (the projection and the ``H_res`` mix) follow PyTorch's TF32 setting, as the
reference implementation's do, so parity runs must pin it.

Only the arithmetic changes, so parameter names, shapes, dtypes and initialization stay
Megatron-Core's. :func:`use_ernie55_hyper_connections` switches existing Megatron-Core modules to
:class:`Ernie55HyperConnectionModule` in place; the ERNIE decoder stack and model call it.
"""

from __future__ import annotations

import math

import torch
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor


__all__ = [
    "Ernie55HyperConnectionModule",
    "ernie55_h_aggregate",
    "ernie55_h_post_bda",
    "ernie55_sinkhorn",
    "use_ernie55_hyper_connections",
    "validate_ernie55_mhc_config",
]


def ernie55_sinkhorn(logits: Tensor, iterations: int, eps: float) -> Tensor:
    """Sinkhorn-Knopp projection of ``[..., n, n]`` logits onto doubly stochastic matrices.

    ``M = softmax(logits, -1) + eps``, then a column normalization, then ``iterations - 1`` rounds
    of a row and a column normalization, each dividing by ``sum + eps``. Plain autograd.

    Args:
        logits: Residual-mixing logits, ``[..., n, n]``.
        iterations: Total number of column normalizations (``mhc_sinkhorn_iterations``).
        eps: Epsilon added after the softmax and to every normalizer.

    Returns:
        The projected matrices, same shape and dtype as ``logits``.
    """
    matrix = logits.softmax(dim=-1) + eps
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


def ernie55_h_aggregate(streams: Tensor, h_pre: Tensor) -> Tensor:
    """``sum_j h_pre_j x_j`` over the stream axis, in the promoted dtype (fp32 for fp32 ``h_pre``).

    Args:
        streams: ``[s, b, n, C]``.
        h_pre: ``[s, b, n]``.

    Returns:
        ``[s, b, C]``.
    """
    with torch.autocast(device_type=streams.device.type, enabled=False):
        return (streams * h_pre.unsqueeze(-1)).sum(dim=-2)


def ernie55_h_post_bda(
    h_res: Tensor, original_residual: Tensor, h_post: Tensor, x: Tensor, bias: Tensor | None
) -> Tensor:
    """``H_res^T @ residual + h_post * (x [+ bias])`` in fp32, rounded once to the residual dtype.

    Args:
        h_res: ``[s, b, n, n]`` fp32 residual mixing matrices.
        original_residual: ``[s, b, n, C]`` streams before the update.
        h_post: ``[s, b, n]`` fp32 expansion weights.
        x: ``[s, b, C]`` layer output.
        bias: Optional ``[C]`` layer output bias.

    Returns:
        ``[s, b, n, C]`` in ``original_residual.dtype``.
    """
    s, b, n, channels = original_residual.shape
    residual = original_residual.float()
    with torch.autocast(device_type=residual.device.type, enabled=False):
        mixed = torch.bmm(h_res.reshape(s * b, n, n).transpose(1, 2), residual.reshape(s * b, n, channels))
        mixed = mixed.view(s, b, n, channels)
        output = h_post.unsqueeze(-1) * x.float().unsqueeze(2)
        if bias is not None:
            output = output + h_post.unsqueeze(-1) * bias.float().view(1, 1, 1, channels)
        output = output + mixed
    return output.to(original_residual.dtype)


class Ernie55HyperConnectionModule(HyperConnectionModule):
    """Megatron-Core's ``HyperConnectionModule`` with the ERNIE 5.5 eager fp32 mapping and mixing.

    The ERNIE stack converts the modules Megatron-Core built with
    :func:`use_ernie55_hyper_connections`, so their parameters, initialization and checkpoint keys
    are unchanged; constructing one directly gives the same module. ``compute_mappings`` returns
    fp32 ``h_pre``, ``h_post`` and ``H_res``; ``aggregate`` returns the fp32 layer input; the
    residual update (``_h_post_bda_op``, which Megatron-Core's forward calls) mixes in fp32 and
    rounds once to the stream dtype. Both ways of getting an instance point Megatron-Core's
    operation slots at the ERNIE operations; without that, Megatron-Core's compiled residual update
    would run. The fused-kernel path and the dropout path (``apply_h_res``) are not used:
    :func:`validate_ernie55_mhc_config` requires ``use_fused_mhc=False`` and ``hidden_dropout=0``.
    """

    def __init__(self, config: TransformerConfig, layer_number: int) -> None:
        super().__init__(config, layer_number)
        self._use_ernie55_operations()

    def _use_ernie55_operations(self) -> None:
        """Point the operation slots Megatron-Core's forward calls at the ERNIE 5.5 operations."""
        self._sinkhorn_op = ernie55_sinkhorn
        self._h_aggregate_op = ernie55_h_aggregate
        self._h_post_bda_op = ernie55_h_post_bda
        self._proj_rms_compute_h_op = None

    def compute_mappings(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute ``h_pre`` and ``h_post`` (``[s, b, n]``) and ``H_res`` (``[s, b, n, n]``), all fp32.

        Args:
            x: ``[s, b, n*C]`` streams (fp32 in the decoder, bf16 in the MTP layer).

        Returns:
            ``(h_pre, h_post, h_res)`` in fp32.
        """
        s, b, width = x.shape
        n = self.n
        x_fp32 = x.float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            # The projection multiplies by a contiguous copy of W^T: the operand layout of the
            # reference implementation, which selects the GEMM kernel and with it the rounding.
            proj = torch.matmul(x_fp32, self.mapping_proj.weight.float().T.contiguous())
            r = 1.0 / (x_fp32.norm(dim=-1, keepdim=True) / math.sqrt(width) + self.norm_eps)
            alpha = torch.cat(
                [self.alpha_pre.expand(n), self.alpha_post.expand(n), self.alpha_res.expand(n * n)]
            ).float()
            h = r * proj * alpha + self.bias.float()
            h_pre = h[..., :n].sigmoid() + self.compute_h_eps
            h_post = h[..., n : 2 * n].sigmoid() * 2
            h_res = ernie55_sinkhorn(h[..., 2 * n :].reshape(s, b, n, n), self.sinkhorn_iterations, self.sinkhorn_eps)
        return h_pre, h_post, h_res

    def aggregate(self, x: Tensor, h_pre: Tensor, out: Tensor | None = None) -> Tensor:
        """Aggregate the streams into the fp32 layer input ``sum_j h_pre_j x_j``.

        Args:
            x: ``[s, b, n*C]`` streams.
            h_pre: ``[s, b, n]`` fp32 aggregation weights.
            out: Megatron-Core dev's CUDA-graph arena slot; not supported.

        Returns:
            ``[s, b, C]`` in fp32.

        Raises:
            NotImplementedError: If ``out`` is given.
        """
        if out is not None:
            raise NotImplementedError(
                "ERNIE 5.5 mHC does not support caller-owned aggregate outputs (mHC recompute with "
                "attention CUDA graphs); its layer input is fp32 by contract."
            )
        s, b, _ = x.shape
        return ernie55_h_aggregate(x.view(s, b, self.n, self.hidden_size), h_pre)


def use_ernie55_hyper_connections(root: torch.nn.Module) -> int:
    """Convert every Megatron-Core ``HyperConnectionModule`` under ``root`` to the ERNIE 5.5 one (in place).

    Idempotent: modules that are already :class:`Ernie55HyperConnectionModule` are left alone.

    Args:
        root: Any module, e.g. a decoder stack or a whole model.

    Returns:
        The number of hyper-connection modules under ``root`` after the conversion.

    Raises:
        TypeError: If a hyper-connection module is a subclass this function does not know.
    """
    count = 0
    for module in root.modules():
        if not isinstance(module, HyperConnectionModule):
            continue
        if type(module) is HyperConnectionModule:
            module.__class__ = Ernie55HyperConnectionModule
            module._use_ernie55_operations()
        elif type(module) is not Ernie55HyperConnectionModule:
            raise TypeError(
                f"ERNIE 5.5 cannot convert hyper-connection module {type(module).__name__}; it converts "
                "Megatron-Core's HyperConnectionModule only."
            )
        count += 1
    return count


def validate_ernie55_mhc_config(config: object) -> None:
    """Raise ``ValueError`` if ``config`` enables hyper-connections outside the ERNIE 5.5 envelope.

    Configs without hyper-connections pass. With them:

    * ``use_fused_mhc=False``: the ERNIE operations replace the unfused ones only.
    * ``hidden_dropout=0``: with dropout Megatron-Core mixes the residual through ``apply_h_res``,
      outside the ERNIE fp32 update.
    * ``fp32_residual_connection=False``: the decoder carries fp32 streams by its own entry cast.
    * ``mhc_norm_eps_inside_sqrt=False`` (ERNIE adds the epsilon outside the square root) and
      ``mhc_learned_output_contract=True`` (ERNIE contracts the streams with the learned ``hc_head``).
    * With pipeline parallelism, ``pipeline_dtype=torch.float32``: stages exchange the fp32 streams,
      and a narrower pipeline dtype would round them at every stage boundary.

    Args:
        config: Transformer config.

    Raises:
        ValueError: On the first violated requirement.
    """
    if not getattr(config, "enable_mhc_connections", False):
        return

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"ERNIE 5.5 hyper-connections: {message}")

    require(not config.use_fused_mhc, "use_fused_mhc must be False (ERNIE 5.5 mixes eagerly in fp32).")
    require(not config.hidden_dropout, "hidden_dropout must be 0.")
    require(not config.fp32_residual_connection, "fp32_residual_connection must be False.")
    require(
        not getattr(config, "mhc_norm_eps_inside_sqrt", False),
        "mhc_norm_eps_inside_sqrt must be False (ERNIE 5.5 adds the epsilon outside the square root).",
    )
    require(
        getattr(config, "mhc_learned_output_contract", True),
        "mhc_learned_output_contract must be True (ERNIE 5.5 contracts the streams with the learned hc_head).",
    )
    if config.pipeline_model_parallel_size > 1:
        require(
            config.pipeline_dtype == torch.float32,
            f"pipeline_dtype must be torch.float32 with pipeline parallelism, got {config.pipeline_dtype}; "
            "the stages exchange the fp32 residual streams.",
        )
