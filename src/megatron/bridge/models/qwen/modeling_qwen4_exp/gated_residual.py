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

"""Qwen4-Exp gated residual streams and learned output contraction."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator
from torch import Tensor

from megatron.bridge.models.qwen.modeling_qwen4_exp.gated_residual_math import (
    activate_residual_gate,
    mix_residual_streams,
)
from megatron.bridge.models.qwen.modeling_qwen4_exp.norms import grouped_rmsnorm


def gated_residual_group_rmsnorm(x: Tensor, weight: Tensor, n: int, eps: float) -> Tensor:
    """Zero-centered RMSNorm applied independently to each of the ``n`` residual streams.

    Args:
        x: [..., n*C] n-stream hidden states.
        weight: [n*C] zero-centered gain (the effective scale is ``1 + weight``).
        n: Number of residual streams.
        eps: RMSNorm epsilon.

    Returns:
        [..., n*C] normalized streams in ``x.dtype`` (computed in fp32).
    """
    return grouped_rmsnorm(x, weight, n, eps)


class GatedResidualHyperConnection(MegatronModule):
    """Gated Residual hyper-connection of Qwen4-Exp (Qwen3.8-Flash-Next).

    Every sub-layer (attention / MLP) of a decoder layer is wrapped by one instance. With ``n``
    residual streams of width ``C`` travelling between layers as ``[s, b, n*C]``:

        xn      = GroupRMSNorm(x)                                # per-stream, zero-centered gain
        mix     = sigmoid(W_up silu(W_down xn / n))              # [.., n*C] element-wise read gate
        block_in = mean_j(mix_j * xn_j)                          # [.., C]   sub-layer input
        inject  = 2 * sigmoid(W_inject xn / n)                   # [.., n]   per-stream write gate
        x_new   = x + inject_j * F(block_in)  for every stream j

    The interface matches :class:`HyperConnectionModule` so that
    :class:`~megatron.core.transformer.transformer_layer.HyperConnectionTransformerLayer` drives
    both formulations: ``forward`` returns ``(block_input, h_res, h_post)`` where ``h_res`` is
    ``None`` (the residual mixing is the identity) and ``h_post`` is ``inject``, and
    :meth:`fused_h_res_h_post_bda` applies the write gate. With ``use_combine=False`` the module
    is the block-level output mixer: it only returns ``block_in`` (see
    :class:`GatedResidualOutputMixer`).

    Parameters are replicated across tensor-parallel ranks and kept in ``params_dtype`` like the
    reference checkpoints; the norm statistics run in fp32; gates retain activation-dtype operation boundaries.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = 1, use_combine: bool = True):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.n = config.mhc_num_residual_streams
        self.hidden_size = config.hidden_size
        self.rank = config.mhc_gated_residual_rank
        self.use_combine = use_combine
        self.norm_eps = config.layernorm_epsilon
        hc_hidden_size = self.n * self.hidden_size
        dtype = config.params_dtype

        self.hc_norm = nn.Module()
        self.hc_norm.weight = nn.Parameter(torch.zeros(hc_hidden_size, dtype=dtype))
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, self.rank, bias=False, dtype=dtype)
        self.input_mix_weight_up = nn.Linear(self.rank, hc_hidden_size, bias=False, dtype=dtype)
        if use_combine:
            self.block_inject_weight = nn.Linear(hc_hidden_size, self.n, bias=False, dtype=dtype)
        else:
            self.block_inject_weight = None

        if self.config.perform_initialization:
            nn.init.normal_(self.input_mix_weight_down.weight, mean=0.0, std=config.init_method_std)
            nn.init.normal_(self.input_mix_weight_up.weight, mean=0.0, std=config.init_method_std)
            if self.block_inject_weight is not None:
                nn.init.normal_(self.block_inject_weight.weight, mean=0.0, std=config.init_method_std)

        # Replicated (non tensor-parallel) parameters: their gradients must be all-reduced
        # across the TP group when sequence parallelism shards the tokens.
        if self.config.sequence_parallel:
            for param in self.parameters():
                setattr(param, "sequence_parallel", True)

    def _mix(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        n = self.n
        xn = gated_residual_group_rmsnorm(hidden_states, self.hc_norm.weight, n, self.norm_eps)
        down = activate_residual_gate(self.input_mix_weight_down(xn), n)
        block_input = mix_residual_streams(self.input_mix_weight_up(down), xn, n)
        if self.block_inject_weight is None:
            return block_input, None
        inject = 2.0 * torch.sigmoid(self.block_inject_weight(xn) / n)  # [s, b, n]
        return block_input, inject

    def forward(self, hidden_states: Tensor, mhc_recompute_manager=None, return_residual: bool = False):
        """Read the streams.

        Args:
            hidden_states: [s, b, n*C] n-stream hidden states.
            mhc_recompute_manager: Accepted for interface compatibility; not used.
            return_residual: Also return the (unchanged) n-stream residual as a 4th element,
                like :class:`HyperConnectionModule`.

        Returns:
            ``(block_input [s, b, C], h_res=None, h_post [s, b, n])``. ``h_post`` is ``None``
            for the output mixer (``use_combine=False``).
        """
        block_input, inject = self._mix(hidden_states)
        if return_residual:
            return block_input, None, inject, hidden_states
        return block_input, None, inject

    def apply_h_res(self, h_res: Optional[Tensor], residual: Tensor) -> Tensor:
        """The gated residual keeps the streams untouched (identity residual mixing)."""
        return residual

    @staticmethod
    def _inject(residual: Tensor, inject: Tensor, x: Tensor, bias: Optional[Tensor], n: int) -> Tensor:
        s, b, nC = residual.shape
        out = x
        if bias is not None:
            out = out + bias
        update = inject.unsqueeze(-1) * out.unsqueeze(2)  # [s, b, n, C]
        return (residual.view(s, b, n, -1) + update).view(s, b, nC).to(residual.dtype)

    @nvtx_decorator(message="GatedResidual::fused_h_res_h_post_bda")
    def fused_h_res_h_post_bda(
        self,
        h_res: Optional[Tensor],
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager=None,
    ) -> Tensor:
        """Write the sub-layer output back to every stream: ``x + inject_j * (F(x) + bias)``.

        Args:
            h_res: Unused (identity residual mixing).
            original_residual: [s, b, n*C] n-stream hidden states before the sub-layer.
            h_post: [s, b, n] per-stream write gates returned by :meth:`forward`.
            layer_output_with_bias: ``(x [s, b, C], bias [C] or None)``.
            dropout_prob / training / fused: bias-dropout-add settings; dropout is applied to the
                broadcast update when non-zero.
            manager: Accepted for interface compatibility; not used.

        Returns:
            [s, b, n*C] updated streams.
        """
        x, bias = layer_output_with_bias
        if dropout_prob == 0.0 or not training:
            return self._inject(original_residual, h_post, x, bias, self.n)
        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

        s, b, nC = original_residual.shape
        x_expanded = (h_post.unsqueeze(-1) * x.float().unsqueeze(2)).view(s, b, nC).to(x.dtype)
        bias_expanded = (
            (h_post.unsqueeze(-1) * bias.float().view(1, 1, 1, -1)).view(s, b, nC).to(x.dtype)
            if bias is not None
            else None
        )
        bda_func = get_bias_dropout_add(training, fused)
        return bda_func((x_expanded, bias_expanded), original_residual, dropout_prob)


class GatedResidualOutputMixer(GatedResidualHyperConnection):
    """Block-level learned contraction of the Qwen4-Exp residual streams.

    Replaces the final layer norm of a
    :class:`~megatron.core.transformer.transformer_block.TransformerBlock`
    built with the Bridge-local Qwen4-Exp layer spec: it consumes the
    ``[s, b, n*C]`` streams and returns the ``[s, b, C]`` gated mean of their normalized values
    (``hyper_connection_mixer`` in the HF checkpoint). Core's generic mHC path remains
    disabled; the local GPT model expands streams before the decoder so the learned
    mixer receives every stream without Core's unweighted mean contraction.

    The ``(config, hidden_size, eps)`` signature matches the layer-norm builders used in block
    specs.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: Optional[int] = None,
        eps: Optional[float] = None,
    ):
        super().__init__(config, layer_number=0, use_combine=False)
        if hidden_size is not None:
            assert hidden_size == config.hidden_size, "GatedResidualOutputMixer mixes full hidden streams."
        if eps is not None:
            self.norm_eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:  # type: ignore[override]
        """[s, b, n*C] streams -> [s, b, C] mixed output."""
        block_input, _ = self._mix(hidden_states)
        return block_input
