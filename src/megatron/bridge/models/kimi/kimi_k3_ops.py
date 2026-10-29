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

"""Numerical operators used by the Kimi K3 language backbone."""

import torch
from torch import nn


def _patch_fla_kda_hopper_autotune() -> None:
    """Remove a KDA autotune choice that is invalid on Hopper."""
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_kernel_wy_dqkg_fused
    from fla.utils import IS_NVIDIA_HOPPER

    if not IS_NVIDIA_HOPPER:
        return

    autotuner = chunk_kda_bwd_kernel_wy_dqkg_fused.fn
    if getattr(autotuner, "_kimi_k3_hopper_configs_patched", False):
        return

    autotuner.configs = [
        config for config in autotuner.configs if not (config.kwargs["BK"] == 32 and config.num_warps == 4)
    ]
    autotuner._kimi_k3_hopper_configs_patched = True


def kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    *,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Kimi Delta Attention while preserving packed-sequence boundaries."""
    from fla.ops.kda import chunk_kda

    _patch_fla_kda_hopper_autotune()
    output, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        transpose_state_layout=True,
        cu_seqlens=cu_seqlens,
    )
    return output


def situ_and_mul(
    inputs: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    """Apply Kimi K3's SiTU gated activation."""
    gate, linear = torch.chunk(inputs.float(), 2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    linear = linear_beta * torch.tanh(linear / linear_beta)
    return (gate * linear).to(inputs.dtype)


class SiTUAndMul(nn.Module):
    """Module wrapper used by MCore's custom-activation path."""

    def __init__(self, config) -> None:
        super().__init__()
        del config

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply SiTU to a fused gate/up projection."""
        return situ_and_mul(inputs)


class KimiRMSNorm(nn.Module):
    """Kimi's FP32-accumulating RMS normalization."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        device: torch.device | int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the final dimension with FP32 accumulation."""
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(input_dtype)


def attn_res_aggregate(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_proj: nn.Linear,
    score_norm: KimiRMSNorm,
    output_norm: nn.Module,
) -> torch.Tensor:
    """Aggregate Kimi K3 AttnRes snapshots and the current prefix."""
    rows = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
    rows_float = rows.float()
    normalized = rows_float * torch.rsqrt(rows_float.square().mean(dim=-1, keepdim=True) + score_norm.eps)
    score_weight = score_norm.weight.float() * score_proj.weight.squeeze(0).float()
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    mixed = (probabilities.unsqueeze(-1) * rows_float).sum(dim=-2).to(rows.dtype)
    return output_norm(mixed)
