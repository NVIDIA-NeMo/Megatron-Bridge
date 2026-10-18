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

"""D1 gated three-tap convolution composed with MCore TE projections."""

import torch
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from torch.nn import functional as F


class D1ShortConv(MegatronModule):
    """Causal or centred D1 mixer for an unpartitioned language backbone.

    This block implements the short convolution absent from MCore's Mamba
    blocks. Input and output projections use MCore's Transformer Engine
    adapter. The centred variant isolates media queries from following text.

    Args:
        config: Native transformer dimensions, precision and initialization.
        causal: Use left-only taps for d1-3B, or centred taps for Omni.
    """

    def __init__(self, config: TransformerConfig, *, causal: bool) -> None:
        super().__init__(config)
        if config.tensor_model_parallel_size != 1 or config.sequence_parallel:
            raise ValueError("D1ShortConv currently requires TP=1 and sequence_parallel=False")
        self.causal = causal
        width = config.hidden_size
        self.in_proj = TELinear(
            width,
            3 * width,
            parallel_mode="duplicated",
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )
        self.out_proj = TELinear(
            width,
            width,
            parallel_mode="duplicated",
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )
        device = (
            torch.device("cpu") if config.use_cpu_initialization else torch.device("cuda", torch.cuda.current_device())
        )
        self.conv = nn.Conv1d(width, width, 3, groups=width, bias=False, device=device, dtype=config.params_dtype)
        if config.perform_initialization:
            config.init_method(self.conv.weight)
        self.conv.weight.allreduce = True
        self.conv.weight.tensor_model_parallel = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        valid_tokens: torch.Tensor,
        media_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Mix states while respecting padding and the media/text boundary.

        Args:
            hidden_states: Sequence-first states [length, batch, width].
            valid_tokens: Boolean mask [batch, length].
            media_lengths: Long prefix lengths [batch]; zero for causal text.

        Returns:
            Sequence-first states [length, batch, width].
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden_states must have shape [length, batch, hidden_size]")
        length, batch, _ = hidden_states.shape
        if valid_tokens.shape != (batch, length) or valid_tokens.dtype != torch.bool:
            raise ValueError("valid_tokens must be boolean with shape [batch, length]")
        if media_lengths.shape != (batch,) or media_lengths.dtype != torch.long:
            raise ValueError("media_lengths must be long with shape [batch]")
        if valid_tokens.device != hidden_states.device or media_lengths.device != hidden_states.device:
            raise ValueError("states, valid_tokens and media_lengths must share a device")
        if torch.any((media_lengths < 0) | (media_lengths > length)):
            raise ValueError("media_lengths must lie within the sequence")
        masked_states = hidden_states * valid_tokens.transpose(0, 1).unsqueeze(-1)
        projected, _ = self.in_proj(masked_states)
        input_gate, output_gate, values = projected.permute(1, 2, 0).chunk(3, dim=1)
        gated_values = input_gate * values
        if self.causal:
            mixed = F.conv1d(F.pad(gated_values, (2, 0)), self.conv.weight, groups=self.config.hidden_size)
        else:
            neighbours = F.pad(gated_values, (1, 1)).unfold(-1, 3, 1)
            positions = torch.arange(length, device=hidden_states.device).unsqueeze(0)
            allow_right = (positions != media_lengths.unsqueeze(1) - 1).to(hidden_states.dtype)
            weights = self.conv.weight[:, 0]
            left = neighbours[..., 0] * weights[:, 0].unsqueeze(-1)
            centre = neighbours[..., 1] * weights[:, 1].unsqueeze(-1)
            right = neighbours[..., 2] * weights[:, 2].unsqueeze(-1) * allow_right.unsqueeze(1)
            mixed = (left + centre) + right
        output, _ = self.out_proj((output_gate * mixed).permute(2, 0, 1))
        return output
