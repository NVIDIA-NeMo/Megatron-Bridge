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

"""Inkling2 decoder blocks as Hybrid layers joined by Block Attention Residuals.

Each decoder block is an attention layer (Hybrid symbol ``*``) followed by a MoE
layer (``E``), so Hybrid layers ``2 * i`` and ``2 * i + 1`` hold block ``i``.
Each sublayer reads a learned softmax mixture over the completed residual blocks
and the current partial block instead of a single residual stream. Full-attention
blocks mix separate Q, K, and V inputs. Every attention and MoE output passes
through a residual short convolution before it joins the partial block.

HybridStack passes a single tensor from layer to layer, so every layer packs the
AttnRes bank of completed blocks next to the partial block along the hidden
dimension. The width grows by one hidden size per completed block, also across
pipeline stages, whose communication exchanges tensor shapes.
"""

import torch
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.transformer_layer import TransformerLayer
from torch import nn

from megatron.bridge.models.inkling.modeling_inkling2.attention import packed_cu_seqlens
from megatron.bridge.models.inkling.modeling_inkling2.context_parallel import context_parallel_layout
from megatron.bridge.models.inkling.modeling_inkling2.ops import (
    attn_res_mix,
    document_positions,
    residual_add,
    situ_and_mul,
    static_short_conv,
)
from megatron.bridge.models.kimi.kimi_k3_pipeline import bank_num_rows


class Inkling2SiTU(nn.Module):
    """Softcapped SwiGLU expert activation, installed through MCore's custom-activation path."""

    def __init__(self, config) -> None:
        super().__init__()
        self.gate_softcap = config.inkling2_mlp_gate_softcap
        self.up_softcap = config.inkling2_mlp_up_softcap
        self.compiled = config.inkling2_compile_ops

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the activation to a fused ``[gate, up]`` projection."""
        return situ_and_mul(inputs, self.gate_softcap, self.up_softcap, compiled=self.compiled)


def num_attn_res_blocks_before(block_index: int, block_size: int) -> int:
    """Return the number of completed AttnRes blocks that decoder block ``block_index`` reads."""
    return 0 if block_index == 0 else bank_num_rows(block_index, block_size)


def unpack_attn_res_state(state: torch.Tensor, hidden_size: int, count: int) -> list[torch.Tensor]:
    """Split a packed AttnRes state into its ``count`` hidden-size slices."""
    if state.shape[-1] != count * hidden_size:
        raise ValueError(f"AttnRes state width {state.shape[-1]} != {count} x {hidden_size}")
    return list(state.split(hidden_size, dim=-1))


class _Inkling2HybridLayer(TransformerLayer):
    """AttnRes bookkeeping shared by the attention and MoE layers of a decoder block."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = self.config
        if config.hidden_dropout != 0.0:
            raise ValueError("Inkling2 requires hidden_dropout=0")
        self.block_index = (self.layer_number - 1) // 2
        self.block_size = config.inkling2_attn_res_block_size
        self.starts_attn_res_block = self.block_index % self.block_size == 0
        self.num_blocks_before = num_attn_res_blocks_before(self.block_index, self.block_size)
        self.normalize_values = config.inkling2_attn_res_normalize_values
        self._keep_in_float32_parameter_names: tuple[str, ...] = ()

    def _add_float32_parameter(self, name: str, *shape: int) -> None:
        """Register an AttnRes query, which stays FP32 under BF16 training."""
        parameter = nn.Parameter(torch.zeros(*shape, device=torch.cuda.current_device(), dtype=torch.float32))
        # Each sequence-parallel rank sees different tokens; their gradients add.
        parameter.sequence_parallel = True
        self.register_parameter(name, mark_keep_in_fp32(parameter))
        self._keep_in_float32_parameter_names += (name,)

    def _mix(self, sources: list[torch.Tensor], queries: torch.Tensor, **kwargs) -> torch.Tensor:
        return attn_res_mix(
            sources,
            queries,
            normalize_values=self.normalize_values,
            compiled=self.config.inkling2_compile_ops,
            **kwargs,
        )

    def _normalize(self, norm: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize an AttnRes mixture into the activation dtype."""
        return norm(inputs.to(self.config.params_dtype))

    def _check_supported(self, inference_context) -> None:
        if inference_context is not None:
            raise ValueError("Inkling2 does not support Megatron's KV-cache inference yet")


class Inkling2AttentionLayer(_Inkling2HybridLayer):
    """Attention layer of an Inkling2 decoder block (Hybrid symbol ``*``)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = self.config
        hidden = config.hidden_size
        self.is_local = self.block_index in config.inkling2_local_layers
        if not self.is_local:
            # ``input_layernorm`` normalizes the Q-side mixture; K and V have their own norms.
            self.k_input_norm = TENorm(config, hidden, config.layernorm_epsilon)
            self.v_input_norm = TENorm(config, hidden, config.layernorm_epsilon)
        if self.block_index > 0:
            self._add_float32_parameter("attn_res_query", *((hidden,) if self.is_local else (3, hidden)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inference_context=None,
        packed_seq_params: PackedSeqParams | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Attend over the AttnRes mixtures and hand the block's state to its MoE layer.

        Args:
            hidden_states: Packed ``[partial, *bank]`` state of the previous block.
            packed_seq_params: Document boundaries of a packed batch.

        Returns:
            ``[attention, *bank, partial]`` when the block starts a new AttnRes block,
            otherwise ``[partial, attention, *bank]``; and no context.
        """
        del attention_mask, kwargs
        self._check_supported(inference_context)
        partial, *bank = unpack_attn_res_state(hidden_states, self.config.hidden_size, 1 + self.num_blocks_before)
        if self.block_index == 0:
            query_input = key_input = value_input = partial
        else:
            mixtures = self._mix([*bank, partial], self.attn_res_query)
            query_input = key_input = value_input = mixtures[0]
            if not self.is_local:
                key_input, value_input = mixtures[1], mixtures[2]
        normalized_query = self._normalize(self.input_layernorm, query_input)
        if self.is_local:
            normalized_key = normalized_value = normalized_query
        else:
            normalized_key = self._normalize(self.k_input_norm, key_input)
            normalized_value = self._normalize(self.v_input_norm, value_input)
        attention = self.self_attention(
            normalized_query, normalized_key, normalized_value, packed_seq_params=packed_seq_params
        )
        if self.starts_attn_res_block:
            return torch.cat((attention, *bank, partial), dim=-1), None
        return torch.cat((partial, attention, *bank), dim=-1), None


class Inkling2MoELayer(_Inkling2HybridLayer):
    """MoE layer of an Inkling2 decoder block (Hybrid symbol ``E``)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = self.config
        hidden = config.hidden_size
        self.mlp_conv_weight = nn.Parameter(
            torch.zeros(
                hidden,
                1,
                config.inkling2_sconv_kernel_size,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
        self._add_float32_parameter("mlp_res_query", hidden)
        self.num_blocks_after = self.num_blocks_before + int(self.starts_attn_res_block)
        self.is_last_block = self.layer_number == config.num_layers
        if self.is_last_block:
            self._add_float32_parameter("output_res_query", hidden)

    def _mlp_conv(self, output: torch.Tensor, packed_seq_params: PackedSeqParams | None) -> torch.Tensor:
        sequence_parallel = self.config.sequence_parallel
        if sequence_parallel:
            output = gather_from_sequence_parallel_region(output, tensor_parallel_output_grad=False)
        seq_len, batch = output.shape[:2]
        cu_seqlens = packed_cu_seqlens(packed_seq_params)
        layout = context_parallel_layout(
            self.pg_collection.cp,
            local_tokens=seq_len,
            batch=batch,
            device=output.device,
            width=self.config.inkling2_sconv_kernel_size,
            cu_seqlens=cu_seqlens,
        )
        if layout is None:
            positions = document_positions(seq_len, batch, output.device, cu_seqlens)
        else:
            positions = layout.local_positions
        output = static_short_conv(
            output,
            self.mlp_conv_weight.squeeze(1),
            positions,
            halo=layout,
            compiled=self.config.inkling2_compile_ops,
        )
        if sequence_parallel:
            output = scatter_to_sequence_parallel_region(output)
        return output

    @staticmethod
    def _without_bias(output_with_bias: tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
        output, bias = output_with_bias
        return output if bias is None else output + bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        inference_context=None,
        packed_seq_params: PackedSeqParams | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Run the MoE on its AttnRes mixture and update the partial block.

        Args:
            hidden_states: Packed state from the block's attention layer.
            packed_seq_params: Document boundaries of a packed batch.
            padding_mask: Optional MoE router padding mask.

        Returns:
            The packed ``[partial, *bank]`` state, or the final AttnRes output after
            the last block; and no context.
        """
        del attention_mask, kwargs
        self._check_supported(inference_context)
        hidden = self.config.hidden_size
        if self.starts_attn_res_block:
            attention, *bank = unpack_attn_res_state(hidden_states, hidden, 1 + self.num_blocks_after)
            mlp_input = self._mix([*bank, attention], self.mlp_res_query)
        else:
            partial, attention, *bank = unpack_attn_res_state(hidden_states, hidden, 2 + self.num_blocks_after)
            mlp_input = self._mix([*bank, partial], self.mlp_res_query, partial_addend=attention)
        mlp_output = self._without_bias(
            self.mlp(self._normalize(self.pre_mlp_layernorm, mlp_input[0]), padding_mask=padding_mask)
        )
        mlp_output = self._mlp_conv(mlp_output, packed_seq_params)
        if self.starts_attn_res_block:
            partial = attention + mlp_output
        else:
            partial = residual_add(partial, attention, mlp_output, compiled=self.config.inkling2_compile_ops)
        if self.is_last_block:
            output = self._mix([*bank, partial], self.output_res_query)[0]
            return output.to(partial.dtype), None
        return torch.cat((partial, *bank), dim=-1), None
