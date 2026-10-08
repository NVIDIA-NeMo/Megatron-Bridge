# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Qwen4-Exp integration with the unmodified Megatron-Core Hybrid decoder."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer, TransformerLayerSubmodules

from megatron.bridge.models.qwen.modeling_qwen4_exp.gated_residual import GatedResidualOutputMixer
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import Qwen3VLMultimodalRotaryEmbedding


if TYPE_CHECKING:
    from megatron.bridge.models.qwen.modeling_qwen4_exp.model_config import Qwen4ExpTransformerConfig


@dataclass
class Qwen4ExpLayerSubmodules(TransformerLayerSubmodules):
    """Core layer submodules plus the model-specific lexical embedding."""

    per_layer_embedding: ModuleSpec | type = IdentityOp


class Qwen4ExpTransformerLayer(HyperConnectionTransformerLayer):
    """Inject PLE before reading the gated residual streams."""

    def __init__(
        self,
        config: "Qwen4ExpTransformerConfig",
        submodules: Qwen4ExpLayerSubmodules,
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__(config, submodules, *args, **kwargs)
        self.per_layer_embedding = None
        if submodules.per_layer_embedding is not IdentityOp:
            self.per_layer_embedding = build_module(
                submodules.per_layer_embedding,
                config=config,
                layer_number=self.layer_number,
                pg_collection=self.pg_collection,
            )

    def _forward_attention(
        self, hidden_states: torch.Tensor, *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.per_layer_embedding is not None:
            hidden_states = hidden_states + self.per_layer_embedding(hidden_states)
        return super()._forward_attention(hidden_states, *args, **kwargs)


class Qwen4ExpGatedDeltaNet(GatedDeltaNet):
    """Keep the SiLU convolution while using a separate sigmoid output gate."""

    def _apply_gated_norm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        normalized = self.out_norm(x.reshape(-1, x.shape[-1]))
        gate = gate.reshape(-1, gate.shape[-1]).float()
        if self.config.linear_attention_output_gate_activation == "sigmoid":
            gate = torch.sigmoid(gate)
        else:
            gate = self.act_fn(gate)
        return (normalized * gate).to(x_dtype)


class Qwen4ExpHybridStack(HybridStack):
    """Core hybrid execution with Qwen4-Exp's learned output stream mixer."""

    def __init__(
        self,
        config: "Qwen4ExpTransformerConfig",
        submodules: HybridStackSubmodules,
        **kwargs: object,
    ) -> None:
        super().__init__(config, submodules, post_layer_norm=False, **kwargs)
        self.post_layer_norm = True
        if self.post_process:
            self.final_norm = GatedResidualOutputMixer(config)


class _Qwen4ExpMultimodalRotaryEmbedding(Qwen3VLMultimodalRotaryEmbedding):
    """Adapt explicit VLM positions to HybridModel's rotary invocation."""

    position_ids: torch.Tensor | None = None
    packed_seq_params: PackedSeqParams | None = None

    def forward(self, max_seq_len: int, *, packed_seq: bool = False) -> torch.Tensor:
        if self.position_ids is None:
            raise ValueError("Qwen4-Exp multimodal rotary embedding requires position_ids.")
        return super().forward(self.position_ids, self.mrope_section, self.packed_seq_params)


class Qwen4ExpHybridModel(HybridModel):
    """Prepare lexical inputs and residual streams for the Core Hybrid decoder.

    Every hybrid position keeps the original attention-plus-MLP decoder block,
    including both gated residual connections. PLE layer numbers and checkpoint
    parameter indices therefore remain unchanged.
    """

    def __init__(
        self,
        config: "Qwen4ExpTransformerConfig",
        *,
        rotary_percent: float,
        rotary_base: int,
        mrope_section: list[int] | None = None,
        seq_len_interpolation_factor: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            config=config,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            **kwargs,
        )
        if mrope_section is not None:
            self.rotary_pos_emb = _Qwen4ExpMultimodalRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                rotary_base=rotary_base,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                cp_group=self.pg_collection.cp,
            )
            self.rotary_pos_emb.mrope_section = mrope_section

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        decoder_input: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        runtime_gather_output: bool | None = None,
        *,
        packed_seq_params: PackedSeqParams | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Execute the language decoder, retaining raw IDs for PLE with VLM embeddings."""
        if padding_mask is not None:
            raise NotImplementedError("Qwen4-Exp padding masks are not supported yet; use packed inputs.")
        if inference_context is not None or kwargs.get("inference_params") is not None:
            raise NotImplementedError("Qwen4-Exp does not support cached inference yet.")
        if self.config.ple_layer_ids:
            if input_ids is None:
                raise ValueError("Qwen4-Exp PLE requires raw input_ids, including with decoder_input.")
            cu_seqlens = None
            if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
                cu_seqlens = packed_seq_params.cu_seqlens_q_padded
                if cu_seqlens is None:
                    cu_seqlens = packed_seq_params.cu_seqlens_q
            for layer in self.decoder.layers:
                if layer.per_layer_embedding is not None:
                    layer.per_layer_embedding.prepare(input_ids, cu_seqlens)

        if decoder_input is None:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
            if self.config.sequence_parallel and not self.embedding.scatter_to_sequence_parallel:
                decoder_input = tensor_parallel.scatter_to_sequence_parallel_region(
                    decoder_input, group=self.pg_collection.tp
                )
        decoder_input = decoder_input.repeat(1, 1, self.config.mhc_num_residual_streams)
        multimodal_rope = isinstance(self.rotary_pos_emb, _Qwen4ExpMultimodalRotaryEmbedding)
        if multimodal_rope:
            self.rotary_pos_emb.position_ids = position_ids
            self.rotary_pos_emb.packed_seq_params = packed_seq_params
        try:
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                runtime_gather_output=runtime_gather_output,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )
        finally:
            if multimodal_rope:
                self.rotary_pos_emb.position_ids = None
                self.rotary_pos_emb.packed_seq_params = None
