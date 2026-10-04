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

"""Qwen4-Exp integration with the unmodified Megatron-Core GPT decoder."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import torch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import WrappedTensor


if TYPE_CHECKING:
    from megatron.bridge.models.qwen.modeling_qwen4_exp.provider import Qwen4ExpModelProvider


@dataclass
class Qwen4ExpLayerSubmodules(TransformerLayerSubmodules):
    """Core layer submodules plus the model-specific lexical embedding."""

    per_layer_embedding: Union[ModuleSpec, type] = IdentityOp


class Qwen4ExpTransformerLayer(HyperConnectionTransformerLayer):
    """Inject PLE before reading the gated residual streams."""

    def __init__(self, config: "Qwen4ExpModelProvider", submodules: Qwen4ExpLayerSubmodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        self.per_layer_embedding = None
        if submodules.per_layer_embedding is not IdentityOp:
            self.per_layer_embedding = build_module(
                submodules.per_layer_embedding,
                config=config,
                layer_number=self.layer_number,
                pg_collection=self.pg_collection,
            )

    def _forward_attention(self, hidden_states, *args, **kwargs):
        if self.per_layer_embedding is not None:
            hidden_states = hidden_states + self.per_layer_embedding(hidden_states)
        return super()._forward_attention(hidden_states, *args, **kwargs)


class Qwen4ExpGatedDeltaNet(GatedDeltaNet):
    """Keep the SiLU convolution while using a separate sigmoid output gate."""

    def _apply_gated_norm(self, x, gate):
        x_dtype = x.dtype
        normalized = self.out_norm(x.reshape(-1, x.shape[-1]))
        gate = gate.reshape(-1, gate.shape[-1]).float()
        if self.config.linear_attention_output_gate_activation == "sigmoid":
            gate = torch.sigmoid(gate)
        else:
            gate = self.act_fn(gate)
        return (normalized * gate).to(x_dtype)


class Qwen4ExpGPTModel(GPTModel):
    """Prepare lexical inputs and expand streams outside Core's mHC block path.

    The Core block's generic mHC path contracts streams with an unweighted mean.
    Qwen4-Exp instead uses a learned output mixer in the final-layernorm slot.
    Keeping Core's mHC flag disabled lets the existing block execute the local
    hyper-connection layers without first discarding their stream dimension.
    """

    def _preprocess(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        decoder_input: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        if padding_mask is not None:
            raise NotImplementedError("Qwen4-Exp padding masks are not supported yet; use packed inputs.")
        if inference_context is not None:
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

        result = super()._preprocess(
            input_ids,
            position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )
        hidden = result[0]
        wrapped = isinstance(hidden, WrappedTensor)
        if wrapped:
            hidden = hidden.unwrap()
        hidden = hidden.repeat(1, 1, self.config.mhc_num_residual_streams)
        if wrapped:
            hidden = WrappedTensor(hidden)
        return (hidden, *result[1:])
