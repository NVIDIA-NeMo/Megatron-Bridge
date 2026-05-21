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

"""Compatibility helpers for Megatron-Core Qwen-VL integration."""

from __future__ import annotations

import inspect
import warnings
from contextlib import nullcontext

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from torch import Tensor


def _checkpointed_forward_with_padding_mask(
    self: MultiTokenPredictionLayer,
    hidden_states: Tensor,
    decoder_input: Tensor,
    attention_mask: Tensor | None = None,
    padding_mask: Tensor | None = None,
    context: Tensor | None = None,
    context_mask: Tensor | None = None,
    rotary_pos_emb: Tensor | None = None,
    rotary_pos_cos: Tensor | None = None,
    rotary_pos_sin: Tensor | None = None,
    attention_bias: Tensor | None = None,
    inference_params: InferenceParams | None = None,
    packed_seq_params: PackedSeqParams | None = None,
    sequence_len_offset: Tensor | None = None,
) -> Tensor:
    def custom_forward(
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: Tensor | None,
        padding_mask: Tensor | None,
        context: Tensor | None,
        context_mask: Tensor | None,
        rotary_pos_emb: Tensor | None,
        rotary_pos_cos: Tensor | None,
        rotary_pos_sin: Tensor | None,
        sequence_len_offset: Tensor | None,
    ) -> Tensor:
        return self._proj_and_transformer_layer(
            hidden_states=hidden_states,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

    if self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed:
        outer_quantization_context = get_fp8_context(self.config)
    else:
        outer_quantization_context = nullcontext()

    def checkpoint_handler() -> Tensor:
        if self.config.fp8 or self.config.fp4:
            from megatron.core.extensions.transformer_engine import te_checkpoint

            return te_checkpoint(
                custom_forward,
                self.config.distribute_saved_activations,
                tensor_parallel.random.get_cuda_rng_tracker,
                parallel_state.get_tensor_model_parallel_group(),
                hidden_states,
                decoder_input,
                attention_mask,
                padding_mask,
                context,
                context_mask,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            )

        return tensor_parallel.checkpoint(
            custom_forward,
            self.config.distribute_saved_activations,
            hidden_states,
            decoder_input,
            attention_mask,
            padding_mask,
            context,
            context_mask,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )

    if self.config.recompute_method == "uniform":
        assert self.config.recompute_num_layers == 1, "recompute_num_layers must be 1 for MTP recompute"
        with outer_quantization_context:
            return checkpoint_handler()
    if self.config.recompute_method == "block":
        warnings.warn("recompute_method == 'block' is not supported for MTP yet. Skipping recompute.", stacklevel=2)
        return self._proj_and_transformer_layer(
            hidden_states=hidden_states,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

    raise ValueError("Invalid activation recompute method.")


def ensure_mtp_checkpointed_forward_accepts_padding_mask() -> bool:
    """Install the MTP recompute padding-mask guard when the MCore helper is missing it.

    Returns:
        True when the guard is installed, False when the current MCore already has the
        native parameter or the guard was installed earlier.
    """
    checkpointed_forward = MultiTokenPredictionLayer._checkpointed_forward
    checkpointed_parameters = inspect.signature(checkpointed_forward).parameters
    if "padding_mask" in checkpointed_parameters:
        return False
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in checkpointed_parameters.values()):
        return False

    projection_parameters = inspect.signature(MultiTokenPredictionLayer._proj_and_transformer_layer).parameters
    if "padding_mask" not in projection_parameters:
        return False

    # TODO: remove this guard when NVIDIA/Megatron-LM fixes
    # MultiTokenPredictionLayer._checkpointed_forward to accept padding_mask on both main and dev.
    MultiTokenPredictionLayer._checkpointed_forward = _checkpointed_forward_with_padding_mask
    return True
