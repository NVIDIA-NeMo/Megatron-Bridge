# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math
from typing import Mapping

import torch

from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping


def get_common_mapping_list() -> list:
    """
    Returns a list of common parameter mappings for the DeepSeek family of models.
    """
    param_mappings = {
        # Embed
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        # Attention
        "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
        "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
        # Reference: https://github.com/NVIDIA/NeMo/blob/50cceb9c90ea1f440d1e14074fa13bd45f60a1c4/nemo/collections/llm/gpt/model/deepseek.py#L637-L650
        #  In deepseek, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
        #  (a) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
        #  (b) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
        "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
        "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
        "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
        "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
        "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
        # Mcore local spec
        "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
        # Dense MLP
        "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        # MoE
        "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
        "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
        "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
        # LM Head
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
        # MLA
        "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
        "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
        "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
        # Mcore local spec
        "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
        # For models without MLA
        "decoder.layers.*.self_attention.linear_q_proj.weight": "model.layers.*.self_attn.q_proj.weight",
    }

    # TODO: mtp layers

    mapping_list = []
    # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
    for megatron_param, hf_param in param_mappings.items():
        mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

    mapping_list.extend(
        [
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                up="model.layers.*.mlp.experts.*.up_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                up="model.layers.*.mlp.shared_experts.up_proj.weight",
            ),
        ]
    )

    return mapping_list


def dequantize_fp8_blockwise(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 block-wise quantized weight tensor to higher precision.

    Block sizes are inferred from the shapes of *weight* and *scale_inv*:
    ``block_m = ceil(M / scale_inv.shape[0])``, and likewise for the column
    dimension.  This matches the DeepSeek-V3 / Kimi-K2.5 FP8 convention where
    ``weight_block_size = [128, 128]``.

    Args:
        weight: FP8 weight tensor, shape ``[M, N]`` (``torch.float8_e4m3fn``).
        scale_inv: Per-block inverse scale factors, shape
            ``[ceil(M/block_m), ceil(N/block_n)]``.
        dtype: Target output dtype (default ``torch.bfloat16``).

    Returns:
        Dequantized tensor of shape ``[M, N]`` in *dtype*.
    """
    M, N = weight.shape
    scale_rows, scale_cols = scale_inv.shape
    block_m = math.ceil(M / scale_rows)
    block_n = math.ceil(N / scale_cols)

    padded_M = scale_rows * block_m
    padded_N = scale_cols * block_n

    if M != padded_M or N != padded_N:
        result = torch.zeros(padded_M, padded_N, dtype=dtype, device=weight.device)
        result[:M, :N] = weight.to(dtype)
    else:
        result = weight.to(dtype)

    result = result.reshape(scale_rows, block_m, scale_cols, block_n)
    result.mul_(scale_inv[:, None, :, None].to(dtype))
    result = result.reshape(padded_M, padded_N)

    if M != padded_M or N != padded_N:
        result = result[:M, :N].contiguous()
    return result


def maybe_dequantize_fp8_weight(
    hf_param: str,
    hf_weights: torch.Tensor,
    hf_state_dict: Mapping[str, torch.Tensor],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Return *hf_weights* dequantized to *dtype* when FP8, otherwise pass through.

    Detection heuristic: the weight has ``float8_e4m3fn`` dtype **and** a
    matching ``{hf_param}_scale_inv`` key exists in *hf_state_dict*.
    """
    if not hasattr(torch, "float8_e4m3fn") or hf_weights.dtype != torch.float8_e4m3fn:
        return hf_weights

    scale_inv_key = hf_param + "_scale_inv"
    if scale_inv_key not in hf_state_dict:
        return hf_weights

    return dequantize_fp8_blockwise(
        hf_weights,
        hf_state_dict[scale_inv_key],
        dtype=dtype,
    )
