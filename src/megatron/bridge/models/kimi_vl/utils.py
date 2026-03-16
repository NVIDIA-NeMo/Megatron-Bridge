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


import torch
import math
from typing import Mapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

try:
    import apex  # noqa: F401

    HAVE_APEX = True
except ImportError:
    HAVE_APEX = False



def get_common_configs(hf_pretrained: PreTrainedCausalLM) -> dict:
    """
    Returns a dictionary of common configurations for the DeepSeek family of models.
    """
    hf_config = hf_pretrained.config

    configs = {}

    if not HAVE_APEX:
        configs["gradient_accumulation_fusion"] = False

    if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling is not None:
        configs["rotary_scaling_factor"] = hf_config.rope_scaling["factor"]
        configs["mscale"] = hf_config.rope_scaling["mscale"]
        configs["mscale_all_dim"] = hf_config.rope_scaling["mscale_all_dim"]
    else:
        configs["rotary_scaling_factor"] = 1.0
        configs["mscale"] = 1.0
        configs["mscale_all_dim"] = 1.0

    configs["num_layers"] = hf_config.num_hidden_layers
    configs["hidden_size"] = hf_config.hidden_size
    configs["ffn_hidden_size"] = hf_config.intermediate_size
    configs["num_attention_heads"] = hf_config.num_attention_heads
    configs["kv_channels"] = hf_config.num_key_value_heads
    configs["q_lora_rank"] = hf_config.q_lora_rank
    configs["num_moe_experts"] = hf_config.n_routed_experts
    configs["moe_ffn_hidden_size"] = hf_config.moe_intermediate_size
    configs["moe_shared_expert_intermediate_size"] = hf_config.moe_intermediate_size * hf_config.n_shared_experts
    configs["moe_layer_freq"] = [0] * hf_config.first_k_dense_replace + [1] * (
        hf_config.num_hidden_layers - hf_config.first_k_dense_replace
    )
    configs["moe_router_topk"] = hf_config.num_experts_per_tok
    configs["moe_router_num_groups"] = hf_config.n_group
    configs["moe_router_group_topk"] = hf_config.topk_group
    configs["moe_router_topk_scaling_factor"] = hf_config.routed_scaling_factor
    configs["kv_lora_rank"] = hf_config.kv_lora_rank
    configs["qk_head_dim"] = hf_config.qk_nope_head_dim
    configs["qk_pos_emb_head_dim"] = hf_config.qk_rope_head_dim
    configs["v_head_dim"] = hf_config.v_head_dim

    # Ensure MLA is enabled
    configs["multi_latent_attention"] = True
    configs["generation_config"] = hf_pretrained.generation_config
    configs["vocab_size"] = hf_config.vocab_size
    configs["rotary_base"] = hf_config.rope_theta
    configs["init_method_std"] = hf_config.initializer_range
    configs["layernorm_epsilon"] = hf_config.rms_norm_eps

    return configs


def dequantize_int4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    group_size: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    """Dequantize INT4 packed weights to bfloat16.

    Extracts local tensors from DTensors before unpacking (bitwise ops don't work on DTensor).
    Both weight_packed and weight_scale should have matching sharding so .to_local() gives
    corresponding slices automatically.

    Args:
        weight_packed: INT4 packed weights [out_features, in_features // 8], may be DTensor
        weight_scale: Per-group scales [out_features, num_groups], should be DTensor with same sharding
        weight_shape: Original shape [2], stores global dimensions
        group_size: Elements per scale group (default 32)
        device: Target device for computation
    """

    is_packed_dtensor = hasattr(weight_packed, "device_mesh")
    is_scale_dtensor = hasattr(weight_scale, "device_mesh")

    if is_packed_dtensor:
        weight_packed = weight_packed.to_local()

    if is_scale_dtensor:
        weight_scale = weight_scale.to_local()

    local_out, local_packed_in = weight_packed.shape
    local_in = local_packed_in * 8  # 8 INT4 values per int32

    use_cuda = device == "cuda" and torch.cuda.is_available()

    if use_cuda:
        weight_packed = weight_packed.cuda()
        weight_scale = weight_scale.cuda()

    # Unpack INT4: [out, packed_in] -> [out, packed_in, 8] -> [out, in_features]
    shifts = torch.arange(8, device=weight_packed.device) * 4

    packed_unsqueezed = weight_packed.unsqueeze(-1)
    unpacked = ((packed_unsqueezed >> shifts) & 0xF).float()
    unpacked = unpacked.reshape(local_out, local_in)

    # Convert unsigned 4-bit (0-15) to signed (-8 to 7) using OFFSET BINARY
    # This matches compressed-tensors library which packs as: value + 8
    # So unpack as: value - 8
    unpacked = unpacked - 8

    # Apply scale - both are now local tensors with corresponding slices
    scale = weight_scale.float()
    if scale.ndim == 1:
        local_num_groups = scale.numel() // local_out
        scale = scale.view(local_out, local_num_groups)
    else:
        scale = scale.view(local_out, -1)

    local_num_groups = scale.shape[1]
    elements_per_group = local_in // local_num_groups

    # repeat_interleave expands [local_out, local_num_groups] -> [local_out, local_in]
    scale_expanded = scale.repeat_interleave(elements_per_group, dim=1)

    if scale_expanded.shape[1] < local_in:
        # Pad if needed
        scale_expanded = torch.nn.functional.pad(
            scale_expanded, (0, local_in - scale_expanded.shape[1]), value=scale_expanded[:, -1:].mean()
        )
    scale_expanded = scale_expanded[:, :local_in]
    result = unpacked * scale_expanded

    result = result.to(torch.bfloat16)

    return result


def quantize_to_int4(
    weight: torch.Tensor,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize bfloat16/float16 weights to INT4 packed format.

    Returns:
        weight_packed: INT4 values packed into int32 (8 values per int32)
        weight_scale: Per-group scale factors (float16)
        weight_shape: Original tensor shape (int64)
    """
    out_features, in_features = weight.shape
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int64)

    # Convert to float32 for computation
    w = weight.float()

    # Compute per-group scales (group along in_features dimension)
    num_groups = (in_features + group_size - 1) // group_size
    w_grouped = w.view(out_features, num_groups, -1)

    # Symmetric quantization: scale = max(abs(w)) / 7
    group_max = w_grouped.abs().amax(dim=-1)
    scale = group_max / 7.0
    scale = scale.clamp(min=1e-10)  # Avoid division by zero

    # Quantize: w_q = round(w / scale), clamp to [-8, 7]
    scale_expanded = scale.unsqueeze(-1).expand_as(w_grouped)
    w_q = (w_grouped / scale_expanded).round().clamp(-8, 7)

    # Convert signed [-8, 7] to unsigned [0, 15] for packing
    w_q = w_q.view(out_features, -1)[:, :in_features]
    w_q = torch.where(w_q < 0, w_q + 16, w_q).to(torch.uint8)

    # Pack 8 INT4 values into each int32 along the in_features dimension
    # HF format: [out_features, in_features//8] - 2D packed tensor
    assert in_features % 8 == 0, f"in_features must be divisible by 8, got {in_features}"

    # Reshape to [out_features, in_features//8, 8] for packing along dim 1
    w_q_grouped = w_q.view(out_features, in_features // 8, 8).to(torch.int32)

    # Pack 8 nibbles into 1 int32
    packed = torch.zeros(out_features, in_features // 8, dtype=torch.int32, device=weight.device)
    for i in range(8):
        packed |= (w_q_grouped[:, :, i] & 0xF) << (i * 4)

    weight_packed = packed.cpu()
    weight_scale = scale.to(torch.float16).cpu()

    return weight_packed, weight_scale, weight_shape

def is_quantized_expert_key(key: str) -> bool:
    if "mlp.experts." in key and ".weight" in key:
        if "shared_experts" in key:
            return False
        if ".layers.0." in key:
            return False
        return True
    return False
