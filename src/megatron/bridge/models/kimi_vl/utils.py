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
