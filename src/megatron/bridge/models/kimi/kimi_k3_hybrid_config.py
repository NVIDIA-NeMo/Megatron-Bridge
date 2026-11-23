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

"""Architecture translation for the native Kimi K3 Hybrid model."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from functools import partial
from typing import Any

from megatron.bridge.models.hybrid import BridgeHybridMLAModelConfig
from megatron.bridge.models.transformer_config import MLATransformerConfig
from megatron.bridge.utils.activation_map import str_to_callable


def kimi_k3_hybrid_pattern(*, num_layers: int, kda_layers: Sequence[int], moe_layers: Sequence[int]) -> str:
    """Translate one-based KDA indices and a per-block MoE mask into native entries.

    Each source block contributes its attention followed by its feed-forward
    entry. Depth and dense/expert placement always come from the source model.
    """
    if num_layers < 1 or len(moe_layers) != num_layers:
        raise ValueError("Kimi K3 needs a positive block count and one MoE flag per block.")
    if len(set(kda_layers)) != len(kda_layers) or any(layer < 1 or layer > num_layers for layer in kda_layers):
        raise ValueError("Kimi K3 KDA indices must be unique one-based indices within the source blocks.")
    if any(flag not in (0, 1) for flag in moe_layers):
        raise ValueError("Kimi K3 MoE flags must be zero or one.")
    kda = set(kda_layers)
    return "".join(("K" if i + 1 in kda else "+") + ("E" if moe else "-") for i, moe in enumerate(moe_layers))


def kimi_k3_pipeline_pattern(
    pattern: str, *, pipeline_size: int, blocks_per_stage: tuple[int, ...] | None = None
) -> str:
    """Partition complete HF blocks into native pipeline stages, preserving block order."""
    entries = pattern.replace("|", "")
    if len(entries) % 2 or any(
        entries[i] not in "K+" or entries[i + 1] not in "-E" for i in range(0, len(entries), 2)
    ):
        raise ValueError("Kimi K3 requires alternating attention and feed-forward entries.")
    blocks = len(entries) // 2
    if pipeline_size < 1 or pipeline_size > blocks:
        raise ValueError("Kimi K3 requires at least one complete block per pipeline stage.")
    if blocks_per_stage is None:
        if pattern.count("|") + 1 == pipeline_size:
            if any(not stage or len(stage) % 2 for stage in pattern.split("|")):
                raise ValueError("Kimi K3 pipeline boundaries must separate complete source blocks.")
            return pattern
        quotient, remainder = divmod(blocks, pipeline_size)
        blocks_per_stage = tuple(quotient + int(rank < remainder) for rank in range(pipeline_size))
    if (
        len(blocks_per_stage) != pipeline_size
        or sum(blocks_per_stage) != blocks
        or any(count < 1 for count in blocks_per_stage)
    ):
        raise ValueError("Kimi K3 pipeline block counts must cover every block exactly once.")
    offset = 0
    stages = []
    for count in blocks_per_stage:
        stages.append(entries[offset : offset + 2 * count])
        offset += 2 * count
    return "|".join(stages)


def native_kimi_k3_kwargs(source: dict[str, Any]) -> dict[str, Any]:
    """Translate the existing HF architecture mapping to native Core field names.

    Router policy, precision, vocabulary, and execution settings are inherited
    from the source mapping. This does not select a benchmark architecture.
    """
    required = {
        "enable_attention_residuals",
        "kda_f_lora_rank",
        "moe_use_norm_before_up_proj",
        "kda_beta_proj_column_parallel",
        "kda_split_qkv_projections",
        "kda_conv_in_fp32",
        "kda_fused_norm_gate",
        "kda_qk_norm_in_kernel",
        "kda_transpose_state_layout",
        "scale_attention_output_init",
        "attn_res_query_init",
        "attn_res_query_as_matrix",
        "attn_res_fp32_params",
        "attn_res_single_source_param_grad",
        "mla_output_gate_sigmoid_fp32",
    }
    if not required.issubset({field.name for field in fields(MLATransformerConfig)}):
        raise RuntimeError(
            "Native Kimi K3 requires Core dev with KDA separate-projection, precision, and "
            "AttnRes parameter-layout controls; the unmodified pinned dev revision is insufficient."
        )
    result = dict(source)
    layers = result["num_layers"]
    pattern = kimi_k3_hybrid_pattern(
        num_layers=layers, kda_layers=result.pop("kimi_kda_layers"), moe_layers=result.pop("moe_layer_freq")
    )
    block_size = result.pop("kimi_attn_res_block_size")
    if block_size < 1:
        raise ValueError("Kimi K3 AttnRes block size must be positive.")
    heads = result.pop("kimi_linear_num_heads")
    head_dim = result.pop("kimi_linear_head_dim")
    result.pop("transformer_layer_spec")
    # The GPT-only RoPE scaling control has no role in K3's no-RoPE model.
    result.pop("rope_scaling_factor")
    result.update(
        num_layers=2 * layers,
        hybrid_layer_pattern=pattern,
        is_hybrid_model=True,
        experimental_attention_variant="kda",
        linear_conv_kernel_dim=result.pop("kimi_linear_conv_kernel_size"),
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=heads,
        linear_num_value_heads=heads,
        kda_f_lora_rank=head_dim,
        kda_gate_lora_rank=None,
        kda_safe_gate=True,
        kda_beta_proj_column_parallel=True,
        kda_split_qkv_projections=True,
        kda_conv_in_fp32=True,
        kda_fused_norm_gate=True,
        kda_qk_norm_in_kernel=True,
        kda_transpose_state_layout=True,
        scale_attention_output_init=False,
        kda_lower_bound=result.pop("kimi_kda_gate_lower_bound"),
        enable_attention_residuals=True,
        attn_res_block_layers=2 * block_size,
        attn_res_impl="torch",
        attn_res_norm_fusion=False,
        attn_res_fp32_params=False,
        attn_res_single_source_param_grad=False,
        attn_res_query_as_matrix=True,
        attn_res_query_init="linear",
        no_rope_freq=1,
        attention_latent_norm_epsilon=result["layernorm_epsilon"],
        attention_output_gate=True,
        gated_attention_proj_granularity="elementwise",
        attention_output_gate_fusion=False,
        mla_output_gate_sigmoid_fp32=False,
        mla_down_proj_fusion=False,
        moe_use_norm_before_up_proj=True,
        moe_latent_up_projection_rmsnorm=False,
        activation_func=str_to_callable("situlu"),
        use_te_activation_func=False,
        gdn_pre_gated_delta_rule_fusion=False,
    )
    return result


def kimi_k3_hybrid_stack_spec(config):
    """Select native Kimi modules with the checkpoint's latent-norm cast boundary."""
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.ops.norm.torch_norm import TorchRMSNorm

    if config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("Kimi K3 does not support virtual pipeline parallelism yet.")
    if (config.context_parallel_size > 1 or config.dynamic_context_parallel) and "K" in config.hybrid_layer_pattern:
        raise ValueError("Kimi K3 KDA context parallelism is not supported yet.")
    if config.hidden_dropout != 0:
        raise ValueError("Kimi K3 requires hidden_dropout=0.")
    spec = deepcopy(hybrid_stack_spec)
    norm = partial(TorchRMSNorm, cast_before_affine=True)
    attention = spec.submodules.mla_layer.submodules.self_attention.submodules
    attention.q_layernorm = norm
    attention.kv_layernorm = norm
    moe = get_moe_module_spec(
        use_te=True,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
    )
    moe.keywords["submodules"].fc2_norm = norm
    spec.submodules.moe_layer.submodules.mlp = moe
    return spec


@dataclass(kw_only=True)
class KimiK3HybridModelConfig(BridgeHybridMLAModelConfig):
    """Native KDA/MLA and dense/MoE entries with Kimi attention residuals."""

    hybrid_stack_spec: Any = kimi_k3_hybrid_stack_spec
    pipeline_block_counts: tuple[int, ...] | None = None

    def finalize(self) -> None:
        """Partition complete source blocks before finalizing the native stack."""
        self.hybrid_layer_pattern = kimi_k3_pipeline_pattern(
            self.hybrid_layer_pattern,
            pipeline_size=self.pipeline_model_parallel_size,
            blocks_per_stage=self.pipeline_block_counts,
        )
        super().finalize()
