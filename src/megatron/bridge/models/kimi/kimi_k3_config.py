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

"""Architecture translation for the native Kimi K3 model."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from megatron.bridge.models.hybrid import BridgeHybridMLAModelConfig
from megatron.bridge.models.transformer_config import MLATransformerConfig


@dataclass
class KimiK3TransformerConfig(MLATransformerConfig):
    """MLA transformer config with the settings that Kimi K3's KDA, MLA and AttnRes layers read."""

    variable_seq_lengths: bool = True
    """A pipeline stage passes the AttnRes snapshot bank packed with its hidden states, so the tensor width
    depends on the stage boundary and pipeline communication exchanges shapes."""

    kimi_kda_layers: tuple[int, ...] = ()
    """1-based numbers of the layers that use Kimi Delta Attention (KDA); the other layers use MLA."""

    kimi_linear_num_heads: int = 96
    kimi_linear_head_dim: int = 128
    kimi_linear_conv_kernel_size: int = 4
    kimi_kda_gate_lower_bound: float = -5.0
    kimi_attn_res_block_size: int = 12
    """Number of layers whose outputs each AttnRes snapshot covers."""


def kimi_k3_layer_pattern(*, num_layers: int, kda_layers: Sequence[int], moe_layers: Sequence[int]) -> str:
    """Translate one-based KDA indices and a per-block MoE mask into layers.

    Each source block contributes its attention followed by its feed-forward
    layer. Depth and dense/expert placement always come from the source model.
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
    layers = pattern.replace("|", "")
    if len(layers) % 2 or any(layers[i] not in "K+" or layers[i + 1] not in "-E" for i in range(0, len(layers), 2)):
        raise ValueError("Kimi K3 requires alternating attention and feed-forward layers.")
    blocks = len(layers) // 2
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
        stages.append(layers[offset : offset + 2 * count])
        offset += 2 * count
    return "|".join(stages)


def kimi_k3_stack_spec(config):
    """Compose the existing Bridge Kimi components through HybridModel's stack API."""
    from megatron.core.transformer.spec_utils import ModuleSpec

    from megatron.bridge.models.kimi.modeling_kimi_k3.stack import KimiK3Stack

    if config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("Kimi K3 does not support virtual pipeline parallelism yet.")
    if config.context_parallel_size > 1 or config.dynamic_context_parallel:
        raise ValueError("Kimi K3 context parallelism is not supported yet.")
    if config.hidden_dropout != 0:
        raise ValueError("Kimi K3 requires hidden_dropout=0.")
    if config.cuda_graph_impl != "none":
        raise ValueError("Kimi K3 CUDA graphs are not supported yet.")
    unsupported = (
        "fp8",
        "fp4",
        "quant_recipe",
        "mtp_num_layers",
        "wide_residual",
        "enable_mhc_connections",
        "enable_attention_residuals",
        "cpu_offloading",
        "fine_grained_activation_offloading",
        "overlap_moe_expert_parallel_comm",
        "moe_shared_expert_overlap",
        "moe_shortcut_connection",
        "pipeline_model_parallel_layout",
        "num_layers_in_first_pipeline_stage",
        "num_layers_in_last_pipeline_stage",
    )
    for name in unsupported:
        if getattr(config, name, None):
            raise ValueError(
                f"Kimi K3 does not support {name}; use the documented unquantized training configuration."
            )
    validate_kimi_k3_geometry(config)
    return ModuleSpec(module=KimiK3Stack, params={"pattern": config.hybrid_layer_pattern})


def validate_kimi_k3_geometry(config) -> None:
    """Reject conflicting Kimi and Hybrid geometry before construction or export."""
    for kimi, native in (
        ("kimi_linear_num_heads", "linear_num_key_heads"),
        ("kimi_linear_num_heads", "linear_num_value_heads"),
        ("kimi_linear_head_dim", "linear_key_head_dim"),
        ("kimi_linear_head_dim", "linear_value_head_dim"),
        ("kimi_linear_conv_kernel_size", "linear_conv_kernel_dim"),
    ):
        if getattr(config, kimi) != getattr(config, native):
            raise ValueError(f"Kimi K3 {kimi} and {native} must agree.")


@dataclass(kw_only=True)
class KimiK3ModelConfig(BridgeHybridMLAModelConfig):
    """HF-derived layers with Bridge-owned Kimi computation and depth state."""

    transformer_config_class: ClassVar[type] = KimiK3TransformerConfig
    hybrid_stack_spec: Any = kimi_k3_stack_spec
    pipeline_block_counts: tuple[int, ...] | None = None

    def finalize(self) -> None:
        """Partition complete source blocks before finalizing the Hybrid stack."""
        self.hybrid_layer_pattern = kimi_k3_pipeline_pattern(
            self.hybrid_layer_pattern,
            pipeline_size=self.pipeline_model_parallel_size,
            blocks_per_stage=self.pipeline_block_counts,
        )
        super().finalize()
