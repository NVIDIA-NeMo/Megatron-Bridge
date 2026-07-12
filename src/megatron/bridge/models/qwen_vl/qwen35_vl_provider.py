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

"""HybridModel providers for dense and MoE Qwen3.5 vision-language models."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import transformers
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm, TERowParallelLinear
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from packaging.version import Version as PkgVersion

from megatron.bridge.models.qwen.qwen_hybrid import configure_qwen_hybrid_layers
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLHybridModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import get_vision_model_config
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import PatchMergerSubmodules
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider, Qwen3VLMoEModelProvider


_TRANSFORMERS_HAS_QWEN3_5_MOE = PkgVersion(transformers.__version__) >= PkgVersion("5.2.0")
if _TRANSFORMERS_HAS_QWEN3_5_MOE:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
else:
    Qwen3_5MoeVisionConfig = None  # type: ignore[assignment,misc]

try:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    _TRANSFORMERS_HAS_QWEN3_5 = True
except ImportError:
    _TRANSFORMERS_HAS_QWEN3_5 = False
    Qwen3_5VisionConfig = None  # type: ignore[assignment,misc]


_QWEN_VISUAL_ENCODER_KEY = "qwen_visual"
_IMAGES_MODALITY_KEY = "images"


def _check_qwen3_5_available() -> None:
    if not _TRANSFORMERS_HAS_QWEN3_5:
        raise ImportError(f"Qwen3.5 VL requires transformers support, but found {transformers.__version__}.")


def _check_qwen3_5_moe_available() -> None:
    if not _TRANSFORMERS_HAS_QWEN3_5_MOE:
        raise ImportError(f"Qwen3.5 VL MoE requires transformers >= 5.2.0, but found {transformers.__version__}.")


class _Qwen35VLMIMOAPI:
    """MIMO construction helpers shared by dense and MoE providers."""

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return the token ID associated with the image modality."""
        return {_IMAGES_MODALITY_KEY: self.image_token_id}

    def build_language_spec(self, vp_stage: int | None = None, pp_rank: int | None = None) -> ModuleSpec:
        """Build the Qwen multimodal Hybrid stack spec."""
        del vp_stage, pp_rank
        return self._resolve_hybrid_stack_spec()

    def build_mtp_spec(self, vp_stage: int | None = None) -> None:
        """Return no separate MTP spec because HybridModel carries it in the stack spec."""
        del vp_stage
        return None

    def build_vision_encoder_spec(self) -> ModuleSpec:
        """Build the Qwen3.5 vision encoder spec used by MIMO."""
        return _qwen35_build_vision_encoder_spec(self)

    def build_language_model_spec(self, pp_rank: int | None = 0) -> ModuleSpec:
        """Build the standalone Hybrid language-model spec used by MIMO."""
        self._ensure_hybrid_pattern()
        return ModuleSpec(
            module=Qwen3VLHybridModel,
            params={
                "config": self,
                "hybrid_stack_spec": self.build_language_spec(pp_rank=pp_rank),
                "vocab_size": self.vocab_size,
                "max_sequence_length": self.language_max_sequence_length,
                "hybrid_layer_pattern": self.hybrid_layer_pattern,
                "fp16_lm_cross_entropy": self.fp16_lm_cross_entropy,
                "parallel_output": True,
                "share_embeddings_and_output_weights": self.share_embeddings_and_output_weights,
                "rotary_percent": self.rotary_percent,
                "rotary_base": self.rotary_base,
                "scatter_embedding_sequence_parallel": False,
            },
        )


@dataclass
class Qwen35VLModelProvider(_Qwen35VLMIMOAPI, Qwen3VLModelProvider):
    """HybridModel provider for dense Qwen3.5 vision-language models."""

    modality_keys: ClassVar[dict[str, str]] = {_IMAGES_MODALITY_KEY: _QWEN_VISUAL_ENCODER_KEY}
    vision_config: Any = field(default=None)
    layernorm_zero_centered_gamma: bool = True
    attention_output_gate: bool = True
    experimental_attention_variant: str = "gated_delta_net"
    linear_attention_freq: int | list[int] = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    kv_channels: int | None = 256
    num_query_groups: int = 4
    rotary_base: float = 10000000.0
    rotary_percent: float = 0.25
    seq_length: int = 262144
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    bos_token_id: int = 248045
    eos_token_id: int = 248044
    deepstack_visual_indexes: list[int] = field(default_factory=list)
    hetereogenous_dist_checkpoint: bool = True

    def __post_init__(self) -> None:
        _check_qwen3_5_available()
        if self.vision_config is None:
            self.vision_config = Qwen3_5VisionConfig()
        super().__post_init__()

    def _ensure_hybrid_pattern(self) -> None:
        if self.hybrid_layer_pattern is None:
            if self.num_layers is None:
                raise ValueError("num_layers must be configured for Qwen3.5 VL")
            configure_qwen_hybrid_layers(
                self,
                num_logical_layers=self.num_layers,
                mlp_symbols=Symbols.MLP,
                linear_attention_freq=self.linear_attention_freq,
                mtp_mlp_symbol=Symbols.MLP,
            )

    def finalize(self) -> None:
        self._ensure_hybrid_pattern()
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        self._ensure_hybrid_pattern()
        return super().provide(pre_process, post_process, vp_stage)


@dataclass
class Qwen35VLMoEModelProvider(_Qwen35VLMIMOAPI, Qwen3VLMoEModelProvider):
    """HybridModel provider for MoE Qwen3.5 and Qwen3.6 vision-language models."""

    modality_keys: ClassVar[dict[str, str]] = {_IMAGES_MODALITY_KEY: _QWEN_VISUAL_ENCODER_KEY}
    vision_config: Any = field(default=None)
    layernorm_zero_centered_gamma: bool = True
    attention_output_gate: bool = True
    experimental_attention_variant: str = "gated_delta_net"
    linear_attention_freq: int | list[int] = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 64
    num_moe_experts: int = 512
    moe_router_topk: int = 10
    moe_shared_expert_gate: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_load_balancing_type: str = "global_aux_loss"
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_aux_loss_coeff: float = 1e-3
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    kv_channels: int | None = 256
    num_query_groups: int = 2
    rotary_base: float = 10000000.0
    rotary_percent: float = 0.25
    seq_length: int = 262144
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    bos_token_id: int = 248045
    eos_token_id: int = 248046
    deepstack_visual_indexes: list[int] = field(default_factory=list)
    hetereogenous_dist_checkpoint: bool = True

    def __post_init__(self) -> None:
        _check_qwen3_5_moe_available()
        if self.vision_config is None:
            self.vision_config = Qwen3_5MoeVisionConfig()
        super().__post_init__()

    def _ensure_hybrid_pattern(self) -> None:
        if self.hybrid_layer_pattern is None:
            if self.num_layers is None:
                raise ValueError("num_layers must be configured for Qwen3.5 VL MoE")
            configure_qwen_hybrid_layers(
                self,
                num_logical_layers=self.num_layers,
                mlp_symbols=Symbols.MOE,
                linear_attention_freq=self.linear_attention_freq,
                mtp_mlp_symbol=Symbols.MOE,
            )

    def finalize(self) -> None:
        self._ensure_hybrid_pattern()
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        self._ensure_hybrid_pattern()
        return super().provide(pre_process, post_process, vp_stage)


def _qwen35_build_vision_encoder_spec(provider) -> ModuleSpec:
    if getattr(provider, "use_hf_vision_model", False):
        raise ValueError("use_hf_vision_model is not supported for Qwen3VLVisionModel")
    vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
    vision_layer_spec.submodules.self_attention.module = Qwen3VLSelfAttention
    vision_config = get_vision_model_config(provider.vision_config, megatron_config=provider)
    vision_config.pipeline_model_parallel_size = 1
    vision_config.first_pipeline_num_layers = None
    return ModuleSpec(
        module=Qwen3VLVisionModel,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": vision_layer_spec,
            "patch_merger_spec": PatchMergerSubmodules(
                patch_norm=TENorm,
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
            "pre_process": True,
            "post_process": True,
        },
    )
