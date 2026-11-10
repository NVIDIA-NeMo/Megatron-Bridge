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

"""
Qwen3.5 and Qwen3.6 VL model-provider configurations for Megatron-Core.

Qwen3.5 and Qwen3.6 share a vision-language architecture that combines:
- A hybrid Gated DeltaNet (GDN) + Gated Attention language model (like Qwen3-Next)
- A vision encoder (similar to Qwen3-VL)
- Dense MLP or Mixture of Experts (MoE) with shared experts

This module provides three model providers:

- ``Qwen35VLModelProvider``: Dense variant (e.g., Qwen3.5-27B)
  Reference: https://huggingface.co/Qwen/Qwen3.5-27B

- ``Qwen35TokenClassificationModelProvider``: Dense token-classification variant

- ``Qwen35VLMoEModelProvider``: Qwen3.5/Qwen3.6 MoE variants
  Reference: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
  Reference: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable, ClassVar, List, Literal, Optional, cast

import transformers
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from packaging.version import Version as PkgVersion
from transformers import PretrainedConfig


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

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForTokenClassification  # noqa: F401

    _TRANSFORMERS_HAS_QWEN3_5_TOKEN_CLASSIFICATION = True
except ImportError:
    _TRANSFORMERS_HAS_QWEN3_5_TOKEN_CLASSIFICATION = False

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec

from megatron.bridge.models.common.heads import LinearForLastLayer
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen.modeling_qwen35.hybrid import configure_qwen35_hybrid
from megatron.bridge.models.qwen.modeling_qwen35.model_config import prepare_qwen35_pipeline_config
from megatron.bridge.models.qwen_vl.modeling_qwen35.hybrid_model import (
    Qwen35HybridForTokenClassification,
    Qwen35VLHybridLanguageModel,
    qwen35_vl_hybrid_stack_spec,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.token_classification import Qwen3VLForTokenClassification
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import get_vision_model_config
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import PatchMergerSubmodules
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel
from megatron.bridge.models.qwen_vl.qwen35_vl_builder import Qwen35VLModelBuilder
from megatron.bridge.models.qwen_vl.qwen35_vl_config import Qwen35VLModelConfig
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


_QWEN_VISUAL_ENCODER_KEY = "qwen_visual"
_IMAGES_MODALITY_KEY = "images"


def _check_qwen3_5_available() -> None:
    """Raise a clear error if transformers doesn't have qwen3_5 (dense) support."""
    if not _TRANSFORMERS_HAS_QWEN3_5:
        raise ImportError(
            f"Qwen3.5 VL (dense) requires transformers with qwen3_5 model support, "
            f"but found {transformers.__version__}. "
            "Please upgrade: pip install --upgrade transformers"
        )


def _check_qwen3_5_moe_available() -> None:
    """Raise a clear error if transformers doesn't have qwen3_5_moe support."""
    if not _TRANSFORMERS_HAS_QWEN3_5_MOE:
        raise ImportError(
            f"Qwen3.5/3.6 VL (MoE) requires transformers with qwen3_5_moe support, "
            f"but found {transformers.__version__}. "
            "Please upgrade: pip install --upgrade transformers"
        )


@dataclass
class Qwen35VLModelProvider(GPTModelProvider):
    """
    Model provider for Qwen3.5 VL Dense (Vision-Language) Models.

    Qwen3.5 dense combines a hybrid GDN (Gated DeltaNet) + Gated Attention language
    model architecture with a vision encoder (similar to Qwen3-VL) and a standard
    dense MLP (no Mixture of Experts).

    Key Architecture Details (27B):
    - 64 layers: 16 groups x (3 GDN + 1 Attention)
    - Hidden dim: 5120, Intermediate dim: 17408
    - GDN: 16 QK heads, 48 V heads, head_dim=128
    - Gated Attention: 24 Q heads, 4 KV heads, head_dim=256
    - Vision: depth=27, hidden=1152, no deepstack
    - mRoPE with sections [11, 11, 10], rope_theta=10,000,000
    - partial_rotary_factor=0.25
    """

    modality_keys: ClassVar[dict[str, str]] = {_IMAGES_MODALITY_KEY: _QWEN_VISUAL_ENCODER_KEY}
    MODEL_CLASS: ClassVar[type[Qwen3VLModel] | None] = None
    # The MIMO language spec wires the MTP block, so MIMO keeps MTP layers instead of dropping them.
    mimo_supports_mtp: ClassVar[bool] = True

    # =========================================================================
    # Hybrid Architecture (Qwen3-Next style)
    # =========================================================================
    transformer_layer_spec: ModuleSpec | Callable[["GPTModelProvider"], ModuleSpec] = qwen35_vl_hybrid_stack_spec
    layernorm_zero_centered_gamma: bool = True
    attention_output_gate: bool = True
    experimental_attention_variant: str = "gdn"
    linear_attention_freq: int | list[int] = 4

    # --- Gated DeltaNet (GDN) parameters ---
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48

    # =========================================================================
    # Common LLM parameters
    # =========================================================================
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    kv_channels: int | None = 256
    num_query_groups: int = 4
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    attention_softmax_in_fp32: bool = True
    rotary_base: float = 10000000.0
    rotary_percent: float = 0.25
    seq_length: int = 262144

    # =========================================================================
    # VL-specific parameters
    # =========================================================================
    vision_config: Any = field(default=None)
    position_embedding_type: str = "mrope"
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    apply_rotary_pos_emb_in_fp32: bool = False

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    bos_token_id: int = 248045
    eos_token_id: int = 248044

    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    patch_size: int = 16
    language_max_sequence_length: int = 2048
    scatter_embedding_sequence_parallel: bool = False

    # =========================================================================
    # Freeze options for fine-tuning
    # =========================================================================
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    # =========================================================================
    # Performance
    # =========================================================================
    bias_activation_fusion: bool = True
    use_hf_vision_model: bool = False
    vision_dp_when_cp: bool = False
    # "inherit" preserves legacy decoder-policy inheritance; None disables vision recompute.
    vision_recompute_granularity: Literal["inherit", "full", "selective"] | None = "inherit"
    vision_recompute_method: Literal["uniform", "block"] | None = None
    vision_recompute_num_layers: int | None = None
    vision_recompute_modules: list[str] | None = None
    hetereogenous_dist_checkpoint: bool = True

    mtp_num_layers: Optional[int] = None

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return modality token ids."""
        return {_IMAGES_MODALITY_KEY: self.image_token_id}

    def __post_init__(self):
        _check_qwen3_5_available()
        if self.vision_config is None:
            self.vision_config = Qwen3_5VisionConfig()
        super().__post_init__()

    def finalize(self) -> None:
        if (self.context_parallel_size or 1) > 1:
            self.calculate_per_token_loss = True
        super().finalize()

    def build_language_spec(
        self, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
    ) -> TransformerBlockSubmodules | ModuleSpec:
        """Build the language transformer-block spec."""
        return _qwen35_build_language_block_spec(self, vp_stage, pp_rank)

    def build_mtp_spec(
        self, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
    ) -> Optional[TransformerBlockSubmodules | ModuleSpec]:
        """Build the MTP block spec."""
        return _qwen35_build_mtp_block_spec(self, vp_stage, pp_rank)

    def build_vision_encoder_spec(self) -> ModuleSpec:
        """Build the vision encoder spec."""
        return _qwen35_build_vision_encoder_spec(self)

    def build_language_model_spec(self, pp_rank: Optional[int] = 0) -> ModuleSpec:
        """Build the language model spec."""
        return _qwen35_build_language_model_spec(self, pp_rank=pp_rank)

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Qwen3VLModel:
        """Adapt the legacy serialized provider to the shared Hybrid builder."""
        config = _qwen35_provider_model_config(self)
        builder = Qwen35VLModelBuilder(config)
        builder.model_class = getattr(self, "MODEL_CLASS", None)
        pg = self._pg_collection or ProcessGroupCollection.use_mpu_process_groups()
        return builder.build_model(pg, pre_process, post_process, vp_stage)

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> HybridModel:
        """Provide just the language model component without vision."""
        return _provide_qwen35_language_model(self, pre_process, post_process, vp_stage)


@dataclass
class Qwen35TokenClassificationModelProvider(Qwen35VLModelProvider):
    """Serializable provider for Qwen3.5 VL token-classification models."""

    MODEL_CLASS: ClassVar[type[Qwen3VLModel]] = Qwen35HybridForTokenClassification

    num_labels: int | None = None
    classifier_dropout: float = 0.1
    mtp_num_layers: int | None = 0
    mtp_enabled: bool = False
    share_embeddings_and_output_weights: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

    def provide(
        self,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Qwen3VLForTokenClassification:
        """Build a Qwen3.5 VL model with a persistent replicated classifier head.

        Args:
            pre_process: Whether this pipeline stage owns input preprocessing.
            post_process: Whether this pipeline stage owns output postprocessing.
            vp_stage: Virtual-pipeline stage index, when virtual pipelining is enabled.

        Returns:
            The Megatron Qwen3.5 VL token-classification model for this stage.

        Raises:
            ValueError: If ``num_labels`` is not positive or ``logit_dtype`` is configured.
        """
        if self.num_labels is None or self.num_labels <= 0:
            raise ValueError(f"num_labels must be a positive integer, got {self.num_labels!r}.")
        if self.logit_dtype is not None:
            raise ValueError(
                "Qwen3.5 token classification replaces the language-model output layer with a replicated "
                "classification head, which does not support true mixed-precision output GEMMs. "
                "Set logit_dtype=None."
            )

        model = cast(
            Qwen3VLForTokenClassification,
            super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage),
        )
        if model.language_model is not None and model.language_model.post_process:
            model.language_model.output_layer = LinearForLastLayer(
                input_size=self.hidden_size,
                output_size=self.num_labels,
                sequence_parallel=self.sequence_parallel,
                bias=True,
                dropout=self.classifier_dropout,
                output_in_fp32=False,
                tp_group=getattr(getattr(model, "pg_collection", None), "tp", None),
                init_method=self.init_method,
                perform_initialization=self.perform_initialization,
            )
        return model


@dataclass
class Qwen35VLMoEModelProvider(GPTModelProvider):
    """
    Model provider for Qwen3.5 and Qwen3.6 VL MoE models.

    Qwen3.5 and Qwen3.6 combine a hybrid GDN (Gated DeltaNet) + Gated
    Attention language architecture (like Qwen3-Next) with a vision encoder
    (similar to Qwen3-VL) and Mixture of Experts (MoE) with shared experts.
    Model-specific dimensions and MoE geometry are populated from the
    Hugging Face config.

    Default Architecture Details (Qwen3.5-397B-A17B):
    - 60 layers: 15 groups × (3 GDN-MoE + 1 Attention-MoE)
    - Hidden dim: 4096, Token Embedding: 248320
    - GDN: 16 QK heads, 64 V heads, head_dim=128
    - Gated Attention: 32 Q heads, 2 KV heads, head_dim=256
    - MoE: 512 experts, 10 routed + 1 shared, expert dim=1024
    - mRoPE with sections [11, 11, 10], rope_theta=10,000,000
    - partial_rotary_factor=0.25

    Note: num_query_groups corresponds to num_key_value_heads in HF config (for
    standard Gated Attention layers). GDN layers have separate head counts.
    """

    modality_keys: ClassVar[dict[str, str]] = {_IMAGES_MODALITY_KEY: _QWEN_VISUAL_ENCODER_KEY}
    # The MIMO language spec wires the MTP block, so MIMO keeps MTP layers instead of dropping them.
    mimo_supports_mtp: ClassVar[bool] = True

    # =========================================================================
    # Hybrid Architecture (Qwen3-Next style)
    # =========================================================================
    transformer_layer_spec: ModuleSpec | Callable[["GPTModelProvider"], ModuleSpec] = qwen35_vl_hybrid_stack_spec
    layernorm_zero_centered_gamma: bool = True
    attention_output_gate: bool = True
    experimental_attention_variant: str = "gdn"
    linear_attention_freq: int | list[int] = 4  # 1 standard attention per 4 layers

    # --- Gated DeltaNet (GDN) parameters ---
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 64  # 64 V heads for GDN in 397B model

    # =========================================================================
    # MoE parameters
    # =========================================================================
    num_moe_experts: int = 512
    moe_router_topk: int = 10  # 10 routed experts per token
    moe_shared_expert_gate: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_load_balancing_type: str = "global_aux_loss"
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_aux_loss_coeff: float = 1e-3

    # =========================================================================
    # Common LLM parameters
    # =========================================================================
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    kv_channels: int | None = 256  # head_dim for standard Gated Attention
    num_query_groups: int = 2  # KV heads for standard Gated Attention
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    attention_softmax_in_fp32: bool = True
    rotary_base: float = 10000000.0  # rope_theta from HF config
    rotary_percent: float = 0.25  # partial_rotary_factor from HF config
    seq_length: int = 262144  # 262K native context length

    # =========================================================================
    # VL-specific parameters
    # =========================================================================

    vision_config: Any = field(default=None)

    # Position embedding: Qwen3.5/3.6 use multimodal rope (mRoPE).
    position_embedding_type: str = "mrope"
    # Qwen3.5/3.6 mRoPE is [11, 11, 10] (Qwen3-VL uses [24, 20, 20]).
    # because partial_rotary_factor=0.25, so RoPE dim = 256*0.25 = 64, with sections [11,11,10]
    # for [temporal, height, width] summing to 32 (half of 64 rotary dim).
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    apply_rotary_pos_emb_in_fp32: bool = False

    # Vision-Language token IDs
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    bos_token_id: int = 248045
    eos_token_id: int = 248046

    # Vision model settings
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    patch_size: int = 16
    language_max_sequence_length: int = 2048

    scatter_embedding_sequence_parallel: bool = False

    # =========================================================================
    # Freeze options for fine-tuning
    # =========================================================================
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    # =========================================================================
    # Performance
    # =========================================================================
    bias_activation_fusion: bool = True
    use_hf_vision_model: bool = False
    vision_dp_when_cp: bool = False
    # "inherit" preserves legacy decoder-policy inheritance; None disables vision recompute.
    vision_recompute_granularity: Literal["inherit", "full", "selective"] | None = "inherit"
    vision_recompute_method: Literal["uniform", "block"] | None = None
    vision_recompute_num_layers: int | None = None
    vision_recompute_modules: list[str] | None = None

    # Vision encoder CUDA graph settings
    vision_cuda_graph_impl: str = "none"
    vision_cuda_graph_scope: List[str] = field(default_factory=list)
    max_vision_cuda_graph_seq_length: Optional[int] = None

    # Heterogeneous dist checkpoint (needed for hybrid architecture)
    hetereogenous_dist_checkpoint: bool = True

    mtp_num_layers: Optional[int] = None

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return modality token ids."""
        return {_IMAGES_MODALITY_KEY: self.image_token_id}

    def __post_init__(self):
        _check_qwen3_5_moe_available()
        if self.vision_config is None:
            self.vision_config = Qwen3_5MoeVisionConfig()
        super().__post_init__()

    def finalize(self) -> None:
        if (self.context_parallel_size or 1) > 1:
            self.calculate_per_token_loss = True
        super().finalize()

    def build_language_spec(
        self, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
    ) -> TransformerBlockSubmodules | ModuleSpec:
        """Build the language transformer-block spec."""
        return _qwen35_build_language_block_spec(self, vp_stage, pp_rank)

    def build_mtp_spec(
        self, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
    ) -> Optional[TransformerBlockSubmodules | ModuleSpec]:
        """Build the MTP block spec (or ``None`` when MTP is disabled)."""
        return _qwen35_build_mtp_block_spec(self, vp_stage, pp_rank)

    def build_vision_encoder_spec(self) -> ModuleSpec:
        """Build the vision encoder spec."""
        return _qwen35_build_vision_encoder_spec(self)

    def build_language_model_spec(self, pp_rank: Optional[int] = 0) -> ModuleSpec:
        """Build the language model spec."""
        return _qwen35_build_language_model_spec(self, pp_rank=pp_rank)

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Qwen3VLModel:
        """Adapt the legacy serialized provider to the shared Hybrid builder."""
        config = _qwen35_provider_model_config(self)
        builder = Qwen35VLModelBuilder(config)
        builder.model_class = getattr(self, "MODEL_CLASS", None)
        pg = self._pg_collection or ProcessGroupCollection.use_mpu_process_groups()
        return builder.build_model(pg, pre_process, post_process, vp_stage)

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> HybridModel:
        """Provide just the language model component without vision."""
        return _provide_qwen35_language_model(self, pre_process, post_process, vp_stage)


def _provide_qwen35_language_model(provider, pre_process, post_process, vp_stage):
    pg = provider._pg_collection or ProcessGroupCollection.use_mpu_process_groups()
    vp_size = provider.virtual_pipeline_model_parallel_size
    if pre_process is None:
        pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg.pp)
    if post_process is None:
        post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg.pp)

    assert provider.vocab_size is not None, "vocab_size must be configured before calling provide_language_model()"
    spec = provider.build_language_model_spec()
    if provider.should_pad_vocab:
        spec.params["vocab_size"] = calculate_padded_vocab_size(
            provider.vocab_size, provider.make_vocab_size_divisible_by, provider.tensor_model_parallel_size
        )

    return build_module(
        spec,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
        pg_collection=pg,
    )


def _qwen35_build_language_block_spec(
    config: TransformerConfig,
    vp_stage: Optional[int],
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules | ModuleSpec:
    """Build the language transformer-block spec."""
    return qwen35_vl_hybrid_stack_spec()


def _qwen35_build_mtp_block_spec(
    provider: GPTModelProvider,
    vp_stage: Optional[int],
    pp_rank: Optional[int] = None,
) -> Optional[TransformerBlockSubmodules | ModuleSpec]:
    """Build the MTP block spec."""
    # HybridModel builds the nested MTP stack from its unified layer pattern.
    return None


def _qwen35_provider_model_config(provider: GPTModelProvider) -> Qwen35VLModelConfig:
    """Translate legacy HF-block fields once, at the deprecated provider boundary."""
    outer_fields = {item.name for item in fields(Qwen35VLModelConfig) if item.init} - {"transformer"}
    transformer_class = Qwen35VLModelConfig.transformer_config_class
    transformer_fields = {item.name for item in fields(transformer_class) if item.init}
    kwargs = {name: getattr(provider, name) for name in outer_fields | transformer_fields if hasattr(provider, name)}
    kwargs.pop("transformer_layer_spec", None)
    configure_qwen35_hybrid(
        kwargs,
        PretrainedConfig(
            num_hidden_layers=provider.num_layers,
            num_experts=provider.num_moe_experts,
            full_attention_interval=provider.linear_attention_freq,
        ),
    )
    kwargs["hybrid_stack_spec"] = qwen35_vl_hybrid_stack_spec
    outer = {name: value for name, value in kwargs.items() if name in outer_fields}
    transformer = {name: value for name, value in kwargs.items() if name not in outer_fields}
    # Legacy pipeline layouts count HF blocks; canonical configs count Hybrid positions.
    for name in ("num_layers_in_first_pipeline_stage", "num_layers_in_last_pipeline_stage"):
        if transformer.get(name) is not None:
            transformer[name] *= 2
    if transformer.get("pipeline_model_parallel_layout") is not None:
        from megatron.core.transformer.enums import LayerType
        from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

        layout = transformer["pipeline_model_parallel_layout"]
        if not isinstance(layout, PipelineParallelLayerLayout):
            layout = PipelineParallelLayerLayout(layout, provider.pipeline_model_parallel_size)
        expanded = [
            [entry for layer in stage for entry in ([layer, layer] if layer == LayerType.decoder else [layer])]
            for vp_stage in range(layout.virtual_pipeline_model_parallel_size)
            for stages in layout.layout
            for stage in [stages[vp_stage]]
        ]
        transformer["pipeline_model_parallel_layout"] = PipelineParallelLayerLayout(
            expanded, provider.pipeline_model_parallel_size
        )
    return Qwen35VLModelConfig(transformer=transformer_class(**transformer), **outer)


def _qwen35_build_language_model_spec(provider: GPTModelProvider, pp_rank: Optional[int] = 0) -> ModuleSpec:
    """Build the MIMO decoder spec from the same normalized Hybrid config as full VL."""
    config = prepare_qwen35_pipeline_config(_qwen35_provider_model_config(provider))
    return ModuleSpec(
        module=Qwen35VLHybridLanguageModel,
        params={
            "config": config.transformer,
            "transformer_layer_spec": provider.build_language_spec(pp_rank=pp_rank),
            "hybrid_layer_pattern": config.hybrid_layer_pattern,
            "vocab_size": config.vocab_size,
            "max_sequence_length": config.language_max_sequence_length,
            "fp16_lm_cross_entropy": config.fp16_lm_cross_entropy,
            "logit_dtype": config.logit_dtype,
            "parallel_output": True,
            "share_embeddings_and_output_weights": config.share_embeddings_and_output_weights,
            "position_embedding_type": "mrope",
            "rotary_percent": config.rotary_percent,
            "rotary_base": config.rotary_base,
            "scatter_embedding_sequence_parallel": False,
            "mtp_block_spec": provider.build_mtp_spec(pp_rank=pp_rank),
        },
    )


def _qwen35_build_vision_encoder_spec(provider: GPTModelProvider) -> ModuleSpec:
    """Build the Qwen3VL vision encoder spec."""
    if getattr(provider, "use_hf_vision_model", False):
        raise ValueError("use_hf_vision_model is not supported for Qwen3VLVisionModel")

    vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
    vision_transformer_layer_spec.submodules.self_attention.module = Qwen3VLSelfAttention

    megatron_vision_transformer_config = get_vision_model_config(provider.vision_config, megatron_config=provider)
    megatron_vision_transformer_config.pipeline_model_parallel_size = 1
    megatron_vision_transformer_config.first_pipeline_num_layers = None

    return ModuleSpec(
        module=Qwen3VLVisionModel,
        params={
            "transformer_config": megatron_vision_transformer_config,
            "transformer_layer_spec": vision_transformer_layer_spec,
            "patch_merger_spec": PatchMergerSubmodules(
                patch_norm=TENorm,
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
            "pre_process": True,
            "post_process": True,
        },
    )


def _patch_standard_attention_specs(
    block_spec: Optional[TransformerBlockSubmodules | ModuleSpec],
    attention_cls,
) -> None:
    """Selectively replace standard self-attention specs with ``attention_cls``.

    This handles both the main decoder block spec and the nested TransformerLayer
    spec stored inside MTP block specs.

    In a hybrid block spec, each layer spec has a different self_attention submodule:
    - Standard attention layers have a ``SelfAttention``-like module.
    - GDN layers have a ``GatedDeltaNet``-like module.

    This function patches only the standard attention layers with *attention_cls*
    (e.g. ``Qwen3VLSelfAttention`` for mRoPE support), leaving GDN layers unchanged.

    Detection heuristic: GDN layer specs have ``GatedDeltaNet`` (or similar) as the
    self_attention module, which does NOT have a ``linear_qkv`` submodule. Standard
    attention specs DO have ``linear_qkv``. We use this to distinguish them.
    """
    from megatron.core.transformer.attention import SelfAttention

    if block_spec is None:
        return

    if hasattr(block_spec, "layer_specs"):
        for layer_spec in block_spec.layer_specs:
            _patch_standard_attention_specs(layer_spec, attention_cls)
        return

    if not isinstance(block_spec, ModuleSpec):
        return

    submodules = getattr(block_spec, "submodules", None)
    if submodules is None:
        return

    if hasattr(submodules, "mtp_model_layer"):
        _patch_standard_attention_specs(submodules.mtp_model_layer, attention_cls)

    if not hasattr(submodules, "self_attention"):
        return

    attn_spec = submodules.self_attention
    # Standard attention specs use SelfAttention (or a subclass) as the module.
    # and have linear_qkv in their submodules. GDN specs use GatedDeltaNet.
    if attn_spec.module is SelfAttention or (
        isinstance(attn_spec.module, type) and issubclass(attn_spec.module, SelfAttention)
    ):
        attn_spec.module = attention_cls
