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
Qwen3.5 VL Model Provider configurations for Megatron-Core.

Qwen3.5 is a family of vision-language models that combine:
- A hybrid Gated DeltaNet (GDN) + Gated Attention language model (like Qwen3-Next)
- A vision encoder (similar to Qwen3-VL)
- Dense MLP or Mixture of Experts (MoE) with shared experts

This module provides two model providers:

- ``Qwen35VLModelProvider``: Dense variant (e.g., Qwen3.5-27B)
  Reference: https://huggingface.co/Qwen/Qwen3.5-27B

- ``Qwen35VLMoEModelProvider``: MoE variant (e.g., Qwen3.5-397B-A17B)
  Reference: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import transformers
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from packaging.version import Version as PkgVersion


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

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel


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
            f"Qwen3.5 VL (MoE) requires transformers >= 5.2.0, but found {transformers.__version__}. "
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

    # =========================================================================
    # Hybrid Architecture (Qwen3-Next style)
    # =========================================================================
    transformer_layer_spec: ModuleSpec | Callable[["GPTModelProvider"], ModuleSpec] = (
        get_transformer_block_with_experimental_attention_variant_spec
    )
    layernorm_zero_centered_gamma: bool = True
    attention_output_gate: bool = True
    experimental_attention_variant: str = "gated_delta_net"
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
    hetereogenous_dist_checkpoint: bool = True

    mtp_num_layers: Optional[int] = None

    def __post_init__(self):
        _check_qwen3_5_available()
        if self.vision_config is None:
            self.vision_config = Qwen3_5VisionConfig()
        if self.num_query_groups < self.tensor_model_parallel_size:
            raise ValueError(
                f"TP size {self.tensor_model_parallel_size} should be less than or equal to num_query_groups {self.num_query_groups}. Please use a smaller TP size."
            )
        super().__post_init__()

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Qwen3VLModel:
        """Provide a Qwen3.5 VL dense model instance with vision and language components."""
        language_transformer_config = self
        hf_vision_config = self.vision_config

        block_spec = get_transformer_block_with_experimental_attention_variant_spec(
            language_transformer_config,
            vp_stage=vp_stage,
        )
        _patch_standard_attention_specs(block_spec, Qwen3VLSelfAttention)

        model = Qwen3VLModel(
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=block_spec,
            vision_transformer_config=hf_vision_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self._pg_collection,
        )

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Provide just the language model component without vision."""
        return GPTModelProvider.provide(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


@dataclass
class Qwen35VLMoEModelProvider(GPTModelProvider):
    """
    Model provider for Qwen 3.5 VL (Vision-Language) Models.

    Qwen 3.5 combines a hybrid GDN (Gated DeltaNet) + Gated Attention language model
    architecture (like Qwen3-Next) with a vision encoder (similar to Qwen3-VL) and
    Mixture of Experts (MoE) with shared experts.

    Key Architecture Details (397B-A17B):
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

    # =========================================================================
    # Hybrid Architecture (Qwen3-Next style)
    # =========================================================================
    transformer_layer_spec: ModuleSpec | Callable[["GPTModelProvider"], ModuleSpec] = (
        get_transformer_block_with_experimental_attention_variant_spec
    )
    layernorm_zero_centered_gamma: bool = True
    attention_output_gate: bool = True
    experimental_attention_variant: str = "gated_delta_net"
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

    # Position embedding: Qwen3.5 uses multimodal rope (mRoPE)
    position_embedding_type: str = "mrope"
    # Qwen3.5 mRoPE section is [11, 11, 10] (different from Qwen3-VL's [24, 20, 20])
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

    # Heterogeneous dist checkpoint (needed for hybrid architecture)
    hetereogenous_dist_checkpoint: bool = True

    # TODO: MTP (Multi-Token Prediction) support for VL context.
    # Qwen3.5 model card states "MTP: trained with multi-steps" but it's unclear
    # how MTP interacts with the vision encoder in VL mode.
    mtp_num_layers: Optional[int] = None

    def __post_init__(self):
        _check_qwen3_5_moe_available()
        if self.vision_config is None:
            self.vision_config = Qwen3_5MoeVisionConfig()
        if self.num_query_groups < self.tensor_model_parallel_size:
            raise ValueError(
                f"TP size {self.tensor_model_parallel_size} should be less than or equal to num_query_groups {self.num_query_groups}. Please use a smaller TP size."
            )
        super().__post_init__()

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Qwen3VLModel:
        """Provide a Qwen3.5 VL model instance with vision and language components.

        Qwen3.5 uses a hybrid architecture (GDN + standard attention). The key
        challenge is that Qwen3VLModel.__init__ does::

            language_transformer_layer_spec.submodules.self_attention.module = Qwen3VLSelfAttention

        which assumes a single ModuleSpec and patches ALL layers uniformly.
        For Qwen3.5, only the standard attention layers (every 4th layer) should
        get the Qwen3VLSelfAttention override; GDN layers must be left alone.

        Solution: build the hybrid TransformerBlockSubmodules spec, selectively
        patch only the standard attention layer specs, then pass it to
        Qwen3VLModel. Because GPTModel → TransformerBlock already accepts
        TransformerBlockSubmodules, we just need to bypass the uniform patch
        in Qwen3VLModel.__init__ by calling MegatronModule.__init__ directly
        and constructing the internals ourselves.
        """
        language_transformer_config = self
        hf_vision_config = self.vision_config

        # Build hybrid block spec: produces TransformerBlockSubmodules with
        # per-layer specs (GDN layers get GatedDeltaNet, attention layers get
        # standard SelfAttention + MoE).
        block_spec = get_transformer_block_with_experimental_attention_variant_spec(
            language_transformer_config,
            vp_stage=vp_stage,
        )

        # Selectively patch only the standard (full) attention layer specs
        # with Qwen3VLSelfAttention for mRoPE support. GDN layers are left as-is.
        _patch_standard_attention_specs(block_spec, Qwen3VLSelfAttention)

        # Qwen3VLModel expects a single ModuleSpec and does a uniform patch on
        # line 87. We pass the block spec instead – GPTModel/TransformerBlock
        # already handles TransformerBlockSubmodules natively. The uniform patch
        # line will fail on a TransformerBlockSubmodules, so we bypass it by
        # passing the spec through and letting the model ignore the single-spec
        # assumption. We pass the block_spec as the layer spec; Qwen3VLGPTModel
        # forwards it to TransformerBlock which handles both types.
        model = Qwen3VLModel(
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=block_spec,
            vision_transformer_config=hf_vision_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self._pg_collection,
        )

        # Apply freeze options if any are enabled for fine-tuning
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Provide just the language model component without vision."""
        return GPTModelProvider.provide(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


def _patch_standard_attention_specs(
    block_spec: TransformerBlockSubmodules,
    attention_cls,
) -> None:
    """Selectively replace the self_attention module on standard attention layer specs.

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

    for layer_spec in block_spec.layer_specs:
        attn_spec = layer_spec.submodules.self_attention
        # Standard attention specs use SelfAttention (or a subclass) as the module
        # and have linear_qkv in their submodules. GDN specs use GatedDeltaNet.
        if attn_spec.module is SelfAttention or issubclass(attn_spec.module, SelfAttention):
            attn_spec.module = attention_cls
