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
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional
from typing import TYPE_CHECKING, Callable, List, Optional, Union

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.bridge.models.gpt_provider import GPTModelProvider
import torch.nn.functional as F

try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

if TYPE_CHECKING:
    from megatron.core.transformer import ModuleSpec

if HAVE_TE:
    from megatron.core.utils import is_te_min_version


@dataclass
class KimiK25VLModelProvider(MLAModelProvider):
    """
    Model provider for Kimi K2.5 VL (Vision-Language) Models.

    Inherits language model configuration from KimiK2Provider since the
    Kimi K2.5 language backbone shares the same architecture as Kimi K2
    (MoE with MLA, 384 experts, 61 layers, etc.).

    Minor config differences (rotary_scaling_factor, layernorm_epsilon,
    init_method_std) between K2 and K2.5 are handled at runtime by
    ``get_common_configs()`` in the bridge, which reads actual values
    from the HuggingFace config.

    The vision component (MoonViT3d + PatchMergerMLP) is dynamically loaded
    from the HuggingFace model repository at runtime via ``trust_remote_code``.
    """

    transformer_layer_spec: Union["ModuleSpec", Callable[["GPTModelProvider"], "ModuleSpec"]] = partial(
        get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE
    )

    # Model
    num_layers: int = 61
    hidden_size: int = 7168
    ffn_hidden_size: int = 18432
    num_moe_experts: int = 384
    moe_ffn_hidden_size: int = 2048
    moe_shared_expert_intermediate_size: int = 2048  # 2048 * 1 shared expert
    moe_layer_freq: Union[int, List[int]] = field(default_factory=lambda: [0] + [1] * 60)  # first layer are dense
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True  # swiglu
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    share_embeddings_and_output_weights: bool = False
    num_attention_heads: int = 64
    kv_channels: int = 64
    max_position_embeddings: int = 4096
    seq_length: int = 4096
    rotary_base: float = 50000.0
    make_vocab_size_divisible_by: int = 1280
    mtp_num_layers: Optional[int] = None
    mtp_loss_scaling_factor: Optional[float] = None

    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    qk_layernorm: bool = True

    # MoE
    moe_router_topk: int = 8
    moe_router_num_groups: int = 1
    moe_router_group_topk: int = 1
    moe_router_topk_scaling_factor: float = 2.827
    moe_aux_loss_coeff: float = 1e-3
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 1e-3
    moe_grouped_gemm: bool = True
    moe_router_pre_softmax: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_shared_expert_overlap: bool = True
    moe_router_dtype: Optional[str] = "fp32"

    # MLA
    multi_latent_attention: bool = True
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    rotary_scaling_factor: float = 32
    beta_fast: float = 1.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0

    # Miscellaneous
    init_method_std: float = 0.006
    layernorm_epsilon: float = 1e-6
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    attention_softmax_in_fp32: bool = False
    persist_layer_norm: bool = True
    num_layers_in_first_pipeline_stage: Optional[int] = None
    num_layers_in_last_pipeline_stage: Optional[int] = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False
    vocab_size: int = 163840

    # fusions
    apply_rope_fusion: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    masked_softmax_fusion: bool = True
    gradient_accumulation_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    cross_entropy_fusion_impl: str = "te"
    moe_permute_fusion: bool = is_te_min_version("2.1.0") if HAVE_TE else False

    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False

    # Vision configuration — raw HF KimiK25VisionConfig object, used to construct
    # VisionTowerConfig and ProjectorConfig for the vision tower and mm_projector.
    vision_config: Any = None

    # Path to HuggingFace model directory (required for dynamic module loading
    # of MoonViT3d, PatchMergerMLP, and other custom model classes).
    hf_model_path: Optional[str] = None

    # Token IDs (from Kimi K2.5 config.json)
    bos_token_id: int = 163584
    eos_token_id: int = 163585
    image_token_id: int = 163605  # media_placeholder_token_id in HF config
    # Fields needed by HF's _merge_input_ids_with_image_features (bound via MethodType)
    media_placeholder_token_id: int = 163605
    pad_token_id: int = 163839
    ignore_index: int = -100

    # Freeze options for fine-tuning scenarios
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    # Generation configuration
    generation_config: Any | None = None

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """
        Provide a KimiK25VL model instance with vision and language components.

        Returns:
            KimiK25VLModel: Configured Kimi K2.5 VL model with vision tower,
            multimodal projector, and Kimi K2 language model.
        """
        from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel

        model = KimiK25VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        # Apply freeze options if any are enabled
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """
        Provide just the language model component (Kimi K2 MoE) without vision.

        This is called by KimiK25VLModel to construct only the language backbone,
        while the vision tower and projector are constructed separately.

        Args:
            pre_process: Whether this is the first stage in pipeline parallelism.
            post_process: Whether this is the last stage in pipeline parallelism.
            vp_stage: Virtual pipeline stage number.

        Returns:
            MCoreGPTModel instance (language model only).
        """
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
