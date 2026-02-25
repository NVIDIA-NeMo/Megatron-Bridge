# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.utils import openai_gelu

from megatron.bridge.diffusion.models.flux.flux_model import Flux


logger = logging.getLogger(__name__)


@dataclass
class FluxProvider(TransformerConfig, ModelProviderMixin[VisionModule]):
    """
    FLUX model provider configuration.

    Extends TransformerConfig with FLUX-specific parameters and provides
    model instantiation through the ModelProviderMixin interface.

    Attributes:
        num_layers: Dummy setting (required by base class).
        num_joint_layers: Number of double (joint) transformer blocks.
        num_single_layers: Number of single transformer blocks.
        hidden_size: Hidden dimension size.
        num_attention_heads: Number of attention heads.
        activation_func: Activation function to use.
        add_qkv_bias: Whether to add bias to QKV projections.
        in_channels: Number of input channels (latent channels).
        context_dim: Text encoder context dimension.
        model_channels: Model channel dimension for timestep embedding.
        patch_size: Patch size for image embedding.
        guidance_embed: Whether to use guidance embedding (for FLUX-dev).
        vec_in_dim: Vector input dimension (CLIP pooled output dim).
        rotary_interleaved: Whether to use interleaved rotary embeddings.
        apply_rope_fusion: Whether to apply RoPE fusion.
        guidance_scale: Classifier-free guidance scale.
        ckpt_path: Path to checkpoint for loading weights.
        load_dist_ckpt: Whether to load distributed checkpoint.
        do_convert_from_hf: Whether to convert from HuggingFace format.
        save_converted_model_to: Path to save converted model.
    """

    # Base class requirements
    num_layers: int = 1  # Dummy setting
    hidden_size: int = 3072
    ffn_hidden_size: int = 12288
    num_attention_heads: int = 24
    layernorm_epsilon: float = 1e-06
    hidden_dropout: float = 0
    attention_dropout: float = 0

    # FLUX-specific layer configuration
    num_joint_layers: int = 19
    num_single_layers: int = 38

    # Model architecture
    activation_func: Callable = openai_gelu
    add_qkv_bias: bool = True
    in_channels: int = 64
    context_dim: int = 4096
    model_channels: int = 256
    axes_dims_rope: List[int] = field(default_factory=lambda: [16, 56, 56])
    patch_size: int = 1
    guidance_embed: bool = False
    vec_in_dim: int = 768

    # Rotary embedding settings
    rotary_interleaved: bool = True
    apply_rope_fusion: bool = False

    # Initialization and performance settings
    use_cpu_initialization: bool = True
    gradient_accumulation_fusion: bool = False
    enable_cuda_graph: bool = False
    cuda_graph_scope: Optional[str] = None  # full, full_iteration
    use_te_rng_tracker: bool = False
    cuda_graph_warmup_steps: int = 2

    # Inference settings
    guidance_scale: float = 3.5

    # Checkpoint loading settings
    ckpt_path: Optional[str] = None
    load_dist_ckpt: bool = False
    do_convert_from_hf: bool = False
    save_converted_model_to: Optional[str] = None

    # these attributes are unused for images/videos, we just set because bridge training requires for LLMs
    seq_length: int = 1024
    share_embeddings_and_output_weights: bool = False
    vocab_size: int = 25256 * 8
    make_vocab_size_divisible_by: int = 128

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Flux:
        """
        Create and return a Flux model with this configuration.

        Args:
            pre_process: Whether this is the first pipeline stage (unused for Flux).
            post_process: Whether this is the last pipeline stage (unused for Flux).
            vp_stage: Virtual pipeline stage (unused for Flux).

        Returns:
            Configured Flux model instance.
        """
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            total_layers = self.num_joint_layers + self.num_single_layers
            assert (total_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        model = Flux(config=self)
        return model
