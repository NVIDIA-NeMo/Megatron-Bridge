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

"""
Provider for ERNIE 4.5 VL MoE model.

Maps HuggingFace Ernie4_5_VLMoeConfig to Megatron-Core TransformerConfig
and provides model instantiation logic for the dual-pool MoE architecture.

The language model uses a custom ErnieMultiTypeMoE layer containing both
text_moe_layer and vision_moe_layer as separate MoELayer instances, each
with their own router, experts, and EP support.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Union

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer.spec_utils import ModuleSpec

try:
    from transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeConfig,
        Ernie4_5_VLMoeVisionConfig,
    )
except ImportError:
    # Fallback for environments where the builtin transformers class is not available
    # (e.g. auto_map models only). Use generic types for type hints.
    Ernie4_5_VLMoeConfig = None
    Ernie4_5_VLMoeVisionConfig = None

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.ernie_vl.ernie_decoder_layer_spec import (
    get_ernie45_vl_decoder_block_spec,
)
from megatron.bridge.models.ernie_vl.modeling_ernie45_vl import Ernie45VLModel


@dataclass
class Ernie45VLModelProvider(GPTModelProvider):
    """
    Model provider for ERNIE 4.5 VL MoE.

    This provider extends GPTModelProvider with ERNIE 4.5 VL-specific fields:
    - Vision configuration for the ViT encoder and resampler
    - 3D M-RoPE parameters (mrope_section)
    - Dual-pool MoE configuration (moe_intermediate_size as tuple)
    - Custom decoder layer spec with ErnieMultiTypeMoE
    - Token IDs for image/video placeholder tokens
    - Freeze options for vision/language components
    """

    # VL models shouldn't scatter embeddings across sequence parallel regions
    # because the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False

    # Position embedding type: M-RoPE for multimodal 3D positions
    position_embedding_type: str = "mrope"
    mrope_section: List[int] = field(default_factory=lambda: [22, 22, 20])

    # Vision configuration -- accepts either Ernie4_5_VLMoeVisionConfig (nested) or
    # DFNRopeVisionTransformerConfig (flat/auto_map) or any config-like object.
    vision_config: Any = field(default_factory=lambda: Ernie4_5_VLMoeVisionConfig() if Ernie4_5_VLMoeVisionConfig else None)
    hf_config: Any = None

    # Dual-pool MoE intermediate sizes: (text_ffn_size, vision_ffn_size)
    # This is passed to ErnieMultiTypeMoE which creates separate configs per pool.
    moe_intermediate_size: Tuple[int, int] = (1536, 512)

    # Token IDs (from Ernie4_5_VLMoeConfig defaults)
    image_start_token_id: int = 101304
    image_end_token_id: int = 101305
    image_token_id: int = 100295
    video_start_token_id: int = 101306
    video_end_token_id: int = 101307
    video_token_id: int = 103367

    # Freeze options
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    # Use MG-native ViT instead of HF-wrapped ViT for better TP performance.
    # When False (default), the vision encoder uses the HuggingFace implementation
    # replicated across TP ranks. When True, uses Megatron-Core TransformerBlock
    # with TE modules for TP-native attention and MLP layers.
    use_mg_vit: bool = False

    # Use custom decoder block spec for heterogeneous layers (dense + dual-pool MoE)
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = (
        get_ernie45_vl_decoder_block_spec
    )

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Ernie45VLModel:
        """Build the composite VLM model (vision + resampler + language model).

        Args:
            pre_process: Whether to include pre-processing (embedding + vision). Defaults to first PP stage.
            post_process: Whether to include post-processing (output layer). Defaults to last PP stage.
            vp_stage: Virtual pipeline stage index.

        Returns:
            Ernie45VLModel: Configured ERNIE 4.5 VL MoE model instance.
        """
        model = Ernie45VLModel(
            self,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

        # Apply freeze options if any are enabled
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Build only the language model (MCoreGPTModel) for weight conversion.

        This uses GPTModelProvider.provide() which builds a standard MCoreGPTModel
        but with the custom ErnieMultiTypeMoE layer spec set via transformer_layer_spec.
        The resulting model has both text_moe_layer and vision_moe_layer as proper
        submodules of each MoE transformer layer.

        Args:
            pre_process: Whether to include pre-processing.
            post_process: Whether to include post-processing.
            vp_stage: Virtual pipeline stage index.

        Returns:
            MCoreGPTModel: Configured Megatron-Core GPT model instance with dual-pool MoE.
        """
        return GPTModelProvider.provide(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
