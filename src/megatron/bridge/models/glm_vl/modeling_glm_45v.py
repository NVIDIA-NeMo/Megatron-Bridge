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
GLM 4.5 Vision-Language Model for Megatron.

This module provides the GLM45VLModel class that combines:
- HuggingFace's vision encoder (vision_tower) for image processing
- Megatron's language model for text generation

Reference: https://huggingface.co/zai-org/GLM-4.5V
"""

import types
from typing import Optional

import torch
import transformers
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from packaging.version import Version as PkgVersion
from torch import Tensor
from transformers.models.glm4v.modeling_glm4v import Glm4vModel

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


def is_transformers_min_version(version):
    """Check if minimum version of transformers is installed."""
    try:
        transformers_version = PkgVersion(transformers.__version__)
        return transformers_version >= PkgVersion(version)
    except Exception:
        # If version parsing fails, assume false for safety
        return False


class GLM45VModel(MegatronModule):
    """
    GLM 4.5 Vision-Language (VL) model wrapper for Megatron.

    This class combines HuggingFace's vision components with Megatron's language model:
    - Vision tower (HF): Processes images through the vision encoder
    - Multimodal projector (HF): Projects vision features to language model space
    - Language model (Megatron): Generates text conditioned on vision and text inputs

    The vision encoder forward pass uses HuggingFace implementation via monkey-patching,
    while the language model forward pass uses Megatron's optimized implementation.

    Args:
        config (GPTModelProvider): Model provider containing configuration for language and vision modules.
        pre_process (bool, optional): Whether to construct the vision tower and projector. Default: True.
        post_process (bool, optional): Whether to apply post-processing. Default: True.
        vp_stage (Optional[int], optional): Pipeline stage for model parallelism. Default: None.

    Attributes:
        pre_process (bool): If True, enables vision and multimodal components.
        post_process (bool): If True, enables post-processing.
        vp_stage (Optional[int]): Pipeline stage for model parallelism.
        vision_tower (nn.Module): Vision encoder from HuggingFace.
        multi_modal_projector (nn.Module): Projects vision features to language model space.
        language_model (nn.Module): Megatron language model.
        get_image_features (callable): Method to extract image features (monkey-patched from HF).

    Forward Inputs:
        input_ids (torch.LongTensor, optional): Tokenized input ids for the language model.
        attention_mask (torch.Tensor, optional): Attention mask for the language model.
        position_ids (torch.LongTensor, optional): Position ids for the language model.
        inputs_embeds (torch.FloatTensor, optional): Precomputed input embeddings.
        pixel_values (torch.Tensor, optional): Image tensor(s) for the vision tower.
        labels (torch.Tensor, optional): Target labels for supervised training.
        runtime_gather_output (bool, optional): If True, gather outputs across pipeline stages.
        loss_mask (Tensor, optional): Mask for loss computation.

    Returns:
        Tensor: Model output (e.g., logits or loss, depending on mode).

    Note:
        - If `pre_process` is False, only the language model is constructed.
        - The vision tower and projector are only active if `pre_process` is True.
        - This class is intended for use within the Megatron-LM framework.
        - Requires transformers >= 5.0.0 for Mistral3 model support.
    """

    def __init__(
        self,
        config: GPTModelProvider,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        # Bind methods from HF's Qwen2_5_VLModel to this instance
        # get_placeholder_mask is only available in transformers 4.55+
        if not is_transformers_min_version("4.57.1"):
            raise RuntimeError(
                f"transformers version {transformers.__version__} is not supported. "
                f"get_placeholder_mask requires transformers >= 4.55.0. "
                f"Please upgrade transformers: pip install 'transformers>=4.55.0'"
            )

        if pre_process:
            from transformers.models.glm4v.modeling_glm4v import Glm4vVisionModel

            self.visual = Glm4vVisionModel._from_config(config.vision_config)
            # Ensure HF visual tower params are marked for TP grad sync
            hook_hf_module_setattr_for_tp_grad_sync(self.visual)

        # Initialize Megatron language model
        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Finalize grad requires these to be bound with module
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        # Monkey-patch methods from HuggingFace Mistral3Model
        # This allows us to use HF's image feature extraction logic
        self.get_image_features = types.MethodType(Glm4vModel.get_image_features, self)
        self.get_video_features = types.MethodType(Glm4vModel.get_video_features, self)
        self.get_rope_index = types.MethodType(Glm4vModel.get_rope_index, self)
        self.get_placeholder_mask = types.MethodType(Glm4vModel.get_placeholder_mask, self)

        # Some config requires from HF vision tower
        self.config.spatial_merge_size = getattr(self.config.vision_config, "spatial_merge_size", 2)

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass combining HuggingFace vision encoder with Megatron language model.

        Args:
            input_ids: Tokenized input ids for the language model.
            attention_mask: Attention mask for the language model.
            position_ids: Position ids for the language model.
            inputs_embeds: Precomputed input embeddings.
            pixel_values: Image tensor(s) for the vision tower.
            pixel_values_videos: Video tensor(s) for the vision tower.
            image_grid_thw: Grid of image sizes for the vision tower.
            video_grid_thw: Grid of video sizes for the vision tower.
            second_per_grid_ts: Time interval for each grid along the temporal dimension in the 3D position IDs.
            labels: Target labels for supervised training.
            runtime_gather_output: If True, gather outputs across pipeline stages.
            loss_mask: Mask for loss computation.

        Returns:
            Model output (logits or loss depending on mode).
        """
        if self.pre_process:
            if inputs_embeds is None:
                # Get text embeddings from Megatron language model
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # [seq_len, batch, hidden]

                # Transpose to HF format [batch, seq_len, hidden]
                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Transpose back to Megatron format [seq_len, batch, hidden]
            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()

            if self.config.sequence_parallel:
                inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds)

        # Compute MRoPE position_ids on ALL pipeline stages
        # Each stage has input_ids and visual grid info from the data iterator
        # This avoids any broadcasting overhead
        hf_attention_mask = None
        position_ids, rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask=hf_attention_mask,
        )

        # Forward through Megatron language model
        outputs = self.language_model.forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
        )
        return outputs

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module (patch_embed and blocks).
            freeze_vision_projection (bool): Freeze the vision projection module (merger).
        """
        modules = []

        if freeze_language_model and hasattr(self, "language_model") and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model and hasattr(self, "visual") and self.visual is not None:
            # Vision model consists of patch_embed and blocks
            if hasattr(self.visual, "patch_embed"):
                modules.append(self.visual.patch_embed)
            if hasattr(self.visual, "blocks"):
                modules.append(self.visual.blocks)

        if freeze_vision_projection and hasattr(self, "visual") and self.visual is not None:
            # Vision projection is the merger module
            if hasattr(self.visual, "merger"):
                modules.append(self.visual.merger)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
