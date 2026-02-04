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
Ministral 3 Vision-Language Model for Megatron.

This module provides the Ministral3Model class that combines:
- HuggingFace's vision encoder (vision_tower) for image processing
- HuggingFace's multimodal projector for vision-to-language projection
- Megatron's language model for text generation

Reference: https://huggingface.co/mistralai/Ministral-3-3B-Base-2512
"""

import types
from typing import TYPE_CHECKING, Optional

import torch
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import (
    hook_hf_module_setattr_for_tp_grad_sync,
    slice_batch_for_context_parallel,
)


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


# Import HuggingFace Mistral3 model classes with fallback
try:
    from transformers import Mistral3ForConditionalGeneration
    from transformers.models.mistral3.modeling_mistral3 import Mistral3Model as HFMistral3Model

    HAS_MISTRAL3 = True
except ImportError:
    Mistral3ForConditionalGeneration = None
    HFMistral3Model = None
    HAS_MISTRAL3 = False


class Ministral3Model(MegatronModule):
    """
    Ministral 3 Vision-Language (VL) model wrapper for Megatron.

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

        if pre_process:
            if not HAS_MISTRAL3:
                raise ImportError(
                    "Mistral3 model requires transformers >= 5.0.0. Please upgrade: pip install 'transformers>=5.0.0'"
                )

            # Initialize vision tower from HuggingFace config
            # The vision_tower includes: patch_conv, ln_pre, transformer layers
            from transformers import AutoModel

            self.vision_tower = AutoModel.from_config(config.hf_config.vision_config)

            # Preserve inv_freq as FP32 during dtype conversions (e.g., when wrapped by Float16Module)
            # This is necessary because inv_freq requires FP32 precision for numerical stability
            if hasattr(self.vision_tower, "patch_positional_embedding"):
                pos_emb = self.vision_tower.patch_positional_embedding
                original_apply = pos_emb._apply

                def _apply_preserve_inv_freq(fn):
                    # Save inv_freq before conversion
                    inv_freq_backup = None
                    if hasattr(pos_emb, "inv_freq") and pos_emb.inv_freq is not None:
                        inv_freq_backup = pos_emb.inv_freq.data.clone()

                    # Apply the transformation (e.g., bfloat16 conversion)
                    result = original_apply(fn)

                    # Restore inv_freq to FP32 (only move device, keep dtype)
                    if inv_freq_backup is not None:
                        # Get the new device from the transformation
                        new_device = fn(torch.tensor([1.0])).device
                        pos_emb.inv_freq.data = inv_freq_backup.to(device=new_device)

                    return result

                pos_emb._apply = _apply_preserve_inv_freq

            # Initialize multimodal projector from HuggingFace config
            # The projector includes: norm, linear layers
            from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

            self.multi_modal_projector = Mistral3MultiModalProjector(config.hf_config)

            # Ensure HF visual tower params are marked for TP grad sync
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)
            hook_hf_module_setattr_for_tp_grad_sync(self.multi_modal_projector)

        # Initialize Megatron language model
        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Finalize grad requires these to be bound with module
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        # Monkey-patch methods from HuggingFace Mistral3Model
        # This allows us to use HF's image feature extraction logic
        if HAS_MISTRAL3 and HFMistral3Model is not None:
            self.get_image_features = types.MethodType(HFMistral3Model.get_image_features, self)

        # Some config requires from HF vision tower
        self.config.spatial_merge_size = getattr(self.config.hf_config, "spatial_merge_size", 2)
        self.config.vision_feature_layer = getattr(self.config.hf_config, "vision_feature_layer", -1)

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
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        image_sizes: Optional[torch.Tensor] = None,
        packed_seq_params: Optional["PackedSeqParams"] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward pass combining HuggingFace vision encoder with Megatron language model.

        Args:
            input_ids: Tokenized input ids for the language model.
            attention_mask: Attention mask for the language model.
            position_ids: Position ids for the language model.
            inputs_embeds: Precomputed input embeddings.
            pixel_values: Image tensor(s) for the vision tower.
            labels: Target labels for supervised training.
            runtime_gather_output: If True, gather outputs across pipeline stages.
            loss_mask: Mask for loss computation.

        Returns:
            tuple: (output_tensor, loss_mask) where output_tensor contains model output
                   and loss_mask is the CP-sliced mask for consistent loss computation.
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
                # Get image features using HF's method (monkey-patched)
                image_features = self.get_image_features(
                    pixel_values.to(inputs_embeds.dtype), image_sizes=image_sizes, return_dict=True
                ).pooler_output
                image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

                # Replace image tokens in text embeddings with image features
                assert input_ids is not None
                special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

                if inputs_embeds[special_image_mask].numel() != image_features.numel():
                    image_tokens_in_text = special_image_mask.sum(dim=1).sum(dim=0)[0].item()
                    raise ValueError(
                        f"Number of images does not match number of special image tokens in the input text. "
                        f"Got {image_tokens_in_text} image tokens in the text but "
                        f"{image_features.shape[0] * image_features.shape[1]} tokens from image embeddings."
                    )
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            # Transpose back to Megatron format [seq_len, batch, hidden]
            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()

        # CP slicing: slice embeddings, labels, loss_mask, position_ids, and attention_mask
        # This must happen AFTER vision-text merge so image token positions are correct
        inputs_embeds, labels, loss_mask, position_ids, attention_mask = slice_batch_for_context_parallel(
            inputs_embeds=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
            pg_collection=self.config._pg_collection,
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
            packed_seq_params=packed_seq_params,
        )
        # Return both outputs and the CP-sliced loss_mask for consistent loss computation
        return (outputs, loss_mask)

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
            freeze_vision_model (bool): Freeze the vision model module (vision_tower).
            freeze_vision_projection (bool): Freeze the vision projection module (multi_modal_projector).
        """
        modules = []

        if freeze_language_model and hasattr(self, "language_model") and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model and hasattr(self, "vision_tower") and self.vision_tower is not None:
            modules.append(self.vision_tower)

        if (
            freeze_vision_projection
            and hasattr(self, "multi_modal_projector")
            and self.multi_modal_projector is not None
        ):
            modules.append(self.multi_modal_projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
