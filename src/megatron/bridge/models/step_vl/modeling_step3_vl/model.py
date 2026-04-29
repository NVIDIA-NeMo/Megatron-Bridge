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

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.step_vl.modeling_step3_vl.vision_encoder import StepRoboticsVisionEncoder
from megatron.bridge.utils.common_utils import (
    hook_hf_module_setattr_for_tp_grad_sync,
    slice_batch_for_context_parallel,
)

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


class Step3VLModel(MegatronModule):
    """
    Step3-VL vision-language model wrapper for Megatron.

    Combines the Step3-VL HF vision encoder (StepRoboticsVisionEncoder) with a
    Megatron-Core GPT language model via a simple linear projector.

    Args:
        config: GPTModelProvider subclass (Step3VLModelProvider) with vision fields.
        pre_process: Build vision tower and projector on this stage.
        post_process: Language model post-processing (logits / loss).
        vp_stage: Virtual-pipeline stage index.

    Weight namespace:
        vision_model.**         – all ViT parameters including downsamplers
        vit_large_projector.*   – vision-to-language projection
        language_model.**       – Megatron GPTModel
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
            self.vision_model = StepRoboticsVisionEncoder(config.vision_config)
            # Projector: width*4 (post-downsample channels) → language hidden_size
            self.vit_large_projector = nn.Linear(
                config.vision_config.width * 4,
                config.hidden_size,
                bias=config.projector_bias,
            )
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)

        self.language_model = config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor)

    # ------------------------------------------------------------------
    # Vision helpers
    # ------------------------------------------------------------------

    def _process_image_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply spatial downsampling and linear projection to ViT output patches.

        Args:
            features: (B, P, D) where P = (H/patch)^2, D = vision width (1536).

        Returns:
            (B, P', hidden_size) where P' = (H / (patch * 4))^2 = 169 for 728px input.
        """
        B, P, D = features.shape
        HW = int(P**0.5)
        # (B, D, HW, HW) → downsample → (B, D*4, HW/4, HW/4)
        features = features.permute(0, 2, 1).reshape(B, D, HW, HW)
        features = self.vision_model.vit_downsampler1(features)
        features = self.vision_model.vit_downsampler2(features)
        B, C, h, w = features.shape
        # (B, h*w, C) → project to language hidden dim
        features = features.reshape(B, C, h * w).permute(0, 2, 1)
        return self.vit_large_projector(features)

    def _encode_images(
        self,
        pixel_values: torch.Tensor,
        patch_pixel_values: Optional[torch.Tensor],
        num_patches: Optional[list],
    ) -> torch.Tensor:
        """Encode global images and optional local patches, then merge per-image.

        Returns:
            (total_image_tokens, hidden_size) flat tensor ready for scatter into
            the language-model embedding sequence.
        """
        global_feats = self._process_image_features(self.vision_model(pixel_values))  # (B, P', H)

        if patch_pixel_values is not None:
            patch_feats = self._process_image_features(self.vision_model(patch_pixel_values))  # (N, P', H)
        else:
            patch_feats = None

        # Merge: [local_patches..., global_image] concatenated per sample
        if num_patches is None:
            num_patches = [0] * global_feats.shape[0]

        merged = []
        cur_patch_idx = 0
        for i, n_patch in enumerate(num_patches):
            parts = []
            if n_patch > 0 and patch_feats is not None:
                parts.append(patch_feats[cur_patch_idx : cur_patch_idx + n_patch].reshape(-1, patch_feats.shape[-1]))
            parts.append(global_feats[i])
            cur_patch_idx += n_patch
            merged.append(torch.cat(parts, dim=0) if len(parts) > 1 else parts[0])

        return torch.cat(merged, dim=0)  # (total_tokens, hidden_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        patch_pixel_values: Optional[torch.Tensor] = None,
        num_patches: Optional[list] = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional["PackedSeqParams"] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass fusing HF vision encoder with Megatron language model."""

        if self.pre_process:
            if inputs_embeds is None:
                # [T, B, H] → [B, T, H]
                inputs_embeds = self.language_model.embedding(input_ids=input_ids, position_ids=None)
                inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()

            if pixel_values is not None:
                image_features = self._encode_images(pixel_values, patch_pixel_values, num_patches)
                # Scatter image features into image-token positions
                assert input_ids is not None, "input_ids is required to locate image token positions"
                special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
                image_features = image_features.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()  # [T, B, H]

        # Context-parallel slicing (must happen AFTER vision-text merge)
        inputs_embeds, labels, loss_mask, position_ids, attention_mask = slice_batch_for_context_parallel(
            inputs_embeds=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
            pg_collection=self.config._pg_collection,
        )

        if self.config.sequence_parallel and inputs_embeds is not None:
            inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds)

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
        return outputs, loss_mask

    def freeze(
        self,
        *,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ) -> None:
        """Selectively freeze model components."""
        modules: list[nn.Module] = []
        if freeze_language_model and hasattr(self, "language_model"):
            modules.append(self.language_model)
        if freeze_vision_model and hasattr(self, "vision_model"):
            modules.append(self.vision_model)
        if freeze_vision_projection and hasattr(self, "vit_large_projector"):
            modules.append(self.vit_large_projector)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
