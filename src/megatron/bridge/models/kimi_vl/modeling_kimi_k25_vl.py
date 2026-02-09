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

import types
from typing import Optional

import torch
from megatron.core.transformer.module import MegatronModule
from torch import Tensor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


class KimiK25VLModel(MegatronModule):
    """
    Kimi K2.5 Vision-Language (VL) model wrapper for Megatron.
    Args:
        config (GPTModelProvider): Model provider containing configuration for language and vision modules.
        pre_process (bool, optional): Whether to construct the vision tower and projector. Default: True.
        post_process (bool, optional): Whether to apply post-processing. Default: True.
        vp_stage (Optional[int], optional): Pipeline stage for model parallelism. Default: None.

    Attributes:
        pre_process (bool): If True, enables vision and multimodal components.
        post_process (bool): If True, enables post-processing.
        vp_stage (Optional[int]): Pipeline stage for model parallelism.
        vision_tower (nn.Module): Vision encoder (MoonViT3d vision backbone).
        mm_projector (nn.Module): PatchMergerMLP that projects vision features to language model space.
        language_model (nn.Module): The underlying Kimi K2 language model.
        get_image_features (callable): Method to extract and project image features.

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

        if config.hf_model_path is None:
            raise ValueError("hf_model_path must be set.")
        
        KimiK25ForConditionalGeneration = get_class_from_dynamic_module(
            "modeling_kimi_k25.KimiK25ForConditionalGeneration",
            config.hf_model_path,
        )
        if pre_process:
            # Load vision tower and projector classes from the custom HuggingFace model code
            MoonViT3dPretrainedModel = get_class_from_dynamic_module(
                "modeling_kimi_k25.MoonViT3dPretrainedModel",
                config.hf_model_path,
            )
            PatchMergerMLP = get_class_from_dynamic_module(
                "modeling_kimi_k25.PatchMergerMLP",
                config.hf_model_path,
            )
            ProjectorConfig = get_class_from_dynamic_module(
                "modeling_kimi_k25.ProjectorConfig",
                config.hf_model_path,
            )
            VisionTowerConfig = get_class_from_dynamic_module(
                "modeling_kimi_k25.VisionTowerConfig",
                config.hf_model_path,
            )
            self.vision_tower_config = VisionTowerConfig(config.vision_config)
            self.projector_config = ProjectorConfig(config.vision_config)
            self.vision_tower = MoonViT3dPretrainedModel(self.vision_tower_config)
            self.mm_projector = PatchMergerMLP(self.projector_config) # TODO: support different types of mm projector
            # Ensure HF visual tower params are marked for TP grad sync and future assignments are hooked.
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)
            hook_hf_module_setattr_for_tp_grad_sync(self.mm_projector)
        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Finalize grad requires these to be bound with module
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        self._extract_image_features = types.MethodType(KimiK25ForConditionalGeneration._extract_image_features, self)
        self._merge_input_ids_with_image_features = types.MethodType(KimiK25ForConditionalGeneration._merge_input_ids_with_image_features, self)

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
        image_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            input_ids: Tokenized input ids for the language model.
            attention_mask: Attention mask for the language model.
            position_ids: Position ids for the language model.
            inputs_embeds: Precomputed input embeddings.
            pixel_values: Image tensor for the vision tower.
            image_grid_thw: Tensor of shape ``(num_images, 3)`` containing ``[temporal, height, width]``
                for each image's grid dimensions in the LLM. This corresponds to ``grid_thws`` in
                the HF Kimi K2.5 processor output.
            labels: Target labels for supervised training.
            runtime_gather_output: If True, gather outputs across pipeline stages.
            loss_mask: Mask for loss computation.
        """
        if self.pre_process:
            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # (T, B, D) — Megatron convention

            if pixel_values is not None:
                # Transpose to (B, T, D) for HF's merge function which uses batch-first convention
                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # (T, B, D) -> (B, T, D)

                image_features = self._extract_image_features(pixel_values, image_grid_thw)
                image_features = self.mm_projector(image_features)
                inputs_embeds = inputs_embeds.to(image_features[0].dtype)
                inputs_embeds, attention_mask, labels, position_ids = (
                    self._merge_input_ids_with_image_features(
                        image_features,
                        inputs_embeds,
                        input_ids,
                        attention_mask,
                        labels,
                    ))

                # Transpose back to (T, B, D) for Megatron language model
                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # (B, T, D) -> (T, B, D)

        outputs = self.language_model.forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,  # (B, 1, T, T)
            decoder_input=inputs_embeds,  # (T, B, D)
            labels=labels,  # (B, T)
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
        )
        return outputs

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool):
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

        if freeze_vision_model and hasattr(self, "vision_tower") and self.vision_tower is not None:
            # Vision model consists of patch_embed and blocks
            modules.append(self.vision_tower)

        if (
            freeze_vision_projection
            and hasattr(self, "mm_projector")
            and self.mm_projector is not None
        ):
            # Vision projection is the merger module
            modules.append(self.mm_projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False


