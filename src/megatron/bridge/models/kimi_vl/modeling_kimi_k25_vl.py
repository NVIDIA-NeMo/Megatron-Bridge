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

from typing import List, Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from torch import Tensor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


class KimiK25VLModel(MegatronModule):
    """Kimi K2.5 Vision-Language (VL) model wrapper for Megatron.

    Combines a MoonViT3d vision encoder + PatchMergerMLP projector with the
    Kimi K2 language backbone (MoE + MLA).

    On the first pipeline stage (``pre_process=True``), vision features are
    extracted and merged into the language embeddings.  The merged embeddings
    are then forwarded through the language model.
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

        if pre_process:
            MoonViT3dPretrainedModel = get_class_from_dynamic_module(
                "modeling_kimi_k25.MoonViT3dPretrainedModel",
                config.hf_model_path,
            )

            # Patch MoonViT3dEncoder to add missing use_deterministic_attn attribute
            # (idempotent — only patches once).
            import importlib

            _vit_module = importlib.import_module(MoonViT3dPretrainedModel.__module__)
            if not getattr(_vit_module.MoonViT3dEncoder, "_bridge_init_patched", False):
                _orig_encoder_init = _vit_module.MoonViT3dEncoder.__init__

                def _patched_encoder_init(self, *args, **kwargs):
                    self.use_deterministic_attn = False
                    _orig_encoder_init(self, *args, **kwargs)

                _vit_module.MoonViT3dEncoder.__init__ = _patched_encoder_init
                _vit_module.MoonViT3dEncoder._bridge_init_patched = True

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

            # Reload vision config from HF model path to ensure correct types
            from megatron.bridge.models.hf_pretrained.safe_config_loader import safe_load_config_with_retry

            config.vision_config = safe_load_config_with_retry(
                config.hf_model_path, trust_remote_code=True
            ).vision_config

            self.vision_tower_config = VisionTowerConfig(config.vision_config)
            self.projector_config = ProjectorConfig(config.vision_config)

            self.vision_tower = MoonViT3dPretrainedModel(self.vision_tower_config)
            self.mm_projector = PatchMergerMLP(self.projector_config)

            hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)
            hook_hf_module_setattr_for_tp_grad_sync(self.mm_projector)

        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        self.media_placeholder_token_id = config.media_placeholder_token_id

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def _merge_input_ids_with_image_features(
        self,
        image_features: List[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        target_seq_length: Optional[int] = None,
    ):
        """Merge image features into input embeddings.

        Supports two modes:
        1. Pre-expanded (PP mode): input_ids already has N placeholder tokens per image,
           where N = number of image features. Does simple 1:1 replacement.
        2. Dynamic expansion: input_ids has 1 placeholder per image, expands to N tokens.
        """
        _, embed_dim = image_features[0].shape
        feature_lengths = [x.shape[0] for x in image_features]
        total_image_features = sum(feature_lengths)
        image_features_cat = torch.cat(image_features, dim=0)

        image_token_index = self.media_placeholder_token_id
        pad_token_id = self.config.pad_token_id
        ignore_index = self.config.ignore_index

        batch_size, sequence_length = input_ids.shape

        num_placeholders = (input_ids == image_token_index).sum().item()

        # Pre-expanded mode: simple 1:1 replacement, no sequence length change
        if num_placeholders == total_image_features:
            final_embedding = inputs_embeds.clone()
            image_mask = input_ids == image_token_index
            final_embedding[image_mask] = image_features_cat.to(inputs_embeds.dtype)

            final_attention_mask = attention_mask
            position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_((attention_mask == 0), 1)

            if labels is not None:
                final_labels = labels.clone()
                final_labels[image_mask] = ignore_index
            else:
                final_labels = None

            return final_embedding, final_attention_mask, final_labels, position_ids

        # Dynamic expansion mode
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))

        _token_occupation_table = torch.ones_like(input_ids.flatten())
        _token_occupation_table[input_ids.flatten() == image_token_index] = torch.tensor(
            feature_lengths, dtype=torch.long, device=input_ids.device
        )
        _token_occupation_table = _token_occupation_table.reshape(input_ids.shape)

        natural_max_embed_dim = _token_occupation_table.sum(-1).max().item()
        max_embed_dim = target_seq_length if target_seq_length is not None else natural_max_embed_dim

        batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

        new_token_positions = torch.cumsum(_token_occupation_table, -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )

        target_device = inputs_embeds.device
        batch_indices = batch_indices.to(target_device)
        non_image_indices = non_image_indices.to(target_device)
        text_to_overwrite = text_to_overwrite.to(target_device)
        attention_mask = attention_mask.to(target_device)

        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        image_to_overwrite = torch.full((batch_size, max_embed_dim), True, dtype=torch.bool, device=target_device)
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        final_embedding[image_to_overwrite] = image_features_cat.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        batch_indices_pad, pad_indices = torch.where(input_ids == pad_token_id)
        indices_to_mask = new_token_positions[batch_indices_pad, pad_indices]
        final_embedding[batch_indices_pad, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def _extract_image_features(self, pixel_values, grid_thws):
        """Extract and project image features.

        Returns:
            List[Tensor]: One feature tensor per image, each of shape (num_tokens, hidden_dim).
        """
        all_features = self.vision_tower(pixel_values, grid_thws)
        projected = self.mm_projector(all_features)

        # Split into per-image feature tensors based on grid_thws
        merge_h, merge_w = self.config.vision_config.merge_kernel_size
        feature_list = []
        offset = 0
        for t, h, w in grid_thws.tolist():
            num_tokens = int((h // merge_h) * (w // merge_w))
            feature_list.append(projected[offset : offset + num_tokens])
            offset += num_tokens
        return feature_list

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
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ) -> Tensor:
        if self.pre_process:
            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # (seq_len, batch, hidden)
                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # → (batch, seq_len, hidden)

            has_pixels = pixel_values is not None and pixel_values.size(0) > 0
            not_generation = input_ids is not None and input_ids.shape[1] != 1

            if has_pixels and not_generation:
                pixel_values = pixel_values.to(dtype=next(self.vision_tower.parameters()).dtype)
                image_features = self._extract_image_features(pixel_values, image_grid_thw)

                inputs_embeds = inputs_embeds.to(image_features[0].dtype)

                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    labels,
                )

            # (batch, seq, hidden) → (seq, batch, hidden) for Megatron language model
            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()

            if self.config.sequence_parallel:
                inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds)

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

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool):
        """Freeze model modules for fine-tuning scenarios."""
        modules = []
        if freeze_language_model and hasattr(self, "language_model") and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and hasattr(self, "vision_tower") and self.vision_tower is not None:
            modules.append(self.vision_tower)
        if freeze_vision_projection and hasattr(self, "mm_projector") and self.mm_projector is not None:
            modules.append(self.mm_projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
