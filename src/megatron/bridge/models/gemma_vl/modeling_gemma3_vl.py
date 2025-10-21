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
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from torch import Tensor
from transformers import AutoModel, Gemma3Model

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from megatron.bridge.utils.import_utils import safe_import_from


TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")


class Gemma3VLModel(MegatronModule):
    """
    Gemma3 Vision-Language (VL) model wrapper for Megatron.
    Args:
        config (GPTModelProvider): Model provider containing configuration for language and vision modules.
        pre_process (bool, optional): Whether to construct the vision tower and projector. Default: True.
        post_process (bool, optional): Whether to apply post-processing. Default: True.
        vp_stage (int | None, optional): Pipeline stage for model parallelism. Default: None.

    Attributes:
        pre_process (bool): If True, enables vision and multimodal components.
        post_process (bool): If True, enables post-processing.
        vp_stage (int | None): Pipeline stage for model parallelism.
        vision_tower (nn.Module): Vision encoder (e.g., SigLIP or other vision backbone).
        multi_modal_projector (nn.Module): Projects vision features to language model space.
        language_model (nn.Module): The underlying language model.
        get_image_features (callable): Method to extract image features, compatible with HuggingFace Gemma3Model.

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
        vp_stage: int | None = None,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage
        if pre_process:
            self.vision_tower = AutoModel.from_config(config.vision_config)
            self.multi_modal_projector = Gemma3VLMultimodalProjector(config.vision_projector_config)
            # Ensure HF visual tower params are marked for TP grad sync and future assignments are hooked.
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)
        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Finalize grad requires these to be bound with module
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        self.get_image_features = types.MethodType(Gemma3Model.get_image_features, self)

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
        *,
        loss_mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        if self.pre_process:
            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # [decoder_seq_len, b, h_language]

                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [b, decoder_seq_len, h_language]

            if pixel_values is not None:
                image_features = self.get_image_features(pixel_values)

                # TODO might need to check if input_ids is None
                assert input_ids is not None
                special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

                if inputs_embeds[special_image_mask].numel() != image_features.numel():
                    image_tokens_in_text = special_image_mask.sum(dim=1).item(dim=0)[0]
                    raise ValueError(
                        f"Number of images does not match number of special image tokens in the input text. "
                        f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                        "tokens from image embeddings."
                    )
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # (B, T, D) -> (T, B, D)

        attention_mask = self._compute_attention_mask(input_ids)

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
            and hasattr(self, "multi_modal_projector")
            and self.multi_modal_projector is not None
        ):
            # Vision projection is the merger module
            modules.append(self.multi_modal_projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _compute_attention_mask(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.pre_process:
            return None
        batch_size, seq_len = input_ids.shape
        causal_mask = torch.tril(torch.ones((batch_size, 1, seq_len, seq_len))).to(input_ids.device)

        image_mask = input_ids == self.config.image_token_id
        padded_mask = F.pad(image_mask, (1, 0), value=0)
        boundary = padded_mask[:, 1:] > padded_mask[:, :-1]
        numbered_boundary = torch.cumsum(boundary, dim=-1)
        q_block_indices = image_mask * numbered_boundary
        kv_block_indices = q_block_indices
        bidirectional_mask = torch.logical_and(
            kv_block_indices[:, None, :] == q_block_indices.unsqueeze(-1),
            q_block_indices.unsqueeze(-1) > 0,
        )
        # See te.DotProductAttention for the requirement of custom mask
        attention_mask = ~torch.logical_or(causal_mask, bidirectional_mask.unsqueeze(1))
        return attention_mask


@dataclass
class Gemma3VLMultimodalProjectorConfig(TransformerConfig):
    """Gemma3 VL multimodal projector config"""

    input_size: int = 1152
    hidden_size: int = 2560

    image_size: int = 896
    patch_dim: int = 14
    tokens_per_image: int = 256

    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = True  # x * (1 + w)
    layernorm_epsilon: float = 1e-6

    # Do not change
    num_layers: int = 1
    num_attention_heads: int = 8

    def configure_model(self) -> "Gemma3VLMultimodalProjector":
        """Get module"""
        return Gemma3VLMultimodalProjector(self)


class Gemma3VLMultimodalProjector(MegatronModule):
    """Gemma3 VL multimodal projector"""

    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__(config=config)

        self.patches_per_side = config.image_size // config.patch_dim
        tokens_per_side = int(config.tokens_per_image**0.5)
        kernel_size = self.patches_per_side // tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)

        # TODO: fuse layer norm with proj
        self.mm_soft_embed_norm = TENorm(config, config.input_size, eps=config.layernorm_epsilon)
        self.proj = ColumnParallelLinear(
            input_size=config.input_size,
            output_size=config.hidden_size,
            config=config,
            init_method=config.init_method,
            gather_output=True,
            bias=False,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name=None,
        )

    def forward(self, x):
        """Downsample, norm and projection"""
        # (B, 64*64, M)
        batch_size, _, hidden_size = x.shape
        # (B, M, S)
        x = x.transpose(1, 2)
        # (B, M, 64, 64)
        x = x.reshape(batch_size, hidden_size, self.patches_per_side, self.patches_per_side).contiguous()
        # (B, M, 16, 16)
        x = self.avg_pool(x)
        # (B, M, 256)
        x = x.flatten(2)
        # (B, 256, M)
        x = x.transpose(1, 2)
        # (B, 256, M)
        x = self.mm_soft_embed_norm(x)
        # (B, 256, D)
        x, _ = self.proj(x)
        return x
