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
Gemma 4 Vision-Language (VL) model wrapper for Megatron.

Combines a HuggingFace Gemma4 vision tower + multimodal embedder with a
Megatron-Core GPT language model (Gemma 4 MoE).
"""

import types
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from torch import Tensor
from transformers import AutoModel

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import (
    hook_hf_module_setattr_for_tp_grad_sync,
    slice_batch_for_context_parallel,
)


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


class Gemma4VLModel(MegatronModule):
    """Gemma 4 Vision-Language model wrapping HF vision tower + Megatron language model.

    The vision tower and multimodal embedder (projector) are HF modules loaded
    via ``AutoModel.from_config``. The language model is a Megatron-Core GPTModel
    constructed by the provider.

    Forward flow:
        1. Embed text tokens via language model embedding
        2. If pixel_values provided: run vision tower → embed_vision → scatter into embeddings
        3. Forward through language model decoder
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
            # Vision tower: HF Gemma4VisionModel
            self.vision_tower = AutoModel.from_config(config.vision_config)
            # Multimodal embedder: RMSNorm + Linear projection (vision → language)
            self._init_embed_vision(config)

            # Hook HF vision params for TP grad sync
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)

        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Required for finalize_model_grads
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

    def _init_embed_vision(self, config):
        """Initialize the multimodal embedder (vision → language projection).

        Gemma4's embed_vision is: parameter-free RMSNorm → Linear(vision_hidden, text_hidden).
        We construct it using the HF Gemma4MultimodalEmbedder class.
        """
        try:
            from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder

            self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
        except ImportError:
            # Fallback: manual construction
            from torch import nn

            vision_hidden = config.vision_config.hidden_size
            text_hidden = config.text_config.hidden_size
            eps = config.vision_config.rms_norm_eps

            class _SimpleEmbedder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding_projection = nn.Linear(vision_hidden, text_hidden, bias=False)
                    self._eps = eps

                def forward(self, x):
                    # Parameter-free RMSNorm
                    rms = x.float().pow(2).mean(-1, keepdim=True).add(self._eps).sqrt()
                    x = (x.float() / rms).to(x.dtype)
                    return self.embedding_projection(x)

            self.embed_vision = _SimpleEmbedder()

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def get_image_features(self, pixel_values, image_position_ids=None, **kwargs):
        """Extract and project image features using HF vision tower + embedder.

        Matches HF's Gemma4Model.get_image_features: vision_tower returns
        last_hidden_state (already pooled + standardized), then embed_vision
        projects it to the language model's hidden dimension.
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
            **kwargs,
        )
        last_hidden_state = vision_outputs.last_hidden_state
        projected = self.embed_vision(last_hidden_state)
        return projected

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional["PackedSeqParams"] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass combining HF vision encoder with Megatron language model."""
        if self.pre_process:
            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # [seq_len, batch, hidden]
                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [batch, seq_len, hidden]

            if pixel_values is not None:
                image_features = self.get_image_features(
                    pixel_values, image_position_ids=image_position_ids
                )

                assert input_ids is not None
                special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

                if inputs_embeds[special_image_mask].numel() != image_features.numel():
                    image_tokens_in_text = special_image_mask[:, :, 0].sum().item()
                    raise ValueError(
                        f"Number of images does not match number of special image tokens in the input text. "
                        f"Got {image_tokens_in_text} image tokens in the text but "
                        f"{image_features.numel() // inputs_embeds.shape[-1]} tokens from image embeddings."
                    )
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # (B, T, D) -> (T, B, D)

        # Compute attention mask on FULL sequence (before CP slicing).
        # NOTE: During inference without mm_token_type_ids, HF Gemma4 VLM uses standard
        # causal masking even for image tokens (bidirectional attention requires
        # mm_token_type_ids which is only provided during training). Pass None to use
        # Megatron's default causal masking.
        if attention_mask is None:
            attention_mask = None  # Use Megatron's built-in causal masking

        # CP slicing
        inputs_embeds, labels, loss_mask, position_ids, attention_mask = slice_batch_for_context_parallel(
            inputs_embeds=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
            pg_collection=self.config._pg_collection,
        )

        # SP scatter
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
        return (outputs, loss_mask)

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool):
        """Freeze model modules for fine-tuning."""
        modules = []
        if freeze_language_model and hasattr(self, "language_model"):
            modules.append(self.language_model)
        if freeze_vision_model and hasattr(self, "vision_tower"):
            modules.append(self.vision_tower)
        if freeze_vision_projection and hasattr(self, "embed_vision"):
            modules.append(self.embed_vision)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _compute_attention_mask(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute attention mask with bidirectional attention for image regions."""
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
        attention_mask = ~torch.logical_or(causal_mask, bidirectional_mask.unsqueeze(1))
        return attention_mask
