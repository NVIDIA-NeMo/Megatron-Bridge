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

"""Native causal D1 vision/language composition with a tied vocabulary head."""

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, Lfm2VlConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.lfm2_vl.modeling_lfm2_vl import (
    Lfm2VlCausalLMOutputWithPast,
    Lfm2VlModel,
    Lfm2VlMultiModalProjector,
)

from megatron.bridge.models.d1.modeling_d1.backbone import D1Backbone


class D1VLInputs(nn.Module):
    """Combine existing HF vision processing with a native MCore language backbone.

    Args:
        config: Native language transformer configuration.
        hf_config: Published vision/language configuration.
    """

    def __init__(self, config: TransformerConfig, *, hf_config: Lfm2VlConfig) -> None:
        super().__init__()
        self.config = hf_config
        self.language_model = D1Backbone(
            config,
            vocab_size=hf_config.text_config.vocab_size,
            layer_types=tuple(hf_config.text_config.layer_types),
            causal=True,
            rotary_base=hf_config.text_config.rope_parameters["rope_theta"],
        )
        device = self.language_model.embed_tokens.weight.device
        self.vision_tower = AutoModel.from_config(hf_config.vision_config).to(device=device, dtype=config.params_dtype)
        self.multi_modal_projector = Lfm2VlMultiModalProjector(hf_config).to(device=device, dtype=config.params_dtype)
        for parameter in list(self.vision_tower.parameters()) + list(self.multi_modal_projector.parameters()):
            parameter.allreduce = True
            parameter.tensor_model_parallel = False

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the native language input embedding table."""
        return self.language_model.embed_tokens

    def get_image_features(
        self, *, pixel_values: torch.Tensor, spatial_shapes: torch.Tensor, pixel_attention_mask: torch.Tensor
    ) -> BaseModelOutputWithPooling:
        """Invoke the existing HF vision/grid/projector path without an HF language model.

        Args:
            pixel_values: HF-processed image patches.
            spatial_shapes: Original patch-grid shapes [images, 2].
            pixel_attention_mask: Boolean valid-patch mask [images, patches].

        Returns:
            HF vision output with projected variable-length image features.
        """
        return Lfm2VlModel.get_image_features(
            self,
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            return_dict=True,
        )


class D1VLModel(MegatronModule):
    """Trainable native D1 causal chains preserving the published embedding tie.

    The released tree batching and inference cache are outside this post-training
    chain API. HF perception computation is reused directly.

    Args:
        config: Native language transformer configuration.
        hf_config: Published complete vision/language configuration.
    """

    def __init__(self, config: TransformerConfig, *, hf_config: Lfm2VlConfig) -> None:
        super().__init__(config)
        if not hf_config.tie_word_embeddings:
            raise ValueError("D1 VL requires tied input and vocabulary weights")
        self.pre_process = True
        self.post_process = True
        self.model = D1VLInputs(config, hf_config=hf_config)
        embedding = self.model.get_input_embeddings()
        # Construct the readout without allocating a second vocabulary table.
        self.lm_head = nn.Linear(embedding.embedding_dim, embedding.num_embeddings, bias=False, device="meta")
        self.lm_head.weight = embedding.weight

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the shared native vocabulary/input weight owner."""
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Linear:
        """Return the vocabulary readout sharing the input embedding parameter."""
        return self.lm_head

    def encode_inputs(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode text and optional processed images into native causal states.

        Args:
            input_ids: Long tensor [batch, length].
            attention_mask: Optional boolean real-token mask [batch, length].
            pixel_values: Optional HF-processed image patches.
            spatial_shapes: Required grid shapes when images are supplied.
            pixel_attention_mask: Required patch mask when images are supplied.

        Returns:
            Normalized causal states and optional concatenated image features.
        """
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be long with shape [batch, length]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be boolean with the input_ids shape")
        lengths = attention_mask.sum(dim=1)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        if not torch.equal(attention_mask, positions < lengths[:, None]):
            raise ValueError("D1 causal chain batches require right-padded real tokens")
        values = self.model.get_input_embeddings()(input_ids)
        image_features = None
        if pixel_values is not None:
            if spatial_shapes is None or pixel_attention_mask is None:
                raise ValueError("processed images require spatial_shapes and pixel_attention_mask")
            image_features = self.model.get_image_features(
                pixel_values=pixel_values, spatial_shapes=spatial_shapes, pixel_attention_mask=pixel_attention_mask
            ).pooler_output
            image_features = torch.cat(image_features, dim=0).to(values)
            placeholders = input_ids == self.model.config.image_token_id
            if int(placeholders.sum()) != image_features.shape[0]:
                raise ValueError("image placeholders and projected image features must have equal counts")
            values = values.masked_scatter(placeholders[..., None].expand_as(values), image_features)
        elif spatial_shapes is not None or pixel_attention_mask is not None:
            raise ValueError("image metadata requires pixel_values")
        prefixes = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        hidden = self.model.language_model(values, valid_tokens=attention_mask, media_lengths=prefixes)
        return hidden, image_features

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> Lfm2VlCausalLMOutputWithPast:
        """Return tied-vocabulary logits and optional shifted causal CE loss.

        Args:
            input_ids: Long IDs [batch, length].
            attention_mask: Optional boolean real-token mask [batch, length].
            pixel_values: Optional HF-processed image patches.
            spatial_shapes: Optional image-grid shapes.
            pixel_attention_mask: Optional real-patch mask.
            labels: Optional next-token targets, with -100 for ignored targets.
            logits_to_keep: Number of trailing positions; zero means all positions.

        Returns:
            Differentiable vocabulary logits, causal loss and native language states.
        """
        if logits_to_keep < 0 or (labels is not None and logits_to_keep):
            raise ValueError("logits_to_keep must be nonnegative and zero for causal loss")
        hidden, image_features = self.encode_inputs(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
        )
        selected = hidden[:, -logits_to_keep:, :] if logits_to_keep else hidden
        logits = self.lm_head(selected)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape or labels.dtype != torch.long or labels.device != logits.device:
                raise ValueError("labels must be long with the input_ids shape on the logits device")
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        return Lfm2VlCausalLMOutputWithPast(
            loss=loss, logits=logits, hidden_states=(hidden,), image_hidden_states=image_features
        )
