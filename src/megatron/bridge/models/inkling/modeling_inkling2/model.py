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

"""Inkling2 HybridModel: normalized token and image embeddings, AttnRes decoder blocks, and a soft-capped head."""

from __future__ import annotations

import copy
from typing import Any

import torch
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from torch import Tensor, nn

from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.inkling.modeling_inkling2.spec import build_inkling2_hybrid_stack_spec
from megatron.bridge.models.inkling.modeling_inkling2.vision import Inkling2VisionEncoder
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


class Inkling2OutputLayerMixin(nn.Module):
    """Soft-cap the logits and exclude the padded vocabulary rows from the softmax."""

    def forward(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor | None]:
        output, bias = super().forward(*args, **kwargs)
        softcap = self.config.inkling2_final_logit_softcap
        if softcap:
            output = softcap * torch.tanh(output / softcap)
        unpadded = self.config.inkling2_unpadded_vocab_size
        if unpadded is not None:
            local = output.shape[-1]
            start = 0
            if local != self.output_size:
                start = torch.distributed.get_rank(self.tp_group) * local
            if start + local > unpadded:
                columns = torch.arange(start, start + local, device=output.device)
                output = output.masked_fill(columns >= unpadded, float("-inf"))
        return output, bias


class Inkling2Model(HybridModel):
    """Inkling2 decoder as an MCore HybridModel with an embedding RMSNorm and an optional image encoder.

    The Hybrid pattern repeats ``*E``: each decoder block is an attention layer
    followed by a MoE layer. The image encoder sits on the embedding stage.
    """

    def __init__(
        self,
        config,
        *,
        vocab_size: int,
        pg_collection: ProcessGroupCollection,
        pre_process: bool,
        post_process: bool,
    ) -> None:
        super().__init__(
            config=config.transformer,
            hybrid_stack_spec=build_inkling2_hybrid_stack_spec(config.transformer),
            vocab_size=vocab_size,
            max_sequence_length=config.seq_length,
            hybrid_layer_pattern=config.hybrid_layer_pattern,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=config.fp16_lm_cross_entropy,
            parallel_output=config.parallel_output,
            share_embeddings_and_output_weights=False,
            position_embedding_type="none",
            scatter_embedding_sequence_parallel=False,
            pg_collection=pg_collection,
            **logit_dtype_kwarg(HybridModel, config.logit_dtype),
        )
        self.model_config = config
        if pre_process:
            # The norm runs redundantly on the full sequence before the sequence-parallel
            # scatter, so its gradients already agree across TP ranks.
            norm_config = copy.copy(self.config)
            norm_config.sequence_parallel = False
            self.embed_norm = TENorm(norm_config, self.config.hidden_size, self.config.layernorm_epsilon)
            if config.vision_config and not config.hf_model_text_only:
                self.vision = Inkling2VisionEncoder(config.vision_config, self.config.hidden_size)
                self.vision.to(device=torch.cuda.current_device(), dtype=self.config.params_dtype)
                hook_hf_module_setattr_for_tp_grad_sync(self.vision)
        if post_process:
            extend_instance(self.output_layer, Inkling2OutputLayerMixin)
        self._freeze_requested_modules()

    def _freeze_requested_modules(self) -> None:
        config = self.model_config
        if config.freeze_language_model:
            for name in ("embedding", "embed_norm", "decoder", "output_layer"):
                module = getattr(self, name, None)
                if module is not None:
                    module.requires_grad_(False)
        if config.freeze_vision_model and getattr(self, "vision", None) is not None:
            self.vision.backbone.requires_grad_(False)
        if config.freeze_vision_projection and getattr(self, "vision", None) is not None:
            self.vision.proj_in.requires_grad_(False)
            self.vision.proj_out.requires_grad_(False)

    def embed(
        self,
        input_ids: Tensor,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
    ) -> Tensor:
        """Return full-sequence decoder inputs ``[S, B, H]``: normalized token embeddings with image features.

        Image features replace the placeholder positions after the embedding norm,
        as in the reference implementation.
        """
        embeddings = self.embed_norm(self.embedding(input_ids=input_ids, position_ids=None))
        if pixel_values is None:
            return embeddings
        if getattr(self, "vision", None) is None:
            raise ValueError("This Inkling2 model was built without the image encoder")
        if image_grid_thw is None:
            raise ValueError("Image inputs require image_grid_thw")
        if self.pg_collection.cp.size() > 1:
            # Each CP rank holds only part of the image placeholders, so the features would need a split too.
            raise ValueError("Inkling2 does not support image inputs with context parallelism yet")
        features = self.vision(pixel_values, image_grid_thw.cpu())
        batch_first = embeddings.transpose(0, 1)
        mask = (input_ids == self.model_config.image_token_id).unsqueeze(-1).expand_as(batch_first)
        if int(mask.sum().item()) != features.numel():
            raise ValueError("Image placeholders do not match the encoded image features")
        batch_first = batch_first.masked_scatter(mask, features.to(batch_first.dtype))
        return batch_first.transpose(0, 1).contiguous()

    def forward(
        self,
        input_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        *,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        packed_seq_params: Any = None,
        decoder_input: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Embed tokens and images on the first stage, then run the decoder and output head."""
        if self.pre_process and decoder_input is None:
            if input_ids is None:
                raise ValueError("input_ids are required on the embedding stage")
            decoder_input = self.embed(input_ids, pixel_values, image_grid_thw)
            if self.config.sequence_parallel:
                # HybridModel then scatters the MoE padding mask to match.
                decoder_input = scatter_to_sequence_parallel_region(decoder_input, group=self.pg_collection.tp)
        output = super().forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )
        return output
