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

"""Step3.7 multimodal model orchestrator.

Combines a vision tower, a vision-text fusion step, and a Step-3.5 text
decoder. The forward path is:

* **Input**: ``forward(input_ids, images: list[ImageForInsert], cu_seqlens,
  position_ids, attention_mask, labels, loss_mask, packed_seq_params, ...)``.
* **Vision encode**: ``_encode_images_for_insert(images)`` runs the PE-G/14
  trunk + both downsamplers per :class:`ImageForInsert`, populating
  ``image_features`` (``[N, 169, encoder.output_dim]``).
* **Vision-text fusion**: :class:`ImageInsertEmbedding` (owns
  ``align_projector``: ``encoder.output_dim → hidden_size``) projects the
  features and scatter-inserts them at each ``<im_start>`` (+1) via its
  ``insert_features`` algorithm.
* The combined embedding is handed to the standard Step-3.5 text decoder
  via :class:`Step37GPTModel.forward(decoder_input=...)`.

The model consumes ``list[ImageForInsert]`` directly; there are no
``pixel_values`` / ``image_grid_thw`` Qwen-VL-style kwargs.
"""

from __future__ import annotations

from typing import Optional

import torch
from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.stepfun.modelling_step37.image_insert_embedding import (
    ImageForInsert,
    ImageInsertEmbedding,
)
from megatron.bridge.models.stepfun.modelling_step37.text_model import Step37GPTModel
from megatron.bridge.models.stepfun.modelling_step37.transformer_config import (
    Step37TransformerConfig,
)
from megatron.bridge.models.stepfun.modelling_step37.vision_model import Step37VisionModel


class Step37Model(MegatronModule):
    """Step3.7 multimodal model.

    Args:
        language_transformer_config: Step3.7 ``TransformerConfig`` carrying both
            the Step-3.5 text-decoder fields and the multimodal fields
            (``vision_config``, ``image_token_id``,
            ``understand_projector_stride``, ``projector_bias``).
        language_transformer_layer_spec: Per-layer ``ModuleSpec`` for the text
            decoder — see ``modelling_step37/transformer_block.py``.
        vision_transformer_config: HF ``StepRoboticsVisionEncoderConfig``
            describing the PE-G/14 trunk.
        parallel_output: forwarded to :class:`Step37GPTModel`.
        pre_process / post_process: standard PP-stage flags.
        add_encoder / add_decoder: PP-stage gating for the vision and language
            modules. The vision tower is built only when both ``pre_process``
            and ``add_encoder`` are true.
        pg_collection: process-group bundle (uses MPU defaults if ``None``).
        mtp_block_spec: optional MTP block spec forwarded to GPTModel.
        vp_stage: optional virtual-PP stage index.
    """

    def __init__(
        self,
        language_transformer_config: Step37TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        vision_transformer_config,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.vision_transformer_config = vision_transformer_config
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.language_model = None
        self.image_insert_embedding = None
        self.image_token_id = language_transformer_config.image_token_id

        # Step-3.5's text decoder runs the standard share-embeddings logic;
        # we surface it on the wrapper so finalize_model_grads can find it.
        self.share_embeddings_and_output_weights = False

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.cp_group = pg_collection.cp
        self.tp_group = pg_collection.tp
        self.pp_group = pg_collection.pp
        self.embd_group = getattr(pg_collection, "embd", None)

        if self.pre_process and self.add_encoder:
            self.vision_model = Step37VisionModel(vision_transformer_config)

        if self.add_decoder:
            self.language_model = Step37GPTModel(
                config=language_transformer_config,
                transformer_layer_spec=language_transformer_layer_spec,
                vocab_size=language_transformer_config.vocab_size,
                max_sequence_length=language_transformer_config.language_max_sequence_length,
                parallel_output=parallel_output,
                position_embedding_type="rope",
                rotary_percent=language_transformer_config.rotary_percent,
                pre_process=self.pre_process,
                post_process=self.post_process,
                rotary_base=language_transformer_config.rotary_base,
                fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
                share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
                scatter_embedding_sequence_parallel=False,
                mtp_block_spec=mtp_block_spec,
                vp_stage=vp_stage,
                pg_collection=pg_collection,
            )
            self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

        # ``ImageInsertEmbedding`` owns the projector (encoder.output_dim →
        # hidden_size) and provides the ``insert_features`` scatter logic.
        # Constructed only on PP rank 0 because it references
        # ``language_model.embedding`` for word lookup — non-PP-first stages
        # don't run the fusion step.
        if self.pre_process and self.add_decoder and self.language_model is not None:
            self.image_insert_embedding = ImageInsertEmbedding(
                language_embedding=self.language_model.embedding,
                encoder_output_dim=vision_transformer_config.width * 4,
                hidden_size=language_transformer_config.hidden_size,
                projector_bias=language_transformer_config.projector_bias,
            )

    def shared_embedding_or_output_weight(self):
        if self.add_decoder and self.language_model is not None:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    @property
    def decoder(self):
        return getattr(self.language_model, "decoder", None)

    def set_input_tensor(self, input_tensor) -> None:
        """Standard PP plumbing — encoder_hidden_state on pre_process ranks,
        otherwise forward to the language model.
        """
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should be length 1 for Step3.7"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze any combination of the language tower / vision tower /
        projector for fine-tuning scenarios."""
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.image_insert_embedding is not None:
            modules.append(self.image_insert_embedding.align_projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    # ─── Multimodal hooks ──────────────────────────────────────────────────────

    def _encode_images_for_insert(self, images: Optional[list[ImageForInsert]]) -> Optional[list[ImageForInsert]]:
        """Encode raw image pixels into vision features.

        For each :class:`ImageForInsert` in ``images``, runs the vision tower
        on its raw ``[N, 3, H, W]`` pixels (if ``image_features`` isn't
        already populated) and returns a new ``ImageForInsert`` carrying the
        encoded ``[N, P, encoder.output_dim]`` features. The
        ``insert_start_token`` + RoPE metadata is preserved.

        Vision runs in the **same mesh** as the decoder, so this is a single
        ``self.vision_model(pixels)`` call.
        """
        if not images:
            return images
        if self.vision_model is None:
            return images

        encoder_dtype = next(self.vision_model.parameters()).dtype
        encoder_device = next(self.vision_model.parameters()).device

        processed: list[ImageForInsert] = []
        for insert_image in images:
            if insert_image.image_features is not None:
                processed.append(insert_image)
                continue
            if insert_image.images is None:
                raise ValueError("ImageForInsert requires either images or image_features")
            pixels = insert_image.images
            if pixels.dim() > 4:
                pixels = pixels.view(-1, *pixels.shape[-3:])
            pixels = pixels.to(device=encoder_device, dtype=encoder_dtype)
            if pixels.shape[0] == 0:
                image_features = pixels.new_empty((0, 0, 0))
            else:
                image_features = self.vision_model(pixels)
            processed.append(
                ImageForInsert(
                    insert_start_token=insert_image.insert_start_token,
                    image_features=image_features,
                    rope_cu_seqlens=insert_image.rope_cu_seqlens,
                    rope_max_seq_len=insert_image.rope_max_seq_len,
                )
            )
        return processed

    def forward_head(
        self,
        input_ids: torch.Tensor,
        images: Optional[list[ImageForInsert]] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the fused vision-text input embedding.

        Delegates to :class:`ImageInsertEmbedding` to compute word-embedding
        + ``align_projector`` + ``insert_features`` scatter. Returns the
        fused ``[S, B, H]`` embedding ready for the decoder.
        """
        assert self.image_insert_embedding is not None, (
            "forward_head called without an ImageInsertEmbedding — only valid on PP rank 0 (pre_process=True) ranks."
        )
        return self.image_insert_embedding(
            input_ids=input_ids,
            images=images,
            position_ids=position_ids,
            **kwargs,
        )

    # ─── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[list[ImageForInsert]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        max_seq_len: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        inference_params: Optional[InferenceParams] = None,
        extra_block_kwargs: Optional[dict] = None,
        inference_context: object | None = None,
        runtime_gather_output: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Step3.7 forward.

        Args:
            input_ids: ``[1, T]`` packed token ids (the flickr8k pipeline
                always feeds ``B=1`` because per-pack sub-sequences are
                demarcated by ``cu_seqlens``).
            images: pre-encoded or raw ``list[ImageForInsert]``. Each item's
                ``insert_start_token`` points at the placeholder token id
                (e.g. ``<im_start>``) used by ``insert_features`` to locate
                the 169-token ``<im_patch>`` span.
            cu_seqlens: ``[B_sub+1]`` int32 sub-sequence boundary array
                inside the packed row.
            position_ids: optional per-sub-seq position ids (``None`` lets
                the decoder layer's RoPE module compute them internally).
            packed_seq_params: pre-built FlashAttn varlen ``PackedSeqParams``
                — the flickr8k forward step builds these from ``cu_seqlens``.
            attention_mask / labels / loss_mask / max_seq_len: standard
                multimodal SFT batch fields.
        """
        del inference_context  # API compatibility only
        del cu_seqlens, max_seq_len  # carried by packed_seq_params already
        assert inference_params is None, "Step3.7 forward does not yet support inference_params"

        combined_embeddings = None

        if self.pre_process:
            # 1) Vision encode: list[ImageForInsert]/raw → list[ImageForInsert]/encoded
            processed_images = self._encode_images_for_insert(images)

            # 2) Vision-text fusion: word embedding + align_projector + scatter.
            combined_embeddings = self.forward_head(
                input_ids=input_ids,
                images=processed_images,
                position_ids=position_ids,
            )

        # 3) Decoder + MTP + output_layer.
        output = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            runtime_gather_output=runtime_gather_output,
            **(extra_block_kwargs or {}),
            **kwargs,
        )
        return output


__all__ = ["Step37Model"]
