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

"""Image-insert word embedding for Step3.7.

Defines:
  - :class:`ImageForInsert` — the dataclass that represents one "image
    insertion group" travelling from the data preprocess to the model
    forward. Lives on the model side (vs the data side) because it's
    fundamentally part of the ``Step37Model.forward`` input contract; the
    model-owned Flickr8k data pipeline imports it from this module directly.
  - :class:`ImageInsertEmbedding` — owns ``align_projector``
    (``nn.Linear(encoder.output_dim, hidden_size)``) and provides
    ``insert_features``, which finds each ``<im_start>`` in ``input_ids``,
    offsets by +1 to the first ``<im_patch>``, and **in-place** slices
    ``input_embeddings[start:start+L]`` with the projected feature rows.

``ImageInsertEmbedding`` borrows (does **not** own):
  - ``language_embedding`` — a reference to ``Step37Model.language_model.embedding``
    (Megatron-Core ``LanguageModelEmbedding``). Stored via
    ``object.__setattr__`` to bypass ``nn.Module``'s auto-registration so
    the same Parameter tensor isn't counted twice in ``parameters()`` /
    ``state_dict()``.

Output shape: ``[S, B, H]`` (sequence-first), matching Megatron-Core
``LanguageModelEmbedding.forward``.

This module performs the vision-text fusion step. The caller
(:class:`Step37Model.forward_head`) supplies a pre-encoded
``list[ImageForInsert]`` (with ``image_features`` populated by the vision
tower); this module projects + scatter-inserts them into the text
embedding and returns the fused tensor to be fed as ``decoder_input`` of
the GPTModel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn


logger = logging.getLogger(__name__)


# ─── ImageForInsert dataclass (model-interface type) ────────────────────────


@dataclass
class ImageForInsert:
    """Language-model insert payload for image / multicrop-patch features.

    Lives on the model side (not the data side) because it's part of the
    ``Step37Model.forward`` input contract — the data subpackage
    re-exports it for downstream data-side imports.

    Attributes:
        insert_start_token: Token id after which the visual features are
            inserted (``<im_start>`` for image, ``<patch_start>`` for
            multicrop patches).
        images: Raw image tensor shaped ``[N, 3, H, W]``. Either this
            *or* ``image_features`` is populated; the encoder pipeline
            consumes ``images`` and populates ``image_features``.
        image_features: Optional precomputed features before the
            language projector (``[N, L, C]``). Used for the decoupled-
            encoder mode where the vision tower runs outside the decoder.
        rope_cu_seqlens: Per-image patch cu_seqlens for visual RoPE
            (shape ``[N + 1]``).
        rope_max_seq_len: Max patch count across all images in this
            ``ImageForInsert`` (a Python int for serializability).
    """

    insert_start_token: int
    images: Optional[torch.Tensor] = None
    image_features: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None
    rope_cu_seqlens: Optional[torch.Tensor] = None
    rope_max_seq_len: Optional[int] = None


class ImageInsertEmbedding(nn.Module):
    """Word embedding + image-feature projection + ``<im_start>`` scatter-insert."""

    def __init__(
        self,
        language_embedding,
        encoder_output_dim: int,
        hidden_size: int,
        projector_bias: bool = False,
    ):
        super().__init__()
        # Reference to GPTModel.embedding — held as a non-Module attribute so
        # nn.Module auto-tracking does NOT re-register the embedding's
        # parameters under this module's state_dict. The embedding stays
        # canonically owned by GPTModel.
        object.__setattr__(self, "_language_embedding", language_embedding)

        self.align_projector = nn.Linear(encoder_output_dim, hidden_size, bias=projector_bias)
        nn.init.kaiming_normal_(self.align_projector.weight, mode="fan_in", nonlinearity="relu")
        if self.align_projector.bias is not None:
            nn.init.zeros_(self.align_projector.bias)

    # ─── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_feature_list(
        image_features: Union[torch.Tensor, list[torch.Tensor]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        if isinstance(image_features, torch.Tensor):
            if image_features.dim() != 3:
                raise ValueError(f"Expected image_features with shape [N, L, C], got {tuple(image_features.shape)}")
            features = list(image_features.unbind(0))
        else:
            features = []
            for feature in image_features:
                if feature.dim() == 3:
                    features.extend(list(feature.unbind(0)))
                elif feature.dim() == 2:
                    features.append(feature)
                else:
                    raise ValueError(f"Expected feature tensor rank 2 or 3, got rank {feature.dim()}")
        return [feature.to(device=device, dtype=dtype) for feature in features]

    @staticmethod
    def insert_features(
        input_embeddings: torch.Tensor,
        image_features: Union[torch.Tensor, list[torch.Tensor]],
        input_ids: torch.Tensor,
        flag: int,
    ) -> torch.Tensor:
        """Scatter-insert projected image features at ``<im_start>`` positions.

        Finds every position where ``input_ids == flag`` (the ``<im_start>``
        id), shifts by +1 to land on the first ``<im_patch>`` placeholder,
        and overwrites the next ``feature.shape[0]`` rows of
        ``input_embeddings`` (sequence-first ``[S, B, H]``) with the
        provided image-feature rows.
        """
        feature_list = ImageInsertEmbedding._normalize_feature_list(
            image_features,
            device=input_embeddings.device,
            dtype=input_embeddings.dtype,
        )
        if not feature_list:
            return input_embeddings

        insert_locations = torch.nonzero(input_ids == flag, as_tuple=False)
        if insert_locations.numel() == 0:
            return input_embeddings
        insert_locations = insert_locations.clone()
        insert_locations[:, 1] += 1

        if len(feature_list) != insert_locations.shape[0]:
            logger.warning(
                "Mismatch between image features and insert locations: %s features vs %s locations; "
                "truncating to the overlap.",
                len(feature_list),
                insert_locations.shape[0],
            )
        count = min(len(feature_list), insert_locations.shape[0])
        if count == 0:
            return input_embeddings

        output = input_embeddings.clone()
        for location, feature in zip(insert_locations[:count], feature_list[:count], strict=False):
            batch_idx = int(location[0].item())
            start = int(location[1].item())
            end = min(start + feature.shape[0], output.shape[0])
            if end <= start:
                continue
            output[start:end, batch_idx] = feature[: end - start]
        return output

    # ─── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.IntTensor,
        images: Optional[list[ImageForInsert]] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute word embeddings and scatter-insert pre-encoded image features.

        Args:
            input_ids: ``[B, S]`` long token ids (placeholder positions live
                at the ``insert_start_token`` of each ``ImageForInsert`` —
                typically ``<im_start>``).
            images: list of :class:`ImageForInsert`. Each item **must** have
                ``image_features`` populated (the vision tower runs upstream,
                e.g. inside ``Step37Model._encode_images_for_insert``).
            position_ids: forwarded to the underlying word embedding (None is
                accepted; Step-3.5's per-layer rotary is computed inside the
                decoder, so the position arg here is normally ignored).

        Returns:
            Fused embedding ``[S, B, H]`` (sequence-first), ready to feed
            into the GPT decoder via ``decoder_input``.
        """
        input_embeddings = self._language_embedding(input_ids=input_ids, position_ids=position_ids)
        if not images:
            return input_embeddings

        target_dtype = self.align_projector.weight.dtype
        target_device = self.align_projector.weight.device
        for insert_image in images:
            if insert_image.image_features is None:
                raise ValueError(
                    "ImageInsertEmbedding expects pre-encoded image_features "
                    "(populated by Step37Model._encode_images_for_insert)"
                )
            image_features = insert_image.image_features
            if isinstance(image_features, torch.Tensor):
                image_features = image_features.to(device=target_device, dtype=target_dtype)
                image_features = self.align_projector(image_features)
            else:
                image_features = [
                    self.align_projector(feature.to(device=target_device, dtype=target_dtype))
                    for feature in image_features
                ]

            input_embeddings = self.insert_features(
                input_embeddings=input_embeddings,
                image_features=image_features,
                input_ids=input_ids,
                flag=insert_image.insert_start_token,
            )
        return input_embeddings


__all__ = ["ImageForInsert", "ImageInsertEmbedding"]
