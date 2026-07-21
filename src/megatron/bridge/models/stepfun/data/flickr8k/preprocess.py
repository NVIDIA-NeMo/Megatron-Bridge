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

"""Per-step ``preprocess`` for Step3.7 multimodal SFT.

Runs once per micro-batch. Takes the packed dict from :func:`pack_samples`
plus the precomputed image_paths and produces the dict that is fed to the
model (already on CUDA, with ``images`` = ``list[ImageForInsert]``):

  - PIL ``Image.open`` + ``.convert("RGB")``  (zero-image fallback on error)
  - ``Image.resize((size, size), BILINEAR)`` with size = ``image_size``
    (728 for ``IMAGE_ITEM_TYPE``) or ``patch_image_size`` (504 for
    ``PATCH_ITEM_TYPE``)
  - ``/255 → tensor - CLIP_mean / CLIP_std`` (the CLIP RGB normalization)
  - stack to ``[N, 3, H, W]`` bf16 / cuda via :func:`build_image_for_insert`
  - attach ``rope_cu_seqlens`` via :func:`compute_rope_args` (patch_size = 14)
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image

from megatron.bridge.models.stepfun.data.flickr8k.multimodal_utils import (
    PATCH_ITEM_TYPE,
    build_image_for_insert,
    compute_rope_args,
)


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
logger = logging.getLogger(__name__)


def _load_image(path: str) -> Image.Image:
    """Open an RGB ``PIL.Image``; fall back to a 224×224 zero-image on
    read failure so a single broken jpg does not crash the run."""
    try:
        if path.startswith("s3://"):
            # Optional dep — only imported when an s3 path is hit.
            from megfile import smart_open  # type: ignore[import-untyped]

            with smart_open(path, "rb") as f:
                return Image.open(BytesIO(f.read())).convert("RGB")
        return Image.open(path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image from %s is broken or unavailable; using zero-image. Error: %s", path, exc)
        return Image.new(size=(224, 224), mode="RGB")


def _image_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    """Resize → ``/255`` → CLIP-normalize → ``[3, H, W]`` float32 CPU.

    Arithmetic order: BILINEAR resize, then divide by 255 before mean/std.
    Operates on a contiguous numpy float32 array for a deterministic
    rounding sequence.
    """
    image = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(_CLIP_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(_CLIP_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def load_images(
    image_paths: list[tuple[str, int]],
    *,
    image_size: int,
    patch_image_size: int,
) -> list[tuple[torch.Tensor, int]]:
    """Load and preprocess images for a packed batch.

    Returns ``[(tensor[3, H, W], image_type), ...]``, with H/W chosen per
    ``image_type``: full image ``= image_size`` (default 728),
    multicrop patch ``= patch_image_size`` (default 504).
    """
    images = []
    for path, image_type in image_paths:
        size = patch_image_size if int(image_type) == PATCH_ITEM_TYPE else image_size
        images.append((_image_to_tensor(_load_image(path), int(size)), int(image_type)))
    return images


def preprocess_packed_batch(
    batch: dict,
    *,
    img_start_token_id: int,
    patch_start_token_id: int,
    image_size: int,
    patch_image_size: int,
    encoder_patch_size: int,
    only_pp_first_stage: bool = True,
) -> dict[str, Any]:
    """Build the model input dict from a packed batch.

    Moves the packed dict's tensors to CUDA and (on PP rank 0) loads
    images + builds ``list[ImageForInsert]``. The ``images`` list is
    GPU-resident bf16 and ready for ``Step37Model.forward``.

    ``only_pp_first_stage`` should be ``False`` when running outside a
    pipeline-parallel context, or ``True`` to honor PP rank 0 gating.
    """
    cu_seqlens = batch["cu_seqlens"].to("cuda")
    position_id = batch["position_id"].to("cuda")
    max_seq_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])

    if "tokens" not in batch:
        model_input = {
            "cu_seqlens": cu_seqlens,
            "max_seq_len": max_seq_len,
            "position_id": position_id,
        }
        return model_input

    tokens = batch["tokens"].to("cuda")
    labels = batch["labels"].to("cuda")
    loss_masks = batch["loss_masks"].to("cuda")

    # PP-rank gating: PP rank 0 loads images; other PP stages skip.
    # When parallel state is uninitialized, default to loading images.
    is_first_pp_stage = True
    if only_pp_first_stage:
        try:
            from megatron.core import parallel_state

            if parallel_state.is_initialized():  # type: ignore[attr-defined]
                is_first_pp_stage = parallel_state.is_pipeline_first_stage()
        except Exception:
            # Fall through — best-effort guard, always load if state probe fails.
            is_first_pp_stage = True

    images: list = []
    if is_first_pp_stage:
        image_count = int(torch.sum(tokens == img_start_token_id).item())
        patch_count = int(torch.sum(tokens == patch_start_token_id).item())
        loaded_images_preprocessed = load_images(
            batch.get("image_paths", []),
            image_size=image_size,
            patch_image_size=patch_image_size,
        )
        images = build_image_for_insert(
            loaded_images_preprocessed,
            patch_start_id=patch_start_token_id,
            image_start_id=img_start_token_id,
            limit_images=image_count,
            limit_patches=patch_count,
            rope_args_fn=lambda imgs: compute_rope_args(list(imgs), int(encoder_patch_size)),
            to_cuda=True,
        )

    model_input = {
        "input_ids": tokens[None].contiguous(),
        "labels": labels[None].contiguous(),
        "loss_masks": loss_masks[None].contiguous(),
        "images": images,
        "cu_seqlens": cu_seqlens,
        "max_seq_len": max_seq_len,
        "position_id": position_id,
    }
    return model_input
