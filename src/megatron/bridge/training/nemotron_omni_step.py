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

"""Nemotron Omni training step -- extends llava_step with sound support.

Adds ``sound_clips`` and ``sound_length`` to the model forward kwargs so that
LLaVAModel processes audio embeddings alongside vision embeddings.
"""

import logging
from functools import partial
from typing import Iterable

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import get_batch_on_this_cp_rank

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import get_packed_seq_params
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.pg_utils import get_pg_collection


logger = logging.getLogger(__name__)


def get_batch_from_iterator(
    data_iterator: Iterable,
    skip_getting_attention_mask_from_dataset: bool = True,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
) -> dict[str, torch.Tensor]:
    """Get a batch of data from the iterator, including optional sound tensors.

    Handles two batch formats:
    - **HF collate path**: raw ``pixel_values``, ``num_patches`` keys
    - **Energon path**: ``visual_inputs`` (GenericVisualInputs container)
    Both carry ``sound_clips`` / ``sound_length`` when audio is present.
    """
    batch = next(data_iterator)
    required_device_keys = set()
    required_host_keys = set()

    if not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")

    # Vision: either raw tensors (HF collate) or container (Energon)
    if "pixel_values" in batch:
        required_device_keys.add("pixel_values")
    if "num_patches" in batch:
        required_device_keys.add("num_patches")
    if "visual_inputs" in batch:
        required_device_keys.add("visual_inputs")

    # Sound
    if "sound_clips" in batch:
        required_device_keys.add("sound_clips")
    if "sound_length" in batch:
        required_device_keys.add("sound_length")

    # Temporal video embedder
    if "imgs_sizes" in batch:
        required_device_keys.add("imgs_sizes")
    if "num_frames" in batch:
        required_device_keys.add("num_frames")
    if "num_image_tiles" in batch:
        required_device_keys.add("num_image_tiles")

    if "cu_seqlens" in batch:
        required_device_keys.add("cu_seqlens")
        if "cu_seqlens_unpadded" in batch:
            required_device_keys.add("cu_seqlens_unpadded")
        required_host_keys.add("cu_seqlens_argmin")
        if "cu_seqlens_unpadded_argmin" in batch:
            required_host_keys.add("cu_seqlens_unpadded_argmin")
        required_host_keys.add("max_seqlen")

    if is_first_pp_stage:
        required_device_keys.update(("tokens", "input_ids", "position_ids"))
    if is_last_pp_stage:
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in batch.items():
        if key in required_device_keys:
            if key == "visual_inputs":
                if val is None:
                    _batch_required_keys[key] = None
                else:
                    _batch_required_keys[key] = val
                    for k, v in val.__dict__.items():
                        val.__dict__[k] = v.cuda(non_blocking=True) if v is not None else None
            else:
                _batch_required_keys[key] = val.cuda(non_blocking=True) if val is not None else None
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu() if val is not None else None
        else:
            _batch_required_keys[key] = None

    return _batch_required_keys


def _resolve_images(batch: dict) -> torch.Tensor | None:
    """Extract images from either raw pixel_values or GenericVisualInputs container."""
    if "pixel_values" in batch and batch["pixel_values"] is not None:
        return batch["pixel_values"]
    vi = batch.get("visual_inputs")
    if vi is not None and hasattr(vi, "pixel_values"):
        return vi.pixel_values
    return None


# Matches nemotron_omni_provider.py: patch_dim is hardcoded to 16 when building LLaVAModel.
_VISION_PATCH_DIM = 16


def _build_vision_packed_seq_params(
    imgs_sizes: torch.Tensor | None,
) -> PackedSeqParams | None:
    """Build vision PackedSeqParams from per-frame (H, W).

    RADIO's dynamic-resolution + class-token path reads ``packed_seq_params.cu_seqlens_q``
    to insert class tokens at per-image boundaries. We build cu_seqlens from the
    pre-grouping ``imgs_sizes`` (one entry per frame); ``_apply_temporal_grouping``
    rebuilds it after tubelet fusion.
    """
    if imgs_sizes is None or imgs_sizes.numel() == 0:
        return None
    sizes = imgs_sizes.tolist() if torch.is_tensor(imgs_sizes) else list(imgs_sizes)
    seq_lens = [(int(h) // _VISION_PATCH_DIM) * (int(w) // _VISION_PATCH_DIM) for h, w in sizes]
    cu = [0]
    for sl in seq_lens:
        cu.append(cu[-1] + sl)
    device = imgs_sizes.device if torch.is_tensor(imgs_sizes) else torch.device("cpu")
    cu_tensor = torch.tensor(cu, dtype=torch.int32, device=device)
    max_len = torch.tensor(max(seq_lens) if seq_lens else 0, dtype=torch.int32, device=device)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_tensor,
        cu_seqlens_kv=cu_tensor,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
    )


def get_batch(data_iterator: Iterable, cfg: ConfigContainer, *, pg_collection) -> tuple:
    """Generate a batch with vision and sound tensors."""
    is_first = is_pp_first_stage(pg_collection.pp)
    is_last = is_pp_last_stage(pg_collection.pp)
    if (not is_first) and (not is_last):
        return (None,) * 18

    batch = get_batch_from_iterator(
        data_iterator,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
        is_first_pp_stage=is_first,
        is_last_pp_stage=is_last,
    )

    images = _resolve_images(batch)
    sound_clips = batch.get("sound_clips")
    sound_length = batch.get("sound_length")
    imgs_sizes = batch.get("imgs_sizes")
    num_frames = batch.get("num_frames")
    num_image_tiles = batch.get("num_image_tiles")

    # LLaVAModel._process_embedding_token_parallel does the LM-side CP split *after*
    # vision/text embedding merge. Pre-splitting input_ids/labels here would break
    # the image-token merge (num_image_tiles count would not match the CP-local
    # image-token count in input_ids), so leave the LM tensors full-sequence.
    if images is not None:
        batch["images"] = images

    vision_packed_seq_params = _build_vision_packed_seq_params(imgs_sizes)

    assert batch.get("tokens") is not None or batch.get("input_ids") is not None

    return (
        batch.get("images"),
        batch.get("num_patches"),
        batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids"),
        batch.get("labels"),
        batch.get("loss_mask"),
        batch.get("attention_mask"),
        batch.get("position_ids"),
        batch.get("cu_seqlens"),
        batch.get("cu_seqlens_unpadded"),
        batch.get("cu_seqlens_argmin"),
        batch.get("cu_seqlens_unpadded_argmin"),
        batch.get("max_seqlen"),
        sound_clips,
        sound_length,
        imgs_sizes,
        num_frames,
        vision_packed_seq_params,
        num_image_tiles,
    )


def forward_step(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Forward training step for Nemotron Omni (vision + audio + language)."""
    timers = state.timers
    straggler_timer = state.straggler_timer

    pg_collection = get_pg_collection(model)

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        (
            images,
            num_patches,
            input_ids,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            cu_seqlens,
            cu_seqlens_unpadded,
            cu_seqlens_argmin,
            cu_seqlens_unpadded_argmin,
            max_seqlen,
            sound_clips,
            sound_length,
            imgs_sizes,
            num_frames,
            vision_packed_seq_params,
            num_image_tiles,
        ) = get_batch(data_iterator, state.cfg, pg_collection=pg_collection)
    timers("batch-generator").stop()

    # LLaVAModel.forward() requires a non-None images tensor even for audio-only batches
    if images is None:
        images = torch.tensor([], dtype=torch.bfloat16, device=input_ids.device).reshape(0, 0, 0)
    elif images.dtype != torch.bfloat16:
        images = images.to(dtype=torch.bfloat16)

    forward_args = {
        "images": images,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
    }

    if sound_clips is not None:
        forward_args["sound_clips"] = sound_clips.to(dtype=torch.bfloat16)
        forward_args["sound_length"] = sound_length

    if imgs_sizes is not None:
        forward_args["imgs_sizes"] = imgs_sizes
    if num_frames is not None:
        forward_args["num_frames"] = num_frames
    import os as _os
    if _os.environ.get("NOMNI_DEBUG_TILES") == "1":
        print(f"[DEBUG step] num_image_tiles={num_image_tiles if num_image_tiles is None else (num_image_tiles.shape, num_image_tiles.tolist()[:10])}", flush=True)
    if num_image_tiles is not None:
        forward_args["num_image_tiles"] = num_image_tiles
    if vision_packed_seq_params is not None:
        forward_args["vision_packed_seq_params"] = vision_packed_seq_params

    if cu_seqlens is not None:
        packed_seq_params = {
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": cu_seqlens_argmin,
            "max_seqlen": max_seqlen,
        }
        if cu_seqlens_unpadded is not None:
            packed_seq_params["cu_seqlens_unpadded"] = cu_seqlens_unpadded
        if cu_seqlens_unpadded_argmin is not None:
            packed_seq_params["cu_seqlens_unpadded_argmin"] = cu_seqlens_unpadded_argmin
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_params)

    check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
    with straggler_timer:
        output_tensor = model(**forward_args)

    loss_function = partial(
        masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )

    return output_tensor, loss_function
