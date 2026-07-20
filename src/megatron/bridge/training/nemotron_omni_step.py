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

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import get_packed_seq_params
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.pg_utils import get_pg_collection


logger = logging.getLogger(__name__)

_PACKED_SEQ_DEVICE_KEYS = ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")
_PACKED_SEQ_HOST_KEYS = ("max_seqlen_q", "max_seqlen_kv")
_PACKED_SEQ_PARAM_KEYS = (*_PACKED_SEQ_DEVICE_KEYS, *_PACKED_SEQ_HOST_KEYS, "total_tokens")


# ---------------------------------------------------------------------------
# Debug: dump loss-active tokens (labels != IGNORE_INDEX) on step 0.
# Enable with NEMOTRON_OMNI_DUMP_MASK=1. Imports scoped to this block only.
# ---------------------------------------------------------------------------
import os as _os_dump  # noqa: E402
import torch.distributed as _dist_dump  # noqa: E402

from megatron.bridge.data.datasets.utils import IGNORE_INDEX as _IGNORE_INDEX_DUMP  # noqa: E402


def _dump_loss_active_tokens(
    state: GlobalState,
    input_ids: torch.Tensor | None,
    labels: torch.Tensor | None,
    packed_seq_params: dict | None,
) -> None:
    """Decode and print loss-active tokens on step 0 when NEMOTRON_OMNI_DUMP_MASK=1.

    Rank 0 prints, on the first training step, the token spans where
    ``labels != IGNORE_INDEX`` — the tokens the model is being trained to
    predict. Loads the real HF tokenizer from the recipe's Energon
    task-encoder ``hf_processor_path`` because Nemotron Omni uses NullTokenizer
    at the Megatron layer. Prints the full row's loss-active text; does NOT
    try to split by cu_seqlens_q because those positions are in image-expanded
    coordinates, not compact token positions.
    """
    if _os_dump.environ.get("NEMOTRON_OMNI_DUMP_MASK", "0") != "1":
        return
    if input_ids is None or labels is None:
        return
    if state.train_state.step != 0:
        return
    if _dist_dump.is_initialized() and _dist_dump.get_rank() != 0:
        return

    # Prefer the HF processor tokenizer (real BPE); fall back to state.tokenizer.
    tokenizer = None
    try:
        task_cfg = getattr(state.cfg.dataset, "task_encoder", None)
        hf_path = getattr(task_cfg, "hf_processor_path", None) if task_cfg is not None else None
        if hf_path:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    except Exception as exc:
        print(f"[NEMOTRON_OMNI_DUMP_MASK] AutoTokenizer load failed: {exc!r}", flush=True)
    if tokenizer is None:
        try:
            tokenizer = state.tokenizer
        except Exception:
            tokenizer = None

    def _decode(ids: list[int]) -> str:
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                return tokenizer.decode(ids, skip_special_tokens=False)
            except Exception as exc:
                return f"<decode failed {exc!r}; raw ids: {ids[:32]}{'...' if len(ids) > 32 else ''}>"
        return f"<{len(ids)} raw token ids: {ids[:32]}{'...' if len(ids) > 32 else ''}>"

    ids2 = input_ids.detach().cpu()
    lbl2 = labels.detach().cpu()
    if ids2.dim() == 1:
        ids2 = ids2.unsqueeze(0)
        lbl2 = lbl2.unsqueeze(0)

    cu = None
    if packed_seq_params is not None:
        cu_tensor = packed_seq_params.get("cu_seqlens_q")
        if cu_tensor is not None:
            cu = cu_tensor.detach().cpu().tolist()

    print("=" * 80, flush=True)
    print("[NEMOTRON_OMNI_DUMP_MASK] loss-active tokens (labels != IGNORE_INDEX)", flush=True)
    print(f"  input_ids.shape={tuple(input_ids.shape)}  labels.shape={tuple(labels.shape)}", flush=True)
    if cu is not None:
        print(
            f"  cu_seqlens_q={cu}  (image-expanded coords; total merged={cu[-1]}, "
            f"compact packed width={ids2.size(1)})",
            flush=True,
        )
    print("=" * 80, flush=True)

    for row in range(ids2.size(0)):
        row_ids = ids2[row]
        row_lbl = lbl2[row]
        active_mask = row_lbl != _IGNORE_INDEX_DUMP
        full_active_ids = row_ids[active_mask].tolist()
        active_positions = active_mask.nonzero(as_tuple=False).flatten().tolist()
        first_pos = active_positions[0] if active_positions else None
        last_pos = active_positions[-1] if active_positions else None
        print(
            f"[row {row}] total loss-active tokens: {len(full_active_ids)} "
            f"(first_pos={first_pos}, last_pos={last_pos})",
            flush=True,
        )
        # Also dump the full row (compact form) so we can see the full context
        # around the assistant spans.
        print(f"[row {row}] FULL compact row decoded:\n{_decode(row_ids.tolist())!r}\n", flush=True)
        print(f"[row {row}] LOSS-ACTIVE tokens decoded (concatenated across all assistant turns):", flush=True)
        print(f"  {_decode(full_active_ids)!r}\n", flush=True)
        # Contiguous-run split: each maximal contiguous unmasked run = one
        # assistant turn's loss span. This is the real "per-turn" view.
        if active_positions:
            runs: list[tuple[int, int]] = []
            run_start = active_positions[0]
            prev = active_positions[0]
            for p in active_positions[1:]:
                if p == prev + 1:
                    prev = p
                else:
                    runs.append((run_start, prev + 1))
                    run_start = p
                    prev = p
            runs.append((run_start, prev + 1))
            print(f"[row {row}] found {len(runs)} contiguous loss-active span(s):", flush=True)
            for k, (s, e) in enumerate(runs):
                span_ids = row_ids[s:e].tolist()
                print(
                    f"  span[{k}] positions=[{s}:{e}] len={e - s} decoded={_decode(span_ids)!r}",
                    flush=True,
                )
    print("=" * 80, flush=True)


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

    if is_first_pp_stage:
        # The first stage owns the vision and sound encoders.
        required_device_keys.update(
            key
            for key in (
                "pixel_values",
                "num_patches",
                "visual_inputs",
                "sound_clips",
                "sound_length",
                "imgs_sizes",
                "num_frames",
                "num_image_tiles",
            )
            if key in batch
        )

    if "cu_seqlens_q" in batch:
        required_device_keys.update(key for key in _PACKED_SEQ_DEVICE_KEYS if key in batch)
        required_host_keys.update(key for key in _PACKED_SEQ_HOST_KEYS if key in batch)

    if is_first_pp_stage or is_last_pp_stage:
        input_key = "tokens" if batch.get("tokens") is not None else "input_ids"
        required_device_keys.add(input_key)
    if is_first_pp_stage:
        required_device_keys.add("position_ids")
    if is_last_pp_stage:
        required_device_keys.update(("labels", "loss_mask"))
        if "num_image_tiles" in batch:
            # LLaVA expands labels around image placeholders on the last stage.
            required_device_keys.add("num_image_tiles")

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
        elif key == "total_tokens" and "cu_seqlens_q" in batch:
            _batch_required_keys[key] = int(val) if val is not None else None
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


def _uses_packed_sequence_metadata(cfg: ConfigContainer) -> bool:
    """Return whether every decoder PP stage needs THD boundary metadata."""
    dataset_cfg = getattr(cfg, "dataset", None)
    offline_packing_specs = getattr(dataset_cfg, "offline_packing_specs", None)
    if getattr(dataset_cfg, "enable_offline_packing", False):
        packed_sequence_size = getattr(offline_packing_specs, "packed_sequence_size", None)
        return packed_sequence_size is None or packed_sequence_size > 0
    return bool(getattr(dataset_cfg, "enable_in_batch_packing", False))


def get_batch(data_iterator: Iterable, cfg: ConfigContainer, *, pg_collection) -> tuple:
    """Generate a batch with vision and sound tensors."""
    is_first = is_pp_first_stage(pg_collection.pp)
    is_last = is_pp_last_stage(pg_collection.pp)
    skip_attention_mask = getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True)
    if (not is_first) and (not is_last) and skip_attention_mask and not _uses_packed_sequence_metadata(cfg):
        return (None,) * 14

    batch = get_batch_from_iterator(
        data_iterator,
        skip_attention_mask,
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

    input_ids = batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")
    if is_first or is_last:
        assert input_ids is not None

    return (
        batch.get("images"),
        batch.get("num_patches"),
        input_ids,
        batch.get("labels"),
        batch.get("loss_mask"),
        batch.get("attention_mask"),
        batch.get("position_ids"),
        {key: batch[key] for key in _PACKED_SEQ_PARAM_KEYS if batch.get(key) is not None}
        if batch.get("cu_seqlens_q") is not None
        else None,
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
    dataset_cfg = state.cfg.dataset
    if getattr(dataset_cfg, "enable_in_batch_packing", False) and getattr(
        dataset_cfg, "defer_in_batch_packing_to_step", False
    ):
        raise ValueError(
            "nemotron_omni_step requires collate-time in-batch packing; set defer_in_batch_packing_to_step=False"
        )

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
            packed_seq_params,
            sound_clips,
            sound_length,
            imgs_sizes,
            num_frames,
            vision_packed_seq_params,
            num_image_tiles,
        ) = get_batch(data_iterator, state.cfg, pg_collection=pg_collection)
    timers("batch-generator").stop()

<<<<<<< HEAD
    _dump_loss_active_tokens(state, input_ids, labels, packed_seq_params)

=======
>>>>>>> origin/main
    # Encoder and last stages need an empty image sentinel for text/audio-only
    # batches. Middle stages may legitimately have no input_ids or image tensor.
    if images is None and input_ids is not None:
        images = torch.tensor([], dtype=torch.bfloat16, device=input_ids.device).reshape(0, 0, 0)
    elif images is not None and images.dtype != torch.bfloat16:
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
    if num_image_tiles is not None:
        forward_args["num_image_tiles"] = num_image_tiles
    if vision_packed_seq_params is not None:
        forward_args["vision_packed_seq_params"] = vision_packed_seq_params

    if packed_seq_params is not None:
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_params)

    check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
    with straggler_timer:
        model_output = model(**forward_args)
        if isinstance(model_output, tuple):
            output_tensor, model_loss_mask = model_output
            if model_loss_mask is not None:
                loss_mask = model_loss_mask
        else:
            output_tensor = model_output

    # Multimodal expansion can return a strided prefix slice after truncation;
    # normalize it at the model/step boundary before the shared loss reducer.
    if loss_mask is not None:
        loss_mask = loss_mask.contiguous()

    loss_function = partial(
        masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )

    return output_tensor, loss_function
