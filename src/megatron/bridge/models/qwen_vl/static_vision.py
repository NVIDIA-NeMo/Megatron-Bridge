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

"""Opt-in static-shape vision inputs for Qwen3-VL / Qwen3.5-VL (``model.vision_static_shapes``).

The dynamic vision path reads ``image_grid_thw`` on the host inside the training step (MRoPE ids, vision rotary
ids, position-embedding interpolation, ``cu_seqlens``) and scatters the vision embeddings into the LLM sequence
with a boolean mask. None of that can be captured in a full-iteration CUDA graph. In static mode the collator (or
the task encoder) computes all of it on the CPU, padded to a fixed patch cap, so every microbatch has the same
tensor shapes and dtypes and the training step runs without host synchronisation.

Vision attention is per image, as in the eager path and in Hugging Face: THD attention with static ``cu_seqlens``
(one segment per image slot plus one trailing padding segment) and ``max_seqlen`` = patch cap as a Python int.
"""

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Tuple

import torch


logger = logging.getLogger(__name__)

# Tensors the static step moves to the GPU and hands to the model. [0] is the 3D MRoPE ids; [1:] are the inputs of
# Qwen3VLVisionModel.forward_static plus the scatter index.
STATIC_VISION_KEYS = (
    "mrope_position_ids",
    "vision_pixel_values",
    "vision_cu_seqlens",
    "vision_patch_is_real",
    "vision_pos_embed_idx",
    "vision_pos_embed_w",
    "vision_rope_pos_ids",
    "vision_dst_index",
)

# Host-only Python ints, popped by StaticVisionFlopsIterator before the batch reaches the step or the graph loader.
STATIC_VISION_FLOP_KEYS = (
    "flops_vision_patch_sum",
    "flops_vision_patch_sq_sum",
    "flops_vision_merged_token_sum",
)

# Keys of the dynamic path that the static step never reads; dropped to save H2D copies and static buffers.
_DROPPED_KEYS = (
    "mm_token_type_ids",
    "second_per_grid_ts",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
)

OVER_CAP_POLICIES = ("downscale", "error")


class StaticVisionOverCapError(ValueError):
    """A sample or microbatch does not fit the static vision patch cap."""


@dataclass(frozen=True)
class QwenVLStaticVisionSpec:
    """Static vision layout shared by the data pipeline and the model.

    Built from the model config by :func:`configure_static_vision`; users do not set it.
    """

    patch_cap: int  # P: vision patches per microbatch (before the spatial merge)
    max_images: int  # N: image slots per microbatch
    seq_length: int  # S: LLM sequence length (== model.seq_length)
    spatial_merge_size: int = 2
    patch_size: int = 16
    temporal_patch_size: int = 2
    num_grid_per_side: int = 48  # int(sqrt(num_position_embeddings))
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    over_cap_policy: str = "downscale"

    def __post_init__(self) -> None:
        unit = self.spatial_merge_size**2
        if self.patch_cap <= 0 or self.patch_cap % unit != 0:
            raise ValueError(f"static vision: patch_cap={self.patch_cap} must be a positive multiple of {unit}")
        if self.max_images < 1:
            raise ValueError(f"static vision: max_images={self.max_images} must be >= 1")
        if self.patch_cap < unit * (self.max_images + 1):
            raise ValueError(
                f"static vision: patch_cap={self.patch_cap} < {unit} * (max_images + 1); "
                "no room for the dummy and padding segments"
            )
        if self.seq_length <= 0:
            raise ValueError(f"static vision: seq_length={self.seq_length} must be positive")
        if self.over_cap_policy not in OVER_CAP_POLICIES:
            raise ValueError(f"static vision: over_cap_policy={self.over_cap_policy!r} not in {OVER_CAP_POLICIES}")

    @property
    def merge_unit(self) -> int:
        """Patches per merged vision token."""
        return self.spatial_merge_size**2

    @property
    def num_merged_rows(self) -> int:
        """Rows of the vision tower output (M = P / merge_unit)."""
        return self.patch_cap // self.merge_unit

    @property
    def patch_dim(self) -> int:
        """Flattened pixel values per patch."""
        return 3 * self.temporal_patch_size * self.patch_size * self.patch_size

    @property
    def per_image_patch_budget(self) -> int:
        """Largest per-image patch count that always fits, whatever the other slots hold."""
        unit = self.merge_unit
        return ((self.patch_cap - unit) // self.max_images) // unit * unit


# ---------------------------------------------------------------------------------------------------------------
# CPU replicas of the vision tower's host loops (Qwen3VLVisionModel.fast_pos_embed_interpolate / rot_pos_emb).
# They must stay bitwise equal to the model functions; the unit tests compare them.
# ---------------------------------------------------------------------------------------------------------------


def _merge_order_perm(h: int, w: int, merge: int) -> torch.Tensor:
    """Raster index of each patch in merge order (block_row, block_col, intra_row, intra_col)."""
    return torch.arange(h * w).view(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).reshape(-1)


def pos_embed_interp_static(
    grid_thw: torch.Tensor, num_grid_per_side: int = 48, spatial_merge_size: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinear position-embedding corner indices ``[4, n]`` (int64) and weights ``[4, n]`` (fp32), merge order.

    The index and weight math is a CPU copy of ``Qwen3VLVisionModel.fast_pos_embed_interpolate`` (the model also
    runs it on the CPU). The model then permutes the summed embeddings into merge order; that permutation and the
    frame repeat are applied here to the indices and weights instead, which gives the same rows because the
    lookup, the products and the sum are row-wise.
    """
    n_side = num_grid_per_side
    idx_parts, w_parts = [], []
    for t, h, w in grid_thw.reshape(-1, 3).tolist():
        h_idxs = torch.linspace(0, n_side - 1, h)
        w_idxs = torch.linspace(0, n_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=n_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=n_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * n_side
        base_h_ceil = h_idxs_ceil * n_side

        indices = torch.stack(
            [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
        ).to(torch.long)
        weights = torch.stack(
            [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
        ).to(torch.float32)

        perm = _merge_order_perm(h, w, spatial_merge_size)
        idx_parts.append(indices[:, perm].repeat(1, t))
        w_parts.append(weights[:, perm].repeat(1, t))

    if not idx_parts:
        return torch.zeros((4, 0), dtype=torch.long), torch.zeros((4, 0), dtype=torch.float32)
    return torch.cat(idx_parts, dim=1).contiguous(), torch.cat(w_parts, dim=1).contiguous()


def rot_pos_ids_static(grid_thw: torch.Tensor, spatial_merge_size: int = 2) -> torch.Tensor:
    """(row, col) of every patch in merge order, ``[n, 2]`` int64; the id loop of ``rot_pos_emb``."""
    m = spatial_merge_size
    parts = []
    for t, h, w in grid_thw.reshape(-1, 3).tolist():
        merged_h, merged_w = h // m, w // m
        block_rows = torch.arange(merged_h)
        block_cols = torch.arange(merged_w)
        intra_row = torch.arange(m)
        intra_col = torch.arange(m)
        row_idx = block_rows[:, None, None, None] * m + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * m + intra_col[None, None, None, :]
        row_idx = row_idx.expand(merged_h, merged_w, m, m).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, m, m).reshape(-1)
        coords = torch.stack((row_idx, col_idx), dim=-1)
        if t > 1:
            coords = coords.repeat(t, 1)
        parts.append(coords)
    if not parts:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.cat(parts, dim=0).to(torch.long).contiguous()


# ---------------------------------------------------------------------------------------------------------------
# Samples over the patch cap. The graph cannot fall back, so they are resized or rejected on the CPU.
# ---------------------------------------------------------------------------------------------------------------

_OVER_CAP_STATS = {"images": 0, "downscaled": 0}


def over_cap_stats() -> dict:
    """Per-process counters of images seen / downscaled by the static vision pre-check."""
    return dict(_OVER_CAP_STATS)


def _count(key: str, n: int = 1) -> None:
    before = _OVER_CAP_STATS["images"]
    _OVER_CAP_STATS[key] += n
    if key == "images" and before // 1000 != _OVER_CAP_STATS["images"] // 1000:
        logger.info(
            "static vision over-cap pre-check: %d images, %d downscaled",
            _OVER_CAP_STATS["images"],
            _OVER_CAP_STATS["downscaled"],
        )


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    return smart_resize(height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)


def predict_image_patches(
    height: int, width: int, *, patch_size: int, merge_size: int, min_pixels: int, max_pixels: int
) -> int:
    """Patch count the Qwen2-VL-family image processor produces for an image of this size."""
    h_bar, w_bar = _smart_resize(height, width, patch_size * merge_size, min_pixels, max_pixels)
    return (h_bar // patch_size) * (w_bar // patch_size)


def fit_image_to_patch_budget(
    image: Any,
    budget: int,
    *,
    patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
) -> Tuple[Any, bool]:
    """Downscale a PIL image (aspect ratio kept) so the processor yields at most ``budget`` patches.

    Returns ``(image, downscaled)``. Raises :class:`StaticVisionOverCapError` when no size fits (the processor's
    ``min_pixels`` upscaling would undo any downscale). The processor's real grid is re-checked later by
    :func:`build_static_vision_batch`.
    """
    width, height = image.size
    kw = dict(patch_size=patch_size, merge_size=merge_size, min_pixels=min_pixels, max_pixels=max_pixels)
    try:
        if predict_image_patches(height, width, **kw) <= budget:
            return image, False
        factor = patch_size * merge_size
        target = budget * patch_size * patch_size
        while target >= factor * factor:
            h_bar, w_bar = _smart_resize(height, width, factor, factor * factor, target)
            if predict_image_patches(h_bar, w_bar, **kw) <= budget:
                from PIL import Image

                return image.resize((w_bar, h_bar), resample=Image.BICUBIC), True
            target -= factor * factor
    except ValueError as err:  # smart_resize rejects extreme aspect ratios
        raise StaticVisionOverCapError(f"static vision: cannot fit image {width}x{height}: {err}") from err
    raise StaticVisionOverCapError(f"static vision: image {width}x{height} cannot fit {budget} patches")


def _processor_pixel_bounds(processor: Any, min_pixels: Optional[int], max_pixels: Optional[int]):
    image_processor = getattr(processor, "image_processor", processor)
    size = getattr(image_processor, "size", None) or {}
    if min_pixels is None:
        min_pixels = getattr(image_processor, "min_pixels", None) or size.get("shortest_edge")
    if max_pixels is None:
        max_pixels = getattr(image_processor, "max_pixels", None) or size.get("longest_edge")
    patch_size = int(getattr(image_processor, "patch_size", 16))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    return int(min_pixels), int(max_pixels), patch_size, merge_size


def fit_example_images_to_static_budget(
    per_example_images: Sequence[list],
    per_example_videos: Sequence[list],
    processor: Any,
    spec: QwenVLStaticVisionSpec,
    *,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> list:
    """Pre-check the images of one microbatch against the static layout before the processor runs.

    Videos are rejected; the image count must fit ``spec.max_images``. Each image over the per-image budget is
    downscaled (policy ``downscale``) or rejected with :class:`StaticVisionOverCapError` (policy ``error``).
    """
    if any(len(v) for v in per_example_videos):
        raise ValueError("static vision: videos are not supported")
    n_img = sum(len(imgs) for imgs in per_example_images)
    if n_img > spec.max_images:
        raise StaticVisionOverCapError(f"static vision: {n_img} images in a microbatch > max_images {spec.max_images}")
    if n_img == 0:
        return [list(imgs) for imgs in per_example_images]
    lo, hi, patch_size, merge_size = _processor_pixel_bounds(processor, min_pixels, max_pixels)
    if patch_size != spec.patch_size or merge_size != spec.spatial_merge_size:
        raise ValueError(
            f"static vision: processor patch/merge {patch_size}/{merge_size} != model "
            f"{spec.patch_size}/{spec.spatial_merge_size}"
        )
    budget = spec.per_image_patch_budget
    out = []
    for imgs in per_example_images:
        fitted = []
        for image in imgs:
            _count("images")
            if not hasattr(image, "size"):
                raise TypeError(f"static vision: expected PIL images, got {type(image)}")
            if spec.over_cap_policy == "downscale":
                image, downscaled = fit_image_to_patch_budget(
                    image, budget, patch_size=patch_size, merge_size=merge_size, min_pixels=lo, max_pixels=hi
                )
                if downscaled:
                    _count("downscaled")
            else:
                w, h = image.size
                n = predict_image_patches(
                    h, w, patch_size=patch_size, merge_size=merge_size, min_pixels=lo, max_pixels=hi
                )
                if n > budget:
                    raise StaticVisionOverCapError(f"static vision: image {w}x{h} gives {n} > {budget} patches")
            fitted.append(image)
        out.append(fitted)
    return out


# ---------------------------------------------------------------------------------------------------------------
# The batch converter (CPU, dataloader worker).
# ---------------------------------------------------------------------------------------------------------------


def _visual_tensors(batch: dict) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
    """Return (pixel_values, image_grid_thw, has_video) from ``visual_inputs`` or the top-level keys."""
    vi = batch.pop("visual_inputs", None)
    values = {}
    if vi is not None:
        values = vi.normalized_for_model() if hasattr(vi, "normalized_for_model") else dict(vars(vi))
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if values.get(key) is None and batch.get(key) is not None:
            values[key] = batch[key]
    has_video = values.get("pixel_values_videos") is not None or values.get("video_grid_thw") is not None
    return values.get("pixel_values"), values.get("image_grid_thw"), has_video


def build_static_vision_batch(batch: dict, spec: QwenVLStaticVisionSpec) -> dict:
    """Convert one collated microbatch to the static vision layout.

    With P = ``spec.patch_cap``, N = ``spec.max_images``, M = P / merge unit and the batch padded to [B, S], the
    microbatch gets the same tensor shapes and dtypes every time: ``mrope_position_ids`` [3, B, S] (what
    ``Qwen3VLModel`` computes with ``get_rope_index``), ``vision_pixel_values`` [P, C*T*p*p] (real patches first,
    zero rows after), ``vision_cu_seqlens`` [N + 2] int32 (one segment per image slot, unused slots as one-merge-unit
    dummies, then a padding segment ending at P), ``vision_patch_is_real`` [P], ``vision_pos_embed_idx`` /
    ``vision_pos_embed_w`` [4, P], ``vision_rope_pos_ids`` [P, 2] and ``vision_dst_index`` [M] (flat ``b * S + s``
    position of each image token, unique dummy slots >= B * S for padded rows). ``visual_inputs`` becomes None, the
    dynamic-path keys are dropped, and the host FLOP statistics are added as Python ints
    (``STATIC_VISION_FLOP_KEYS``).

    Runs on the CPU in the collator (HF path) or in the task encoder's ``encode_batch`` (Energon path). Every check
    is a CPU check: a bad batch must fail here, never inside the captured graph.
    """
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
    from megatron.bridge.training.utils.flop_utils import vision_patch_stats_from_grid_thw

    unit = spec.merge_unit
    P = spec.patch_cap
    N = spec.max_images
    M = spec.num_merged_rows

    input_ids = batch["input_ids"]
    if input_ids.dim() != 2:
        raise ValueError(f"static vision: input_ids must be [B, S], got {tuple(input_ids.shape)}")
    input_ids = input_ids.cpu()
    B, S = input_ids.shape
    if S != spec.seq_length:
        raise ValueError(
            f"static vision: input_ids length {S} != seq_length {spec.seq_length} "
            "(needs pad_to_max_length=True and dataset seq_length == model seq_length)"
        )

    pv, grid, has_video = _visual_tensors(batch)
    if has_video or bool((input_ids == spec.video_token_id).any()):
        raise ValueError("static vision: videos are not supported")

    grid = torch.zeros((0, 3), dtype=torch.long) if grid is None else grid.reshape(-1, 3).to(torch.long).cpu()
    n_img = int(grid.shape[0])
    if n_img > N:
        raise StaticVisionOverCapError(f"static vision: {n_img} images > max_images {N}")
    if n_img:
        if not bool((grid[:, 0] == 1).all()):
            raise ValueError(f"static vision: only single-frame images are supported, grid={grid.tolist()}")
        if not bool((grid[:, 1:] % spec.spatial_merge_size == 0).all()):
            raise ValueError(f"static vision: grid h/w must be multiples of the merge size, grid={grid.tolist()}")
    per_image = grid.prod(-1)
    n_real = int(per_image.sum())
    if n_img:
        if pv is None or pv.shape[0] != n_real:
            raise ValueError(
                f"static vision: pixel_values rows {None if pv is None else pv.shape[0]} != patches {n_real}"
            )
        pv = pv.reshape(n_real, -1)
        if pv.shape[1] != spec.patch_dim:
            raise ValueError(f"static vision: patch dim {pv.shape[1]} != {spec.patch_dim}")
    if n_real + unit * (N - n_img + 1) > P:
        raise StaticVisionOverCapError(
            f"static vision: {n_real} patches + {unit * (N - n_img + 1)} dummy/padding rows > patch_cap {P}"
        )
    is_image_token = input_ids == spec.image_token_id
    n_img_tokens = int(is_image_token.sum())
    if n_img_tokens != n_real // unit:
        raise ValueError(
            f"static vision: {n_img_tokens} image tokens != {n_real // unit} merged patches (truncated image span?)"
        )

    # 3D MRoPE ids: the call of Qwen3VLModel.forward, on the CPU, with attention_mask=None (pads get text ids).
    mrope, _ = get_rope_index(
        spec.spatial_merge_size,
        spec.image_token_id,
        spec.video_token_id,
        spec.vision_start_token_id,
        input_ids,
        image_grid_thw=grid if n_img else None,
        video_grid_thw=None,
        attention_mask=None,
        packed_seq_params=None,
    )
    mrope = mrope.to(torch.long).contiguous()
    if tuple(mrope.shape) != (3, B, S):
        raise ValueError(f"static vision: MRoPE ids shape {tuple(mrope.shape)} != {(3, B, S)}")

    pixels = torch.zeros((P, spec.patch_dim), dtype=torch.float32)
    patch_is_real = torch.zeros((P,), dtype=torch.bool)
    pos_idx = torch.zeros((4, P), dtype=torch.long)
    pos_w = torch.zeros((4, P), dtype=torch.float32)
    rope_ids = torch.zeros((P, 2), dtype=torch.long)
    if n_img:
        pixels[:n_real] = pv.to(torch.float32)
        patch_is_real[:n_real] = True
        idx_r, w_r = pos_embed_interp_static(grid, spec.num_grid_per_side, spec.spatial_merge_size)
        pos_idx[:, :n_real] = idx_r
        pos_w[:, :n_real] = w_r
        rope_ids[:n_real] = rot_pos_ids_static(grid, spec.spatial_merge_size)

    cu = [0]
    for length in per_image.tolist() + [unit] * (N - n_img):
        cu.append(cu[-1] + int(length))
    cu.append(P)
    if any(b <= a for a, b in zip(cu[:-1], cu[1:])) or P - cu[-2] < unit:
        raise ValueError(f"static vision: bad cu_seqlens {cu}")
    cu_seqlens = torch.tensor(cu, dtype=torch.int32)

    flat_pos = torch.nonzero(is_image_token.reshape(-1), as_tuple=False).squeeze(1)
    dst_index = torch.arange(B * S, B * S + M, dtype=torch.long)
    dst_index[: flat_pos.numel()] = flat_pos

    if n_img:
        patch_sum, patch_sq_sum, merged_sum = (
            int(x) for x in vision_patch_stats_from_grid_thw(grid, spatial_merge_size=spec.spatial_merge_size)
        )
    else:
        patch_sum = patch_sq_sum = merged_sum = 0

    for key in _DROPPED_KEYS:
        batch.pop(key, None)
    batch["visual_inputs"] = None
    batch["mrope_position_ids"] = mrope
    batch["vision_pixel_values"] = pixels
    batch["vision_cu_seqlens"] = cu_seqlens
    batch["vision_patch_is_real"] = patch_is_real
    batch["vision_pos_embed_idx"] = pos_idx
    batch["vision_pos_embed_w"] = pos_w
    batch["vision_rope_pos_ids"] = rope_ids
    batch["vision_dst_index"] = dst_index
    batch["flops_vision_patch_sum"] = patch_sum
    batch["flops_vision_patch_sq_sum"] = patch_sq_sum
    batch["flops_vision_merged_token_sum"] = merged_sum
    return batch


# ---------------------------------------------------------------------------------------------------------------
# Model-side helper.
# ---------------------------------------------------------------------------------------------------------------


def scatter_vision_embeds_static(
    combined_sbh: torch.Tensor, vision_embeds: torch.Tensor, dst_index: torch.Tensor
) -> torch.Tensor:
    """Sync-free replacement of ``combined[vision_mask] = vision_embeds`` for ``[S, B, H]`` embeddings.

    ``dst_index[k]`` is the flat ``b * S + s`` position of the k-th image token for real merged rows and a unique
    dummy slot ``>= B * S`` for padded rows, so ``index_copy`` never sees a duplicate index and needs no atomics.
    """
    S, B, H = combined_sbh.shape
    M = dst_index.shape[0]
    flat = combined_sbh.transpose(0, 1).reshape(B * S, H)
    flat = torch.cat([flat, flat.new_zeros(M, H)], dim=0)
    flat = torch.index_copy(flat, 0, dst_index, vision_embeds.to(flat.dtype))
    return flat[: B * S].view(B, S, H).transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------------------------------------------
# Host FLOP statistics outside the graph.
# ---------------------------------------------------------------------------------------------------------------


class StaticVisionFlopsIterator:
    """Wrap the step's data iterator: pop the host FLOP ints and add them to the GlobalState accumulators.

    Installed in ``train_step`` right before ``forward_backward_func``. ``FullCudaGraphWrapper.data_read`` (graph
    runs) and ``get_batch`` (eager runs) call ``next()`` exactly once per microbatch, so eager and graph runs
    report the same FLOPs. The values equal what ``accumulate_flops_metadata`` adds on the dynamic path: every
    sequence is padded to ``S``, and the vision statistics count the real patches.
    """

    def __init__(self, data_iterator: Iterator[dict], global_state: Any):
        self._it = data_iterator
        self._state = global_state

    def __iter__(self) -> "StaticVisionFlopsIterator":
        return self

    def __next__(self) -> dict:
        from megatron.bridge.training.utils.flop_utils import _add_flops_accumulator

        batch = next(self._it)
        if not isinstance(batch, dict) or "vision_dst_index" not in batch:
            raise RuntimeError("static vision: the data iterator did not yield a static vision batch")
        B, S = batch["input_ids"].shape
        _add_flops_accumulator(self._state, "_flops_seqlen_sum", B * S)
        _add_flops_accumulator(self._state, "_flops_seqlen_sq_sum", B * S * S)
        _add_flops_accumulator(self._state, "_flops_vision_patch_sum", int(batch.pop("flops_vision_patch_sum")))
        _add_flops_accumulator(self._state, "_flops_vision_patch_sq_sum", int(batch.pop("flops_vision_patch_sq_sum")))
        _add_flops_accumulator(
            self._state, "_flops_vision_merged_token_sum", int(batch.pop("flops_vision_merged_token_sum"))
        )
        setattr(self._state, "_flops_requires_global_reduce", True)
        return batch


# ---------------------------------------------------------------------------------------------------------------
# Config wiring (called from ConfigContainer.validate when model.vision_static_shapes is set).
# ---------------------------------------------------------------------------------------------------------------


def configure_static_vision(cfg: Any) -> QwenVLStaticVisionSpec:
    """Validate a static-vision run and hand the derived spec to the dataset config.

    The model config is the single source of truth: the dataset's ``static_vision`` field is overwritten with the
    spec built here.
    """
    model = cfg.model
    dataset = cfg.dataset
    errors = []
    patch_cap = getattr(model, "vision_patch_cap", None)
    max_images = getattr(model, "vision_max_images", 1)
    vision_config = getattr(model, "vision_config", None)
    merge = int(getattr(vision_config, "spatial_merge_size", getattr(model, "spatial_merge_size", 2)))
    if not isinstance(patch_cap, int) or patch_cap <= 0 or patch_cap % (merge * merge) != 0:
        errors.append(f"model.vision_patch_cap must be a positive multiple of {merge * merge} (got {patch_cap!r})")
    if not isinstance(max_images, int) or max_images < 1:
        errors.append(f"model.vision_max_images must be >= 1 (got {max_images!r})")
    if getattr(model, "vision_cuda_graph_impl", "none") != "none":
        errors.append("model.vision_cuda_graph_impl must be 'none' (per-layer vision graphs use the padded path)")
    if (getattr(model, "context_parallel_size", 1) or 1) != 1:
        errors.append("context_parallel_size must be 1")
    if getattr(model, "sequence_parallel", False):
        errors.append("sequence_parallel must be False")
    if (getattr(model, "pipeline_model_parallel_size", 1) or 1) != 1:
        errors.append("pipeline_model_parallel_size must be 1")
    if getattr(model, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        errors.append("virtual pipeline parallelism is not supported")
    dist_train = getattr(model, "dist_train", None)
    if dist_train is not None and getattr(dist_train, "use_dist_train", False):
        errors.append("dist_train is not supported")
    # The spec goes to the dataset config (mock VLM SFT) or to its HF energon task-encoder config.
    target = None
    if dataset is not None and hasattr(dataset, "static_vision"):
        target = dataset
    elif dataset is not None and hasattr(getattr(dataset, "task_encoder", None), "static_vision"):
        target = dataset.task_encoder
        if getattr(dataset, "packing_buffer_size", None) is not None:
            errors.append("energon packing (dataset.packing_buffer_size) is not supported")
    if target is None:
        name = type(dataset).__name__
        if dataset is not None and getattr(dataset, "task_encoder", None) is not None:
            name += f" / {type(dataset.task_encoder).__name__}"
        errors.append(f"dataset config {name} does not support static vision")
    else:
        if not getattr(dataset, "pad_to_max_length", False):
            errors.append("dataset.pad_to_max_length must be True")
        if getattr(dataset, "enable_in_batch_packing", False):
            errors.append("dataset.enable_in_batch_packing must be False")
        ds_len = getattr(dataset, "seq_length", None)
        if ds_len is not None and ds_len != model.seq_length:
            errors.append(f"dataset.seq_length {ds_len} != model.seq_length {model.seq_length}")
    rerun = getattr(cfg, "rerun_state_machine", None)
    if rerun is not None and getattr(rerun, "rerun_mode", "disabled") != "disabled":
        errors.append("rerun_state_machine.rerun_mode must be 'disabled' (a rerun would re-read data)")
    if isinstance(getattr(model, "sequence_packing_scheduler", None), str):
        errors.append("global batch packing (model.sequence_packing_scheduler) is not supported")
    policy = getattr(model, "vision_over_cap_policy", "downscale")
    if policy not in OVER_CAP_POLICIES:
        errors.append(f"model.vision_over_cap_policy must be one of {OVER_CAP_POLICIES} (got {policy!r})")
    if errors:
        raise ValueError("model.vision_static_shapes: " + "; ".join(errors))
    from megatron.bridge.utils.cuda_graph import is_full_iteration_cuda_graph

    if is_full_iteration_cuda_graph(model) and getattr(dataset, "prefetch_factor", 0) is None:
        logger.warning(
            "static vision + full_iteration: FullCudaGraphWrapper.data_read takes all microbatches of an iteration "
            "at once, right after the previous step's host sync, so with the loader's default prefetch (2 batches "
            "per worker) the GPU idles while the workers produce the rest. Set dataset.prefetch_factor so that "
            "prefetch_factor * max(num_workers, 1) >= the microbatches per iteration."
        )

    n_pos = int(getattr(vision_config, "num_position_embeddings", 2304))
    spec = QwenVLStaticVisionSpec(
        patch_cap=patch_cap,
        max_images=max_images,
        seq_length=int(model.seq_length),
        spatial_merge_size=merge,
        patch_size=int(getattr(vision_config, "patch_size", getattr(model, "patch_size", 16))),
        temporal_patch_size=int(
            getattr(vision_config, "temporal_patch_size", getattr(model, "temporal_patch_size", 2))
        ),
        num_grid_per_side=int(n_pos**0.5),
        image_token_id=int(model.image_token_id),
        video_token_id=int(model.video_token_id),
        vision_start_token_id=int(model.vision_start_token_id),
        over_cap_policy=policy,
    )
    target.static_vision = spec
    return spec
