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

import bisect
import dataclasses
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from megatron.energon import Batch, DefaultTaskEncoder, stateless
from transformers import BatchEncoding

from megatron.bridge.data.energon.task_encoder_utils import (
    IGNORE_INDEX,
    ChatMLSample,
    ChatMLWebdataset,  # noqa: F401  -- re-exported for backward compat
    _images_to_pil,
    _tensor_to_pil,  # noqa: F401  -- re-exported for backward compat
    _videos_to_pil,
    cook_chatml_sample,
    find_pattern_indices,
    get_ltor_masks_and_position_ids,
    videohandler,  # noqa: F401  -- re-exported for backward compat
)
from megatron.bridge.training.utils.visual_inputs import Qwen2_5_VLVisualInputs


logger = logging.getLogger(__name__)
_LOW_CONTENT_LOG_COUNT = 0


def _thd_diag_enabled() -> bool:
    return os.environ.get("THD_DIAG", "0") not in ("0", "", "false", "False")


def _thd_diag_low_content_threshold() -> int:
    raw = os.environ.get("THD_DIAG_LOW_CONTENT_TOKENS", "2048")
    try:
        return max(0, int(raw))
    except ValueError:
        return 2048


def _thd_diag_max_low_content_logs() -> int:
    raw = os.environ.get("THD_DIAG_MAX_LOW_CONTENT_LOGS", "100")
    try:
        return max(0, int(raw))
    except ValueError:
        return 100


def _search_for_fit(numbers: List[int], capacity: int) -> int:
    """Binary search for the largest number that fits within capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(item_sizes: List[int], samples: List, max_capacity: int) -> List:
    """Greedy bin-packing with binary search.

    Sorts samples by length ascending, then greedily fills each bin by picking
    the largest item that still fits (via binary search). Returns a list of bins,
    each bin being a list of samples.
    """
    assert len(item_sizes) == len(samples)
    if not item_sizes:
        return []

    sorted_sizes, sorted_samples = zip(*sorted(zip(item_sizes, samples), key=lambda x: x[0]))
    sorted_sizes = list(sorted_sizes)
    sorted_samples = list(sorted_samples)

    if sorted_sizes[-1] > max_capacity:
        raise ValueError(f"Sample size {sorted_sizes[-1]} exceeds max_capacity {max_capacity}")

    knapsacks = []
    while sorted_sizes:
        current_knapsack = []
        remaining = max_capacity
        while True:
            idx = _search_for_fit(sorted_sizes, remaining)
            if idx == -1:
                break
            remaining -= sorted_sizes[idx]
            sorted_sizes.pop(idx)
            current_knapsack.append(sorted_samples.pop(idx))
        knapsacks.append(current_knapsack)
    return knapsacks


def process_vision(
    processor, images, videos, fps=None, model_version: str = "qwen-vl", min_pixels=None, max_pixels=None
):
    """Minimal vision preprocessing wrapper using the provided processor (e.g., HF AutoProcessor)."""
    if images is not None:
        kwargs = {}
        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        image_inputs = processor(images=images, text="", videos=None, return_tensors="pt", **kwargs)
        image_grid_thw = image_inputs.get("image_grid_thw", None)
    else:
        image_inputs = {}
        image_grid_thw = None

    if videos is not None:
        videos_inputs = processor(images=None, text="", videos=videos, return_tensors="pt")
        video_grid_thw = videos_inputs.get("video_grid_thw", None)
    else:
        videos_inputs = {}
        video_grid_thw = None

    return {
        "image_inputs": image_inputs,
        "image_grid_thw": image_grid_thw,
        "video_inputs": videos_inputs,
        "video_grid_thw": video_grid_thw,
    }


def _resolve_hf_mm_token_ids(hf_tokenizer):
    """Resolve HF tokenizer ids for <image> and <video> tokens without nemo constants."""

    def _get(token_str: str, default_id: int) -> int:
        token_attr = getattr(hf_tokenizer, f"{token_str.strip('<>')}_token_id", None)
        if token_attr is not None:
            return int(token_attr)
        try:
            return int(hf_tokenizer.convert_tokens_to_ids(token_str))
        except Exception:
            return default_id

    image_id = _get("<image>", 151655)
    video_id = _get("<video>", 151656)
    return image_id, video_id


@dataclass
class QwenVLTaskSample:
    """Encoded Sample Format For QwenVL"""

    __key__: str
    __subflavors__: Dict

    imgs: List[torch.Tensor]  # (c, h, w)
    videos: List[torch.Tensor]  # (c, h, w)

    image_thw_grids: List[torch.Tensor]
    video_thw_grids: List[torch.Tensor]
    image_input_mask: torch.Tensor
    video_input_mask: torch.Tensor
    text: torch.Tensor
    target: torch.Tensor
    total_len: int = 0


@dataclass
class QwenVLTaskSamplePacked:
    """Packed sample: multiple QwenVLTaskSample concatenated into one sequence."""

    __key__: str
    __subflavors__: Dict
    tokens: torch.Tensor
    target: torch.Tensor
    imgs: list
    videos: list
    image_thw_grids: list
    video_thw_grids: list
    image_input_mask: torch.Tensor
    video_input_mask: torch.Tensor
    cu_lengths: torch.Tensor
    max_length: int
    num_sub_samples: int
    sub_sample_lengths: list = field(default_factory=list)
    num_image_tokens: int = 0
    num_video_tokens: int = 0
    num_text_tokens: int = 0
    num_vit_patches: int = 0


@dataclass
class QwenVLTaskBatch(Batch):
    """Encoded Batch Format For QwenVL.

    Energon 7.x ``Batch`` requires ``__key__`` and ``__restore_key__`` as
    keyword-only init fields.  We keep ``__keys__`` (plural) for downstream
    consumption but satisfy the parent via ``__key__`` (joined) and
    ``__restore_key__`` (empty tuple placeholder).
    """

    __key__: str
    __restore_key__: tuple
    __keys__: List[str]
    __subflavors__: List[Dict]
    # (num_tiles, c, h, w)
    pixel_values: torch.Tensor
    pixel_values_videos: torch.Tensor
    image_grid_thw: torch.Tensor
    video_grid_thw: torch.Tensor
    image_input_mask: torch.Tensor
    video_input_mask: torch.Tensor
    # (n, seq_len)
    input_ids: torch.Tensor
    # (n, seq_len)
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    # Packing metadata (set when using Energon packing)
    cu_lengths: Optional[torch.Tensor] = None
    max_lengths: Optional[torch.Tensor] = None
    cu_seqlens_argmin: Optional[torch.Tensor] = None


def convert_to_qwenvl_content(user_input: str, image_pattern: str = "<image>", video_pattern: str = "<video>"):
    """Split user input into format QwenVL tokenizer accepts."""

    pattern = r"({image}|{video})".format(image=image_pattern, video=video_pattern)
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        if start > cur:
            contents.append({"type": "text", "text": user_input[cur:start].strip(" ")})

        contents.append(
            {
                "type": matched.string[start:end][1:-1],
                matched.string[start:end][1:-1]: str(mm_idx[matched.string[start:end][1:-1]]),
            }
        )

        cur = end
        mm_idx[matched.string[start:end][1:-1]] += 1

    if cur < len(user_input):
        contents.append({"type": "text", "text": user_input[cur : len(user_input)].strip(" ")})

    return contents


class QwenVLTaskEncoder(DefaultTaskEncoder[ChatMLSample, QwenVLTaskSample, QwenVLTaskBatch, dict]):
    """A simple task encoder for captioning."""

    def __init__(
        self,
        tokenizer,
        image_processor,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        patch_size: int = 14,
        max_padding_length: int = 4096,
        min_pixels: int = 200704,
        max_pixels: int = 1003520,
    ):
        super().__init__()

        self.hf_tokenizer = tokenizer
        self.image_processor = image_processor
        self.seq_length = max_padding_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.temporal_patch_size = temporal_patch_size
        self.merge_size = spatial_merge_size
        self.patch_size = patch_size

        self.seq_len = max_padding_length
        self.image_token_id, self.video_token_id = _resolve_hf_mm_token_ids(self.hf_tokenizer)

    def encode_sample(self, sample: ChatMLSample):
        """
        Encode sample to meet training requirement.

        Args:
            sample.imgs: list[PIL.Image.Image]
            sample.videos: list[Tensor]

        Returns:
            sample with necessary fields
        """
        # NOTE: Convert WDS tensor images to PIL to match HF flow format.
        #     WDS imagehandler decodes JPEG to float tensors in [0,1], but the processor
        #     expects PIL images (uint8 [0,255]) for correct rescaling and normalization.
        imgs_for_processing = _images_to_pil(sample.imgs) if sample.imgs is not None and len(sample.imgs) > 0 else None
        videos_for_processing = (
            _videos_to_pil(sample.videos) if sample.videos is not None and len(sample.videos) > 0 else None
        )
        processed_vision = process_vision(
            self.image_processor,
            imgs_for_processing,
            videos_for_processing,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        image_thw_grids = processed_vision["image_grid_thw"]
        video_thw_grids = processed_vision["video_grid_thw"]
        flattened_imgs = processed_vision["image_inputs"]
        flattened_videos = processed_vision["video_inputs"]

        # Normalize conversation to [{"role": ..., "content": ...}, ...]
        conversation = cook_chatml_sample(sample.conversation)

        # Apply Qwen-specific content formatting for user turns:
        # convert_to_qwenvl_content splits text around <image>/<video> placeholders,
        # then media items are reordered first for PreloadedVLMConversationProvider.
        for turn in conversation:
            if turn["role"] == "user":
                content = convert_to_qwenvl_content(turn["content"])
                media_content = [c for c in content if c.get("type") in ("image", "video")]
                text_content = [c for c in content if c.get("type") == "text"]
                turn["content"] = media_content + text_content

        # NOTE: we need to mask all system/user input tokens and assistant generation prefix tokens
        # In transformers >= 5.0, apply_chat_template returns BatchEncoding when tokenize=True
        chat_output = self.hf_tokenizer.apply_chat_template(conversation, tokenize=True, return_tensors="np")
        input_ids = chat_output["input_ids"][0] if isinstance(chat_output, BatchEncoding) else chat_output[0]
        pad_token_id = self.hf_tokenizer.pad_token_id
        target = [pad_token_id for _ in range(len(input_ids))]
        search_start_index = 0
        for turn_idx, turn in enumerate(conversation[1:]):
            if turn["role"] == "assistant":
                answer = turn["content"]
                answer_tokens = self.hf_tokenizer.encode(answer, add_special_tokens=False)
                answer_start, answer_end = find_pattern_indices(input_ids, answer_tokens, search_start_index)
                assert answer_start > 0, "Not found valid answer in conversation."
                target[answer_start:answer_end] = input_ids[answer_start:answer_end]
                search_start_index = answer_end

        # NOTE: expand image_pad & video_pad
        merge_length = self.merge_size**2
        image_token_id, video_token_id = self.image_token_id, self.video_token_id

        image_token_indices = np.where(input_ids == image_token_id)[0]
        if image_token_indices is not None and image_thw_grids is not None:
            assert len(image_token_indices) == len(image_thw_grids), (
                f"With {len(image_thw_grids)} images in the sample, but {len(image_token_indices)} image placeholders!"
            )
        video_token_indices = np.where(input_ids == video_token_id)[0]
        if video_token_indices is not None and video_thw_grids is not None:
            assert len(video_token_indices) == len(video_thw_grids), (
                f"With {len(video_thw_grids)} videos in the sample, but {len(video_token_indices)} video placeholders!"
            )
        if image_thw_grids is not None and video_thw_grids is not None:
            image_thw_grids, video_thw_grids = (
                np.array(image_thw_grids, dtype=np.int64),
                np.array(video_thw_grids, dtype=np.int64),
            )
            # xxx_thw_grids.shape[0] indicates how many '<image>' or '<video>' inside conversation text,
            # minus it and then get patch number, this would get exact number of visual padding size
            target_length = (
                input_ids.shape[0]
                - image_thw_grids.shape[0]
                + image_thw_grids.prod(axis=-1).sum() // merge_length
                - video_thw_grids.shape[0]
                + video_thw_grids.prod(axis=-1).sum() // merge_length
            )
        elif image_thw_grids is not None:
            image_thw_grids = np.array(image_thw_grids, dtype=np.int64)

            target_length = (
                input_ids.shape[0] - image_thw_grids.shape[0] + image_thw_grids.prod(axis=-1).sum() // merge_length
            )
        elif video_thw_grids is not None:
            video_thw_grids = np.array(video_thw_grids, dtype=np.int64)

            target_length = (
                input_ids.shape[0] - video_thw_grids.shape[0] + video_thw_grids.prod(axis=-1).sum() // merge_length
            )
        else:
            target_length = input_ids.shape[0]

        if target_length > self.seq_len:
            logging.warning(f"Long sequence with length {target_length} found, dropped...")
        final_input_ids = np.zeros(target_length, dtype=input_ids.dtype)
        final_input_masks = final_input_ids.copy()

        image_idx, video_idx = 0, 0
        indices = np.sort(np.concatenate([image_token_indices, video_token_indices]))

        cur_x, cur_y = 0, 0
        for idx in indices:
            token_id = input_ids[idx]
            if token_id == image_token_id:
                size = image_thw_grids[image_idx].prod() // merge_length
                image_idx += 1
            elif token_id == video_token_id:
                size = video_thw_grids[video_idx].prod() // merge_length
                video_idx += 1
            # NOTE:
            # input_ids[cur_x:idx] -> final_input_ids[cur_y:cur_y + idx - cur_x]
            # input_ids[idx] -> final_input_ids[cur_y + idx - cur_x: cur_y + idx - cur_x + size]
            final_input_ids[cur_y : cur_y + idx - cur_x] = input_ids[cur_x:idx]
            final_input_masks[cur_y : cur_y + idx - cur_x] = target[cur_x:idx]
            cur_y += idx - cur_x
            final_input_ids[cur_y : cur_y + size] = token_id
            final_input_masks[cur_y : cur_y + size] = pad_token_id
            cur_y += size
            cur_x = idx + 1

        if cur_x < len(input_ids):
            final_input_ids[cur_y:] = input_ids[cur_x:]
            final_input_masks[cur_y:] = target[cur_x:]

        # left shift token by one for labels.
        target = np.roll(final_input_masks, shift=-1)
        target[-1] = pad_token_id

        if (target == pad_token_id).all():
            logging.warning("Sample with all masked label, dropped.")

        image_input_mask = torch.from_numpy(final_input_ids == image_token_id)
        video_input_mask = torch.from_numpy(final_input_ids == video_token_id)
        # collect data
        return QwenVLTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=flattened_imgs["pixel_values"] if flattened_imgs else [],
            videos=flattened_videos["pixel_values_videos"] if flattened_videos else [],
            image_thw_grids=image_thw_grids if flattened_imgs else [],
            video_thw_grids=video_thw_grids if flattened_videos else [],
            image_input_mask=image_input_mask,
            video_input_mask=video_input_mask,
            text=torch.from_numpy(final_input_ids),
            target=torch.from_numpy(target),
            total_len=int(target_length),
        )

    def select_samples_to_pack(self, samples: List[QwenVLTaskSample]) -> List[List[QwenVLTaskSample]]:
        """Select groups of samples to pack together using greedy knapsack.

        Called by Energon's PackingDataset. Returns list of bins, each bin
        is a list of samples whose total length <= self.seq_len.
        """
        lengths = [s.total_len for s in samples]
        return greedy_knapsack(lengths, samples, self.seq_len)

    @stateless
    def pack_selected_samples(self, samples: List[QwenVLTaskSample]) -> QwenVLTaskSamplePacked:
        """Concatenate multiple QwenVLTaskSample into one packed sample.

        Called by Energon's PackingDataset after select_samples_to_pack.
        Produces a QwenVLTaskSamplePacked with cu_lengths for THD attention.
        """
        all_tokens, all_targets = [], []
        all_imgs, all_videos = [], []
        all_image_thw_grids, all_video_thw_grids = [], []
        all_image_masks, all_video_masks = [], []
        sub_lengths = []
        total_image_tokens = 0
        total_video_tokens = 0
        total_text_tokens = 0
        total_vit_patches = 0
        merge_length = self.merge_size ** 2

        for s in samples:
            seq_len = len(s.text)
            sub_lengths.append(seq_len)
            all_tokens.append(s.text)
            all_targets.append(s.target)
            all_image_masks.append(s.image_input_mask)
            all_video_masks.append(s.video_input_mask)

            n_img = int(s.image_input_mask.sum().item())
            n_vid = int(s.video_input_mask.sum().item())
            n_txt = seq_len - n_img - n_vid
            total_image_tokens += n_img
            total_video_tokens += n_vid
            total_text_tokens += n_txt

            if len(s.imgs) > 0:
                if isinstance(s.imgs, torch.Tensor):
                    all_imgs.append(s.imgs.unsqueeze(0) if s.imgs.dim() == 3 else s.imgs)
                else:
                    all_imgs.append(s.imgs)
            if len(s.image_thw_grids) > 0:
                thw = s.image_thw_grids
                if isinstance(thw, np.ndarray):
                    for row in thw:
                        all_image_thw_grids.append(row)
                        total_vit_patches += int(np.prod(row))
                elif isinstance(thw, torch.Tensor):
                    for row in thw:
                        all_image_thw_grids.append(row.numpy())
                        total_vit_patches += int(row.prod().item())
                else:
                    all_image_thw_grids.extend(thw)

            if len(s.videos) > 0:
                if isinstance(s.videos, torch.Tensor):
                    all_videos.append(s.videos.unsqueeze(0) if s.videos.dim() == 3 else s.videos)
                else:
                    all_videos.append(s.videos)
            if len(s.video_thw_grids) > 0:
                thw = s.video_thw_grids
                if isinstance(thw, np.ndarray):
                    for row in thw:
                        all_video_thw_grids.append(row)
                        total_vit_patches += int(np.prod(row))
                elif isinstance(thw, torch.Tensor):
                    for row in thw:
                        all_video_thw_grids.append(row.numpy())
                        total_vit_patches += int(row.prod().item())
                else:
                    all_video_thw_grids.extend(thw)

        packed_tokens = torch.cat(all_tokens, dim=0)
        packed_targets = torch.cat(all_targets, dim=0)
        packed_image_mask = torch.cat(all_image_masks, dim=0)
        packed_video_mask = torch.cat(all_video_masks, dim=0)

        cu_lengths_list = [0]
        for l in sub_lengths:
            cu_lengths_list.append(cu_lengths_list[-1] + l)
        cu_lengths = torch.tensor(cu_lengths_list, dtype=torch.int32)

        total_packed = int(cu_lengths[-1].item())
        max_sub_len = max(sub_lengths)

        if _thd_diag_enabled():
            logger.info(
                f"[PackingStats] packed {len(samples)} samples → {total_packed} tokens "
                f"(seq_len={self.seq_len}, utilization={total_packed / self.seq_len * 100:.1f}%), "
                f"sub_lengths={sub_lengths}, "
                f"image_tokens={total_image_tokens}, video_tokens={total_video_tokens}, "
                f"text_tokens={total_text_tokens}, vit_patches={total_vit_patches}"
            )

        return QwenVLTaskSamplePacked(
            __key__="+".join(s.__key__ for s in samples),
            __subflavors__=samples[0].__subflavors__,
            tokens=packed_tokens,
            target=packed_targets,
            imgs=all_imgs,
            videos=all_videos,
            image_thw_grids=all_image_thw_grids,
            video_thw_grids=all_video_thw_grids,
            image_input_mask=packed_image_mask,
            video_input_mask=packed_video_mask,
            cu_lengths=cu_lengths,
            max_length=max_sub_len,
            num_sub_samples=len(samples),
            sub_sample_lengths=sub_lengths,
            num_image_tokens=total_image_tokens,
            num_video_tokens=total_video_tokens,
            num_text_tokens=total_text_tokens,
            num_vit_patches=total_vit_patches,
        )

    @staticmethod
    def _normalize_visual_block(block: torch.Tensor) -> torch.Tensor:
        """Normalize visual block shape before batch concat.

        Expected downstream format is either:
        - [N, C, H, W] (patch tiles), or
        - [N, D] (flattened patch features).
        """
        if not isinstance(block, torch.Tensor):
            raise TypeError(f"Visual block must be torch.Tensor, got {type(block)}")

        # Common case for processor output: [1, N, D] -> [N, D]
        if block.dim() == 3 and block.shape[0] == 1:
            return block.squeeze(0)

        # Single tile CHW -> add batch dimension.
        if block.dim() == 3 and block.shape[0] in (1, 3):
            return block.unsqueeze(0)

        # Keep already batched formats as-is.
        if block.dim() in (2, 4):
            return block

        return block

    @staticmethod
    def _concat_visual_blocks(blocks: List[torch.Tensor], name: str) -> Optional[torch.Tensor]:
        """Concatenate visual blocks along first dimension with shape diagnostics."""
        if not blocks:
            return None

        normalized = [QwenVLTaskEncoder._normalize_visual_block(b) for b in blocks]
        try:
            return torch.cat(normalized, dim=0)
        except RuntimeError as exc:
            shapes = [tuple(t.shape) for t in normalized]
            logger.error(f"Failed to concat {name} blocks, shapes={shapes}, error={exc}")
            raise

    def _collect_visual_data(self, samples):
        """Gather pixel_values and grid_thw from a list of samples (packed or unpacked)."""
        imgs, image_thw_grids = [], []
        videos, video_thw_grids = [], []

        for s in samples:
            if isinstance(s, QwenVLTaskSamplePacked):
                for img_block in s.imgs:
                    if isinstance(img_block, torch.Tensor):
                        imgs.append(self._normalize_visual_block(img_block))
                for thw in s.image_thw_grids:
                    image_thw_grids.append(np.asarray(thw))
                for vid_block in s.videos:
                    if isinstance(vid_block, torch.Tensor):
                        videos.append(self._normalize_visual_block(vid_block))
                for thw in s.video_thw_grids:
                    video_thw_grids.append(np.asarray(thw))
            else:
                if len(s.imgs) > 0:
                    s_imgs = self._normalize_visual_block(s.imgs) if isinstance(s.imgs, torch.Tensor) else s.imgs
                    if isinstance(s_imgs, torch.Tensor):
                        imgs.append(s_imgs)
                if len(s.image_thw_grids) > 0:
                    if isinstance(s.image_thw_grids, np.ndarray):
                        for row in s.image_thw_grids:
                            image_thw_grids.append(row)
                    else:
                        image_thw_grids.extend(s.image_thw_grids)
                if len(s.videos) > 0:
                    s_vids = self._normalize_visual_block(s.videos) if isinstance(s.videos, torch.Tensor) else s.videos
                    if isinstance(s_vids, torch.Tensor):
                        videos.append(s_vids)
                if len(s.video_thw_grids) > 0:
                    if isinstance(s.video_thw_grids, np.ndarray):
                        for row in s.video_thw_grids:
                            video_thw_grids.append(row)
                    else:
                        video_thw_grids.extend(s.video_thw_grids)

        return imgs, image_thw_grids, videos, video_thw_grids

    def batch(self, samples: List[Union[QwenVLTaskSample, QwenVLTaskSamplePacked]]) -> QwenVLTaskBatch:
        """
        Put encoded sample into Batch, do padding, add labels and visual input masks.

        Handles both unpacked (QwenVLTaskSample) and packed (QwenVLTaskSamplePacked) inputs.
        For packed inputs, pads to self.seq_len and passes cu_lengths through.
        """
        is_packed = isinstance(samples[0], QwenVLTaskSamplePacked)

        imgs, image_thw_grids, videos, video_thw_grids = self._collect_visual_data(samples)

        pad_token_id = self.hf_tokenizer.pad_token_id

        if is_packed:
            batch_size = len(samples)
            fixed_len = self.seq_len
            low_content_threshold = _thd_diag_low_content_threshold()
            max_low_content_logs = _thd_diag_max_low_content_logs()
            diag_enabled = _thd_diag_enabled()

            text_mat = np.full((batch_size, fixed_len), pad_token_id, dtype=np.int64)
            target_mat = np.full((batch_size, fixed_len), pad_token_id, dtype=np.int64)
            image_input_masks = np.zeros((batch_size, fixed_len), dtype=bool)
            video_input_masks = np.zeros((batch_size, fixed_len), dtype=bool)

            cu_lengths_list = []
            max_lengths_list = []

            for i, s in enumerate(samples):
                content_len = len(s.tokens)
                actual_len = min(content_len, fixed_len)
                text_mat[i, :actual_len] = np.array(s.tokens)[:actual_len]
                target_mat[i, :actual_len] = np.array(s.target)[:actual_len]
                image_input_masks[i, :actual_len] = np.array(s.image_input_mask)[:actual_len]
                video_input_masks[i, :actual_len] = np.array(s.video_input_mask)[:actual_len]

                cu = s.cu_lengths.clone()
                cu[cu > fixed_len] = fixed_len
                cu_lengths_list.append(cu)
                max_lengths_list.append(s.max_length)

                pad_len = fixed_len - actual_len
                if diag_enabled:
                    logger.info(
                        f"[BatchStats] sample {i}: {s.num_sub_samples} sub-seqs packed, "
                        f"content={actual_len}, pad={pad_len}, "
                        f"image_toks={s.num_image_tokens}, video_toks={s.num_video_tokens}, "
                        f"text_toks={s.num_text_tokens}, vit_patches={s.num_vit_patches}, "
                        f"utilization={actual_len / fixed_len * 100:.1f}%"
                    )
                    global _LOW_CONTENT_LOG_COUNT
                    if (
                        low_content_threshold > 0
                        and actual_len < low_content_threshold
                        and _LOW_CONTENT_LOG_COUNT < max_low_content_logs
                    ):
                        logger.warning(
                            "[LowContentPackedSample] key=%s content=%d pad=%d seq_len=%d utilization=%.1f%% "
                            "sub_samples=%d sub_lengths=%s image_toks=%d text_toks=%d",
                            s.__key__,
                            actual_len,
                            pad_len,
                            fixed_len,
                            actual_len / fixed_len * 100.0,
                            s.num_sub_samples,
                            s.sub_sample_lengths,
                            s.num_image_tokens,
                            s.num_text_tokens,
                        )
                        _LOW_CONTENT_LOG_COUNT += 1

            tokens = torch.from_numpy(text_mat)
            tokens[tokens == pad_token_id] = 0
            labels = torch.from_numpy(target_mat)
            labels[labels == pad_token_id] = IGNORE_INDEX

            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                data=tokens,
                eod_token=self.hf_tokenizer.eos_token_id,
                eod_mask_loss=False,
                reset_attention_mask=False,
                reset_position_ids=False,
                compute_attention_mask=False,
            )
            loss_mask[labels < 0] = 0.0

            max_cu_len = max(len(cu) for cu in cu_lengths_list)
            padded_cu = torch.zeros((batch_size, max_cu_len), dtype=torch.int32)
            for i, cu in enumerate(cu_lengths_list):
                padded_cu[i, : len(cu)] = cu

            cu_seqlens_argmin_list = [len(cu) for cu in cu_lengths_list]
            cu_seqlens_argmin = torch.tensor(cu_seqlens_argmin_list, dtype=torch.int32)
            max_lengths = torch.tensor(max_lengths_list, dtype=torch.int32)

            sample_keys = [s.__key__ for s in samples]
            batch_obj = QwenVLTaskBatch(
                __key__="+".join(sample_keys),
                __restore_key__=(),
                __keys__=sample_keys,
                __subflavors__=[s.__subflavors__ for s in samples],
                pixel_values=self._concat_visual_blocks(imgs, "image"),
                pixel_values_videos=self._concat_visual_blocks(videos, "video"),
                image_grid_thw=torch.from_numpy(np.array(image_thw_grids)) if len(image_thw_grids) > 0 else None,
                video_grid_thw=torch.from_numpy(np.array(video_thw_grids)) if len(video_thw_grids) > 0 else None,
                image_input_mask=torch.from_numpy(image_input_masks),
                video_input_mask=torch.from_numpy(video_input_masks),
                input_ids=tokens,
                attention_mask=None,
                position_ids=position_ids,
                labels=labels,
                loss_mask=loss_mask,
                cu_lengths=padded_cu,
                max_lengths=max_lengths,
                cu_seqlens_argmin=cu_seqlens_argmin,
            )
            return batch_obj

        # --- Original unpacked path ---
        max_seq_len = max(len(s.text) for s in samples)
        if max_seq_len > self.seq_len:
            logging.warning("max sequence length larger than passed parameter")

        text_mat = np.full((len(samples), max_seq_len), pad_token_id, dtype=np.int64)
        target_mat = np.full((len(samples), max_seq_len), pad_token_id, dtype=np.int64)

        image_input_masks = np.zeros_like(text_mat, dtype=bool)
        video_input_masks = np.zeros_like(text_mat, dtype=bool)
        for i, s in enumerate(samples):
            text_len = min(max_seq_len, len(s.text))
            target_len = min(max_seq_len, len(s.target))

            text_mat[i, :text_len] = np.array(s.text)[:text_len]
            if s.image_input_mask is not None:
                image_input_masks[i, :text_len] = np.array(s.image_input_mask)[:text_len]
            if s.video_input_mask is not None:
                video_input_masks[i, :text_len] = np.array(s.video_input_mask)[:text_len]
            target_mat[i, :target_len] = np.array(s.target)[:target_len]

        tokens = torch.from_numpy(text_mat)
        tokens[tokens == pad_token_id] = 0

        labels = torch.from_numpy(target_mat)
        labels[labels == pad_token_id] = IGNORE_INDEX

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=self.hf_tokenizer.eos_token_id,
            eod_mask_loss=False,
            reset_attention_mask=False,
            reset_position_ids=False,
        )

        loss_mask[labels < 0] = 0.0

        sample_keys = [s.__key__ for s in samples]
        batch_obj = QwenVLTaskBatch(
            __key__="+".join(sample_keys),
            __restore_key__=(),
            __keys__=sample_keys,
            __subflavors__=[s.__subflavors__ for s in samples],
            pixel_values=self._concat_visual_blocks(imgs, "image"),
            pixel_values_videos=self._concat_visual_blocks(videos, "video"),
            image_grid_thw=torch.from_numpy(np.array(image_thw_grids)) if len(image_thw_grids) > 0 else None,
            video_grid_thw=torch.from_numpy(np.array(video_thw_grids)) if len(video_thw_grids) > 0 else None,
            image_input_mask=torch.from_numpy(image_input_masks),
            video_input_mask=torch.from_numpy(video_input_masks),
            input_ids=tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            loss_mask=loss_mask,
        )
        return batch_obj

    def encode_batch(self, batch: QwenVLTaskBatch) -> dict:
        """Encode batch in dict"""

        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]

        raw["visual_inputs"] = Qwen2_5_VLVisualInputs(
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.image_grid_thw,
        )

        if batch.cu_lengths is not None:
            raw["cu_seqlens"] = batch.cu_lengths
            raw["max_seqlen"] = batch.max_lengths
            raw["cu_seqlens_argmin"] = batch.cu_seqlens_argmin

        return raw
