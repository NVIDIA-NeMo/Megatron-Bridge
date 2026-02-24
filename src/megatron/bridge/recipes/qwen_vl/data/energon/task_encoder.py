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

import dataclasses
import json
import logging
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from megatron.energon import Batch, DefaultTaskEncoder
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory
from webdataset.autodecode import Decoder, imagehandler

from megatron.bridge.training.utils.visual_inputs import Qwen2_5_VLVisualInputs


# Local replacements for former nemo dependencies
# Constants for internal multimodal token placeholders and label ignore
IGNORE_INDEX = -100


def get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    eod_mask_loss: bool,
    reset_attention_mask: bool,
    reset_position_ids: bool,
    compute_attention_mask: bool = True,
):
    """Build masks and position ids for a left-to-right model.

    Returns:
        attention_mask: [att_mask_batch, 1, s, s] boolean mask (True means masked)
        loss_mask: [b, s] float mask (1.0 to keep loss, 0.0 to drop)
        position_ids: [b, s] positions
    """
    micro_batch_size, seq_length = data.size()

    att_mask_batch = micro_batch_size if reset_attention_mask else 1
    attention_mask = None
    if compute_attention_mask:
        attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length
        )

    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        for b in range(micro_batch_size):
            eod_index = position_ids[b, data[b] == eod_token]
            if reset_position_ids:
                eod_index = eod_index.clone()
            prev_index = 0
            for j in range(eod_index.size(0)):
                i = eod_index[j]
                if reset_attention_mask and attention_mask is not None:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    if compute_attention_mask and attention_mask is not None:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def find_pattern_indices(sequence: np.ndarray, pattern, start: int = 0):
    """Find the [start, end) indices of the first occurrence of pattern in sequence from start."""
    if not isinstance(sequence, np.ndarray):
        sequence = np.array(sequence)
    pattern = np.array(pattern, dtype=sequence.dtype)
    n, m = sequence.shape[0], pattern.shape[0]
    if m == 0 or start >= n:
        return -1, -1
    end_limit = n - m + 1
    for i in range(start, max(end_limit, start)):
        if np.array_equal(sequence[i : i + m], pattern):
            return i, i + m
    return -1, -1


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
        image_inputs = processor(images=images, videos=None, return_tensors="pt", **kwargs)
        image_grid_thw = image_inputs.get("image_grid_thw", None)
    else:
        image_inputs = {}
        image_grid_thw = None

    if videos is not None:
        videos_inputs = processor(images=None, videos=videos, return_tensors="pt")
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
class ChatMLSample(Sample):
    """multi-turn complex samples with images and videos"""

    conversation: str  # JSON string of GPT-format conversations
    imgs: Optional[List[torch.Tensor]] = None
    videos: Optional[List[List[torch.Tensor]]] = None


class videohandler:
    """Create an video handler."""

    def __init__(self, imagespec):
        self.extensions = ["jpgs", "mp4s"]
        self.extensions_mapping = {"jpgs": "jpg", "mp4s": "jpg"}
        self.image_handler = imagehandler(imagespec)

    def __call__(self, key, data):
        """Perform nested image decoding."""
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in self.extensions:
            return None
        data = pickle.loads(data)
        key = self.extensions_mapping[extension]
        if extension.lower() == "jpgs":
            data = [self.image_handler(key, d) for d in data]
        else:
            data = [[self.image_handler(key, d) for d in video] for video in data]
        return data


class ChatMLWebdataset(DefaultDecoderWebdatasetFactory[ChatMLSample]):
    """Webdataset factory for multi-turn ChatML samples with multimodal support.

    Extends DefaultDecoderWebdatasetFactory to decode webdataset shards into
    ChatMLSample instances, using custom handlers for image and video fields.
    """

    __sample_type__ = ChatMLSample

    def __init__(self, path: EPath, *, auto_decode: bool = True, **kwargs):
        super().__init__(path, auto_decode=auto_decode, **kwargs)
        if auto_decode:
            self._decoder = Decoder(
                [
                    imagehandler(self.image_decode),
                    videohandler(self.image_decode),
                ]
            )


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


@dataclass
class QwenVLTaskBatch(Batch):
    """Encoded Batch Format For QwenVL"""

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
        # NOTE: flatten all images
        #     Input of process_vision:
        processed_vision = process_vision(
            self.image_processor,
            sample.imgs,
            sample.videos,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        image_thw_grids = processed_vision["image_grid_thw"]
        video_thw_grids = processed_vision["video_grid_thw"]
        flattened_imgs = processed_vision["image_inputs"]
        flattened_videos = processed_vision["video_inputs"]

        conversation = (
            json.loads(sample.conversation) if isinstance(sample.conversation, (str, bytes)) else sample.conversation
        )

        _from_system_ = "from" in conversation[0]
        role_key = "from" if "from" in conversation[0] else "role"
        content_key = "value" if "from" in conversation[0] else "content"

        # NOTE: assume the conversation format is: [System]? (User Assistant)+
        converted_conversation = []
        if len(conversation) % 2 != 0:
            converted_conversation.append({"role": "system", "content": conversation[0][content_key]})
            conversation = conversation[1:]

        if _from_system_:  # ['conversations':[{'from':'human', 'value':[]}, {'from':'gpt', 'value':[]}]
            EXPECTED_ROLE = ["human", "gpt"]
            for turn_idx, turn in enumerate(conversation):
                role = turn[role_key]
                if role != EXPECTED_ROLE[turn_idx % len(EXPECTED_ROLE)]:
                    logging.warning(
                        f"Expect conversation organized in order: [sys] human gpt human gpt...,"
                        f"but got role '{role}' in turn {turn_idx}"
                    )
                content = turn[content_key]

                if role == "human":
                    role = "user"
                    content = convert_to_qwenvl_content(content)
                    # Reorder media first to align with PreloadedVLMConversationProvider
                    media_content = [c for c in content if c.get("type") in ("image", "video")]
                    text_content = [c for c in content if c.get("type") == "text"]
                    content = media_content + text_content
                elif role == "gpt":
                    role = "assistant"

                converted_conversation.append({"role": role, "content": content})
        else:  # ['messages':[{'role':'user', 'content':[]}, {'role':'assistant', 'content':[]}]
            EXPECTED_ROLE = ["user", "assistant"]
            for turn_idx, turn in enumerate(conversation):
                role = turn[role_key]
                if role != EXPECTED_ROLE[turn_idx % len(EXPECTED_ROLE)]:
                    logging.warning(
                        f"Expect conversation organized in order: [sys] user assistant user assistant...,"
                        f" but got role '{role}' in turn {turn_idx}"
                    )
                content = turn[content_key]

                if role == "user":
                    content = convert_to_qwenvl_content(content)
                    # Reorder media first to align with PreloadedVLMConversationProvider
                    media_content = [c for c in content if c.get("type") in ("image", "video")]
                    text_content = [c for c in content if c.get("type") == "text"]
                    content = media_content + text_content

                converted_conversation.append({"role": role, "content": content})
        conversation = converted_conversation

        # NOTE: we need to mask all system/user input tokens and assistant generation prefix tokens
        input_ids = self.hf_tokenizer.apply_chat_template(conversation, tokenize=True, return_tensors="np")[0]
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
        )

    def batch(self, samples: List[QwenVLTaskSample]) -> QwenVLTaskBatch:
        """
        Put encoded sample into Batch, do padding, add labels and visual input masks

        Args:
            samples: List of encoded samples

        Returns:
            Batch with necessary fields
        """
        imgs, image_thw_grids = [], []
        for s in samples:
            if len(s.imgs) > 0:
                s_imgs = [img for img in s.imgs.unsqueeze(0)]
                cat_imgs = torch.cat([img for img in s_imgs])
                imgs.append(cat_imgs)
            if len(s.image_thw_grids) > 0:
                s_image_thw_grids = [thw_grids for thw_grids in s.image_thw_grids]
                image_thw_grids.extend(s_image_thw_grids)
        videos, video_thw_grids = [], []
        for s in samples:
            if len(s.videos) > 0:
                s_videos = [video for video in s.videos.unsqueeze(0)]
                cat_videos = torch.cat([video for video in s_videos])
                videos.append(cat_videos)
            if len(s.video_thw_grids) > 0:
                s_video_thw_grids = [thw_grids for thw_grids in s.video_thw_grids]
                video_thw_grids.extend(s_video_thw_grids)
                # assert s_video_thw_grids.prod(dim=-1).sum() == s_videos.shape[0]

        # use the max sample lengths in the batch.
        max_seq_len = max(len(s.text) for s in samples)
        if max_seq_len > self.seq_len:
            logging.warning("max sequence length larger than passed parameter")

        text_mat = np.full((len(samples), max_seq_len), self.hf_tokenizer.pad_token_id, dtype=np.int64)
        target_mat = np.full((len(samples), max_seq_len), self.hf_tokenizer.pad_token_id, dtype=np.int64)

        image_input_masks = np.zeros_like(text_mat, dtype=bool)
        video_input_masks = np.zeros_like(text_mat, dtype=bool)
        for i, s in enumerate(samples):
            # If the sample/target length exceeds the target sequence length, then truncate.
            text_len = min(max_seq_len, len(s.text))
            target_len = min(max_seq_len, len(s.target))

            text_mat[i, :text_len] = np.array(s.text)[:text_len]
            # NOTE: we should assert user input sequence will not be truncated
            if s.image_input_mask is not None:
                image_input_masks[i, :text_len] = np.array(s.image_input_mask)[:text_len]
            if s.video_input_mask is not None:
                video_input_masks[i, :text_len] = np.array(s.video_input_mask)[:text_len]
            target_mat[i, :target_len] = np.array(s.target)[:target_len]

        tokens = torch.from_numpy(text_mat)
        tokens[tokens == self.hf_tokenizer.pad_token_id] = 0

        labels = torch.from_numpy(target_mat)
        labels[labels == self.hf_tokenizer.pad_token_id] = IGNORE_INDEX

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=self.hf_tokenizer.eos_token_id,
            eod_mask_loss=False,
            reset_attention_mask=False,
            reset_position_ids=False,
        )

        loss_mask[labels < 0] = 0.0

        batch = QwenVLTaskBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            pixel_values=torch.vstack(imgs) if len(imgs) > 0 else None,
            pixel_values_videos=torch.vstack(videos) if len(videos) > 0 else None,
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
        return batch

    def encode_batch(self, batch: QwenVLTaskBatch) -> dict:
        """Encode batch in dict"""

        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]

        raw["visual_inputs"] = Qwen2_5_VLVisualInputs(
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.image_grid_thw,
        )

        return raw
