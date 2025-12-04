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

"""
Energon task encoder for VLM style pretraining.

This encoder consumes raw multimodal records (images + chat JSON) and produces
model-ready tensors compatible with Megatron-Bridge training loops.
It intentionally does not import from NeMo.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from megatron.energon import Batch, DefaultTaskEncoder
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys
from transformers import AutoProcessor

from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.training.utils.visual_inputs import Qwen2_5_VLVisualInputs


def _process_vision(image_processor, images: Optional[List[Image.Image]], videos: Optional[Any] = None):
    """
    Minimal vision processing using HFimage processor.

    Returns a dict with keys:
      - "pixel_values": Tensor [N, C, H, W] or None
      - "image_grid_thw": Tensor [N, 3] or None
    """
    if hasattr(image_processor, "__call__"):
        inputs = image_processor(images=images, videos=None, return_tensors="pt") if images else {}
    else:
        inputs = {}

    pixel_values = inputs.get("pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw", None)
    return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}


def _find_pattern_indices(template: Sequence[int], pattern: Sequence[int], search_start_index: int = 0) -> Tuple[int, int]:
    """Locate pattern in template starting at search_start_index. Returns (start, end) or (-1, -1)."""
    t_len, p_len = len(template), len(pattern)
    for i in range(search_start_index, t_len - p_len + 1):
        if all(template[i + j] == pattern[j] for j in range(p_len)):
            return i, i + p_len
    return -1, -1


def _convert_to_vl_content(user_input: str, image_pattern: str = "<image>", video_pattern: str = "<video>"):
    """Split user input into chunks with media placeholders formatted for HF processors."""
    pattern = r"({image}|{video})".format(image=re.escape(image_pattern), video=re.escape(video_pattern))
    contents: List[Dict[str, Any]] = []
    cur = 0
    mm_idx: Dict[str, int] = {"image": 0, "video": 0}
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        if start > cur:
            contents.append({"type": "text", "text": user_input[cur:start].strip(" ")})
        token = matched.group(1)[1:-1]  # 'image' or 'video'
        contents.append({"type": token, token: str(mm_idx[token])})
        mm_idx[token] += 1
        cur = end
    if cur < len(user_input):
        contents.append({"type": "text", "text": user_input[cur:].strip(" ")})
    return contents


@dataclass
class ChatMLSample(Sample):
    """Intermediate sample with decoded images/videos and conversation JSON."""

    imgs: Optional[List[Image.Image]]
    videos: Optional[Any]
    conversation: str | list[dict]


def _cook_chatml_sample(sample: dict) -> ChatMLSample:
    """Convert raw Energon sample into ChatMLSample."""
    imgs = sample.get("jpgs", None)
    if imgs:
        imgs = pickle.loads(imgs)
        if isinstance(imgs, list) and len(imgs) > 0:
            imgs = [Image.fromarray(d) for d in imgs]
        else:
            imgs = None
    videos = sample.get("videos", None)
    if videos:
        videos = pickle.loads(videos)
        if not (isinstance(videos, list) and len(videos) > 0):
            videos = None

    conversation = sample.get("json")
    if isinstance(conversation, (bytes, str)):
        conversation = conversation.decode("utf-8") if isinstance(conversation, bytes) else conversation
    return ChatMLSample(
        **basic_sample_keys(sample),
        imgs=imgs,
        videos=videos,
        conversation=conversation,
    )


@dataclass
class _EncodedSample:
    """Encoded single-sample container."""

    text: torch.Tensor
    target: torch.Tensor
    image_input_mask: Optional[torch.Tensor]
    pixel_values: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]


@dataclass
class _EncodedBatch(Batch):
    """Minimal batch structure expected by Megatron-Bridge training."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    visual_inputs: Optional[Qwen2_5_VLVisualInputs]


class VLTaskEncoder(DefaultTaskEncoder[ChatMLSample, _EncodedSample, _EncodedBatch, dict]):
    """Task encoder that tokenizes chat+vision records"""

    cookers = [Cooker(_cook_chatml_sample)]

    def __init__(self, processor: AutoProcessor, max_padding_length: int = 4096) -> None:
        super().__init__()
        # AutoProcessor with tokenizer and image processor
        self.processor = processor
        self.hf_tokenizer = getattr(processor, "tokenizer", processor)
        self.image_processor = getattr(processor, "image_processor", processor)
        self.seq_len = int(max_padding_length)

    def _normalize_conversation(self, conversation: str | list[dict]) -> list[dict]:
        if isinstance(conversation, (str, bytes)):
            conversation = json.loads(conversation)
        assert isinstance(conversation, list), "conversation must be list[dict] or JSON string"

        role_key = "from" if "from" in conversation[0] else "role"
        content_key = "value" if "from" in conversation[0] else "content"
        converted: List[dict] = []

        # Ensure system prompt exists
        if len(conversation) % 2 == 0:
            converted.append({"role": "system", "content": "You are a helpful assistant."})
        else:
            converted.append({"role": "system", "content": conversation[0][content_key]})
            conversation = conversation[1:]

        expected_roles = ["human", "gpt"] if role_key == "from" else ["user", "assistant"]

        for idx, turn in enumerate(conversation):
            role = turn[role_key]
            content = turn[content_key]
            # Map role names
            if role_key == "from":
                role = "user" if role == "human" else "assistant"
            # Convert user content with media markers into VL content list
            if role == "user" and isinstance(content, str):
                content = _convert_to_vl_content(content)
            converted.append({"role": role, "content": content})

        return converted

    def encode_sample(self, sample: ChatMLSample) -> _EncodedSample:
        # Process vision inputs
        vision = _process_vision(self.image_processor, sample.imgs, None)
        image_thw_grids = vision.get("image_grid_thw")
        pixel_values = vision.get("pixel_values")

        # Normalize conversation and tokenize
        conversation = self._normalize_conversation(sample.conversation)
        input_ids: List[int] = self.hf_tokenizer.apply_chat_template(conversation, tokenize=True, return_tensors=None)

        # Build supervision mask over assistant answers before visual expansion
        target_mask_ids = [self.hf_tokenizer.pad_token_id for _ in range(len(input_ids))]
        search_start = 0
        for turn in conversation[1:]:
            if turn["role"] == "assistant":
                answer = turn["content"]
                # append end markers similar to HF chat templates
                answer_text = answer if isinstance(answer, str) else self.hf_tokenizer.decode(
                    self.hf_tokenizer(answer, add_special_tokens=False)["input_ids"]
                )
                answer_tokens = self.hf_tokenizer.encode(answer_text + "\n", add_special_tokens=False)
                a_start, a_end = _find_pattern_indices(input_ids, answer_tokens, search_start)
                if a_start > 0:
                    # copy token ids where we expect labels to be learned
                    target_mask_ids[a_start:a_end] = input_ids[a_start:a_end]
                    search_start = a_end

        # Expand <|image_pad|> or similar placeholders by patch grid size
        image_pad_id = self.hf_tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad_id = self.hf_tokenizer.convert_tokens_to_ids("<|video_pad|>") if hasattr(self.hf_tokenizer, "convert_tokens_to_ids") else None

        if isinstance(image_thw_grids, torch.Tensor):
            image_grid_thw_np = image_thw_grids.cpu().numpy()
        elif image_thw_grids is not None:
            image_grid_thw_np = np.array(image_thw_grids)
        else:
            image_grid_thw_np = None

        merge_size = getattr(self.image_processor, "merge_size", 2)
        merge_len = int(merge_size) * int(merge_size)

        # Gather positions of special pads in the unexpanded sequence
        img_indices = [i for i, t in enumerate(input_ids) if t == image_pad_id]
        vid_indices = [i for i, t in enumerate(input_ids) if video_pad_id is not None and t == video_pad_id]

        # Compute expanded target length
        target_length = len(input_ids)
        if image_grid_thw_np is not None:
            target_length = target_length - len(img_indices) + int(np.prod(image_grid_thw_np, axis=-1).sum() // merge_len)

        final_input_ids = [0] * target_length
        final_target_ids = [self.hf_tokenizer.pad_token_id] * target_length
        img_ptr = 0
        cur_x = 0
        cur_y = 0
        special_positions = sorted([(i, "image") for i in img_indices] + [(i, "video") for i in vid_indices])
        for idx, kind in special_positions:
            size = 0
            if kind == "image" and image_grid_thw_np is not None:
                size = int(np.prod(image_grid_thw_np[img_ptr]) // merge_len)
                img_ptr += 1
            # Copy pre-chunk
            pre_len = idx - cur_x
            if pre_len > 0:
                final_input_ids[cur_y : cur_y + pre_len] = input_ids[cur_x:idx]
                final_target_ids[cur_y : cur_y + pre_len] = target_mask_ids[cur_x:idx]
                cur_y += pre_len
            # Fill expanded visual pad span with the special token id and pad targets
            if size > 0:
                final_input_ids[cur_y : cur_y + size] = [image_pad_id] * size
                final_target_ids[cur_y : cur_y + size] = [self.hf_tokenizer.pad_token_id] * size
                cur_y += size
            cur_x = idx + 1
        # Trailing tail
        if cur_x < len(input_ids):
            final_input_ids[cur_y:] = input_ids[cur_x:]
            final_target_ids[cur_y:] = target_mask_ids[cur_x:]

        # Left-shift labels for next-token prediction
        tokens_t = torch.tensor(final_input_ids, dtype=torch.long)
        labels_full = torch.tensor(final_target_ids, dtype=torch.long)
        labels = torch.roll(labels_full, shifts=-1)
        labels[-1] = self.hf_tokenizer.pad_token_id

        # Convert labels' pad positions to IGNORE_INDEX
        labels[labels == self.hf_tokenizer.pad_token_id] = IGNORE_INDEX

        # Track visual positions (optional)
        image_input_mask = torch.tensor([t == image_pad_id for t in final_input_ids], dtype=torch.bool)

        return _EncodedSample(
            text=tokens_t,
            target=labels,
            image_input_mask=image_input_mask if image_pad_id is not None else None,
            pixel_values=pixel_values.to(torch.bfloat16) if isinstance(pixel_values, torch.Tensor) else None,
            image_grid_thw=image_thw_grids if isinstance(image_thw_grids, torch.Tensor) else None,
        )

    def batch(self, samples: List[_EncodedSample]) -> _EncodedBatch:
        # Pad to max length across samples
        max_len = max(s.text.size(0) for s in samples)
        pad_id = int(getattr(self.hf_tokenizer, "pad_token_id", 0) or 0)
        input_ids = torch.full((len(samples), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(samples), max_len), IGNORE_INDEX, dtype=torch.long)
        loss_mask = torch.zeros((len(samples), max_len), dtype=torch.float)
        position_ids = torch.arange(max_len, dtype=torch.long).unsqueeze(0).expand(len(samples), -1).clone()

        for i, s in enumerate(samples):
            sl = s.text.size(0)
            input_ids[i, :sl] = s.text
            labels[i, :sl] = s.target
            loss_mask[i, :sl] = (s.target != IGNORE_INDEX).float()

        # Build visual inputs container; stack available tensors along batch dimension
        pixel_values_list = [s.pixel_values.unsqueeze(0) for s in samples if s.pixel_values is not None]
        image_grid_thw_list = [s.image_grid_thw.unsqueeze(0) for s in samples if s.image_grid_thw is not None]
        visual_inputs = None
        if pixel_values_list or image_grid_thw_list:
            pv = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None
            thw = torch.cat(image_grid_thw_list, dim=0) if image_grid_thw_list else None
            visual_inputs = Qwen2_5_VLVisualInputs(pixel_values=pv, image_grid_thw=thw)

        return _EncodedBatch(
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            visual_inputs=visual_inputs,
        )

    def encode_batch(self, batch: _EncodedBatch) -> dict:
        # Convert dataclass batch to plain dict expected by training loop
        return {
            "input_ids": batch.input_ids,
            "labels": batch.labels,
            "loss_mask": batch.loss_mask,
            "position_ids": batch.position_ids,
            "visual_inputs": batch.visual_inputs,
        }

