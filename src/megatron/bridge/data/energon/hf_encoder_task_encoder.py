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

"""Generic HF-encoder VLM task encoder for Energon dataloading.

Works with any HF processor that handles tokenization + vision preprocessing
in a single ``processor()`` call (e.g. Gemma3-VL, Ministral3, GLM-4.5V).
"""

import dataclasses
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from megatron.energon import Batch, DefaultTaskEncoder

from megatron.bridge.data.energon.task_encoder_utils import (
    IGNORE_INDEX,
    ChatMLSample,
    _images_to_pil,
    _videos_to_pil,
    cook_chatml_sample,
    find_pattern_indices,
    get_ltor_masks_and_position_ids,
)
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


@dataclass
class HFEncoderTaskSample:
    """Encoded sample for a generic HF-encoder VLM."""

    __key__: str
    __subflavors__: Dict
    input_ids: torch.Tensor  # [seq_len]
    labels: torch.Tensor  # [seq_len]
    loss_mask: torch.Tensor  # [seq_len]
    visual_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class HFEncoderTaskBatch(Batch):
    """Batched format for a generic HF-encoder VLM.

    Inherits ``__key__``, ``__restore_key__``, ``__subflavors__`` from
    :class:`Batch`.
    """

    input_ids: torch.Tensor  # [B, seq_len]
    labels: torch.Tensor  # [B, seq_len]
    loss_mask: torch.Tensor  # [B, seq_len]
    attention_mask: torch.Tensor  # [B, 1, seq_len, seq_len]
    position_ids: torch.Tensor  # [B, seq_len]
    visual_tensors: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)


class HFEncoderVLMTaskEncoder(DefaultTaskEncoder[ChatMLSample, HFEncoderTaskSample, HFEncoderTaskBatch, dict]):
    """Task encoder for HF-encoder VLMs that rely on ``processor()`` for tokenization + vision.

    Args:
        processor: HF ``AutoProcessor`` instance. Must support ``apply_chat_template``
            and ``__call__(text=..., images=..., ...)`` returning ``input_ids`` and
            visual tensor keys.
        seq_length: Maximum sequence length (tokens are truncated to this).
        visual_keys: Which keys from the processor output to capture as visual
            tensors (e.g. ``("pixel_values",)`` for Gemma3-VL / Ministral3,
            ``("pixel_values", "pixel_values_videos", "image_grid_thw",
            "video_grid_thw")`` for GLM-4.5V).
        min_pixels: Optional min pixel constraint forwarded to the processor.
        max_pixels: Optional max pixel constraint forwarded to the processor.
    """

    def __init__(
        self,
        processor,
        seq_length: int = 4096,
        visual_keys: Sequence[str] = ("pixel_values",),
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ):
        super().__init__()
        self.processor = processor
        self.seq_length = seq_length
        self.visual_keys: Tuple[str, ...] = tuple(visual_keys)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _tokenizer(self):
        """Return the underlying tokenizer from the processor."""
        return getattr(self.processor, "tokenizer", self.processor)

    @property
    def _pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def _eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # encode_sample
    # ------------------------------------------------------------------

    def encode_sample(self, sample: ChatMLSample) -> HFEncoderTaskSample:
        """Encode a single ChatML sample into model-ready tensors.

        1. Convert WDS tensor images/videos to PIL.
        2. Normalize conversation via ``cook_chatml_sample``.
        3. Use the HF processor's ``apply_chat_template`` to get the prompt text,
           then call ``processor(text=..., images=...)`` for joint tokenization +
           vision preprocessing.
        4. Build a loss mask that only supervises assistant turns.
        5. Truncate to ``seq_length``.
        """
        # 1. Images / videos -> PIL
        images_pil = _images_to_pil(sample.imgs) if sample.imgs is not None and len(sample.imgs) > 0 else None
        videos_pil = _videos_to_pil(sample.videos) if sample.videos is not None and len(sample.videos) > 0 else None

        # 2. Normalize conversation
        conversation = cook_chatml_sample(sample.conversation)

        # 2b. Convert <image> placeholders to structured multimodal content
        #     so that apply_chat_template inserts model-specific image tokens.
        has_images = images_pil is not None
        if has_images:
            for turn in conversation:
                text = turn["content"]
                if "<image>" in text:
                    parts = re.split(r"(<image>)", text)
                    content_parts: list = []
                    for part in parts:
                        if part == "<image>":
                            content_parts.append({"type": "image"})
                        elif part.strip():
                            content_parts.append({"type": "text", "text": part.strip()})
                    turn["content"] = content_parts

        # 3. Get the full prompt text from chat template (not tokenized)
        prompt_text = self._tokenizer.apply_chat_template(conversation, tokenize=False)

        # 4. Run processor for joint tokenization + vision preprocessing
        proc_kwargs = {"text": prompt_text, "return_tensors": "pt"}
        if images_pil is not None:
            proc_kwargs["images"] = images_pil
        if videos_pil is not None:
            proc_kwargs["videos"] = videos_pil
        if self.min_pixels is not None:
            proc_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            proc_kwargs["max_pixels"] = self.max_pixels

        proc_output = self.processor(**proc_kwargs)

        input_ids_t = proc_output["input_ids"]  # [1, seq]
        if input_ids_t.dim() == 2:
            input_ids_np = input_ids_t[0].numpy()
        else:
            input_ids_np = input_ids_t.numpy()

        # 5. Build loss mask: only supervise assistant content
        loss_mask_np = np.zeros(len(input_ids_np), dtype=np.float32)
        search_start = 0
        for turn in conversation:
            if turn["role"] == "assistant":
                answer = turn["content"]
                answer_tokens = self._tokenizer.encode(answer, add_special_tokens=False)
                ans_start, ans_end = find_pattern_indices(input_ids_np, answer_tokens, search_start)
                if ans_start >= 0:
                    loss_mask_np[ans_start:ans_end] = 1.0
                    search_start = ans_end

        # 6. Labels = left-shifted input_ids; positions without valid label get IGNORE_INDEX
        labels_np = np.full(len(input_ids_np), IGNORE_INDEX, dtype=np.int64)
        labels_np[:-1] = input_ids_np[1:]
        # Zero out labels where loss_mask is 0 (shift loss_mask accordingly)
        shifted_loss = np.zeros_like(loss_mask_np)
        shifted_loss[:-1] = loss_mask_np[1:]
        labels_np[shifted_loss == 0.0] = IGNORE_INDEX
        # Also shift loss_mask to align with labels
        loss_mask_np = shifted_loss

        # 7. Truncate
        max_len = self.seq_length
        input_ids_np = input_ids_np[:max_len]
        labels_np = labels_np[:max_len]
        loss_mask_np = loss_mask_np[:max_len]

        if len(input_ids_np) > max_len:
            logging.warning(f"Sequence length {len(input_ids_np)} exceeds seq_length {max_len}, truncated.")

        # 8. Collect visual tensors
        visual_tensors: Dict[str, torch.Tensor] = {}
        for key in self.visual_keys:
            val = proc_output.get(key)
            if val is not None:
                if isinstance(val, torch.Tensor):
                    visual_tensors[key] = val
                else:
                    visual_tensors[key] = torch.tensor(val)

        return HFEncoderTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            input_ids=torch.from_numpy(input_ids_np.copy()),
            labels=torch.from_numpy(labels_np.copy()),
            loss_mask=torch.from_numpy(loss_mask_np.copy()),
            visual_tensors=visual_tensors,
        )

    # ------------------------------------------------------------------
    # batch
    # ------------------------------------------------------------------

    def batch(self, samples: List[HFEncoderTaskSample]) -> HFEncoderTaskBatch:
        """Pad and collate a list of encoded samples into a batch."""
        max_seq_len = max(s.input_ids.size(0) for s in samples)
        if max_seq_len > self.seq_length:
            logging.warning(f"Max batch seq_len {max_seq_len} exceeds seq_length {self.seq_length}")

        pad_id = self._pad_token_id
        batch_size = len(samples)

        input_ids_mat = np.full((batch_size, max_seq_len), pad_id, dtype=np.int64)
        labels_mat = np.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=np.int64)
        loss_mask_mat = np.zeros((batch_size, max_seq_len), dtype=np.float32)

        for i, s in enumerate(samples):
            seq_len = min(max_seq_len, s.input_ids.size(0))
            input_ids_mat[i, :seq_len] = s.input_ids.numpy()[:seq_len]
            labels_mat[i, :seq_len] = s.labels.numpy()[:seq_len]
            loss_mask_mat[i, :seq_len] = s.loss_mask.numpy()[:seq_len]

        tokens = torch.from_numpy(input_ids_mat)
        # Replace pad tokens with 0 for model input (consistent with Qwen encoder)
        tokens[tokens == pad_id] = 0

        labels = torch.from_numpy(labels_mat)
        loss_mask_t = torch.from_numpy(loss_mask_mat)

        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=self._eos_token_id,
            eod_mask_loss=False,
            reset_attention_mask=False,
            reset_position_ids=False,
        )

        # Aggregate visual tensors across samples
        all_visual_keys = set()
        for s in samples:
            all_visual_keys.update(s.visual_tensors.keys())

        batched_visual: Dict[str, Optional[torch.Tensor]] = {}
        for key in all_visual_keys:
            tensors = [s.visual_tensors[key] for s in samples if key in s.visual_tensors]
            if tensors:
                batched_visual[key] = torch.cat(tensors, dim=0)
            else:
                batched_visual[key] = None

        return HFEncoderTaskBatch(
            __key__=[s.__key__ for s in samples],
            __restore_key__=(),
            __subflavors__=[s.__subflavors__ for s in samples],
            input_ids=tokens,
            labels=labels,
            loss_mask=loss_mask_t,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_tensors=batched_visual,
        )

    # ------------------------------------------------------------------
    # encode_batch
    # ------------------------------------------------------------------

    def encode_batch(self, batch: HFEncoderTaskBatch) -> dict:
        """Convert batch dataclass to dict, wrapping visual tensors in ``GenericVisualInputs``."""
        raw = dataclasses.asdict(batch)

        # Remove Batch base-class metadata not needed downstream
        for meta_key in ("__key__", "__restore_key__", "__subflavors__", "__sources__"):
            raw.pop(meta_key, None)

        # Replace the raw visual_tensors dict with a GenericVisualInputs container
        vt = batch.visual_tensors if batch.visual_tensors else {}
        raw["visual_inputs"] = GenericVisualInputs(**{k: v for k, v in vt.items() if v is not None})
        raw.pop("visual_tensors", None)

        return raw
