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
Generic mock conversation-style VLM dataset and provider.

This module produces synthetic image(s) and minimal conversations that are
compatible with HF `AutoProcessor.apply_chat_template` and the collate
functions defined in `collate.py`. It is processor-agnostic and can be used
with any multimodal model whose processor supports the standard conversation
schema and optional `images` argument.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy
import torch
from PIL import Image

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider

from .conversation_dataset import VLMConversationDataset


class MockVLMConversationDataset(torch.utils.data.Dataset):
    """Mock VLM dataset yielding conversation-style samples with optional images.

    Each sample contains minimally required tensors; batching and final shaping
    is handled by collate functions and the training step utilities.
    """

    def __init__(self, size: int, config: Any) -> None:
        if Image is None:
            raise ImportError("PIL is required for MockVLMConversationDataset. Please install pillow.")

        self.size = size
        self.config = config

        # Infer tokenizer from processor
        try:
            self._hf_tokenizer = getattr(config._processor, "tokenizer", None)
        except Exception:
            self._hf_tokenizer = None

        if self._hf_tokenizer is None:
            raise ValueError("config._processor must have a 'tokenizer' attribute")

        if self._hf_tokenizer.pad_token_id is None and getattr(self._hf_tokenizer, "eos_token_id", None) is not None:
            # Ensure pad token exists for stable padding/loss masking
            try:
                self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
            except Exception:
                pass

        self._rng = numpy.random.default_rng(seed=self.config.random_seed)

    def __len__(self) -> int:
        return self.size

    def _generate_random_image(self) -> Image.Image:
        w, h = self.config.image_size
        # Simple RGB image with uniform noise
        array = self._rng.integers(low=0, high=256, size=(h, w, 3), dtype=numpy.uint8)
        return Image.fromarray(array, mode="RGB")

    def _build_inputs(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Build conversation with optional images and a text prompt
        num_images = max(0, int(getattr(self.config, "num_images", 1)))
        content = [{"type": "image"} for _ in range(num_images)]
        content.append({"type": "text", "text": self.config.prompt})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "dummy assistant response"},
        ]

        text = self.config._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        images: Optional[list[Image.Image]] = None
        if num_images > 0:
            images = [self._generate_random_image() for _ in range(num_images)]

        processor_kwargs: Dict[str, Any] = {
            "text": [text],
            "padding": "max_length" if self.config.pad_to_max_length else True,
            "return_tensors": "pt",
        }
        if images is not None:
            processor_kwargs["images"] = images

        if self.config.pad_to_max_length:
            processor_kwargs["max_length"] = self.config.sequence_length

        inputs = self.config._processor(**processor_kwargs)

        input_ids: torch.Tensor = inputs.input_ids[0]  # [L]
        # Enforce exact sequence length by truncating or padding with random token ids
        target_len = int(self.config.sequence_length) + 1
        cur_len = input_ids.numel()
        if cur_len > target_len:
            input_ids = input_ids[:target_len]
        elif cur_len < target_len:
            vocab_size = getattr(self._hf_tokenizer, "vocab_size", None)
            if not vocab_size or vocab_size <= 1:
                vocab_size = len(self._hf_tokenizer.get_vocab())
            pad_len = target_len - cur_len
            random_tail = torch.randint(low=0, high=int(vocab_size), size=(pad_len,), dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, random_tail], dim=0)

        pixel_values_t: Optional[torch.Tensor] = None
        image_grid_thw_t: Optional[torch.Tensor] = None
        if images is not None:
            # Ensure per-sample shapes without a leading batch dim
            pixel_values_t = inputs.pixel_values
            if pixel_values_t.dim() == 5 and pixel_values_t.size(0) == 1:
                pixel_values_t = pixel_values_t.squeeze(0)
            image_grid_thw_t = inputs.image_grid_thw
            if image_grid_thw_t.dim() == 3 and image_grid_thw_t.size(0) == 1:
                image_grid_thw_t = image_grid_thw_t.squeeze(0)

        return input_ids, pixel_values_t, image_grid_thw_t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # noqa: ARG002 - randomness is OK
        input_ids, pixel_values, image_grid_thw = self._build_inputs()

        # Ensure at least length 2 to form tokens/labels by shifting
        if input_ids.numel() < 2:
            pad_id = self._hf_tokenizer.pad_token_id or 0
            input_ids = torch.nn.functional.pad(input_ids, (0, 2 - input_ids.numel()), value=pad_id)

        # Create tokens and labels with next-token prediction convention
        tokens = input_ids[:-1].contiguous()
        labels = input_ids[1:].contiguous()

        # Position IDs: [0, 1, ..., L-1]
        position_ids = torch.arange(tokens.numel(), dtype=torch.long, device=tokens.device)

        # Attention mask: 1D valid-token mask (all ones by default)
        attention_mask = torch.ones_like(tokens, dtype=torch.bool)

        # Loss mask: mask out padding positions in labels
        pad_token_id = self._hf_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self._hf_tokenizer, "eos_token_id", 0) or 0

        loss_mask = torch.ones_like(labels, dtype=torch.float)
        loss_mask[labels == pad_token_id] = 0.0

        # Safety for embedding lookup on padding
        tokens = tokens.clone()
        labels = labels.clone()
        tokens[tokens == pad_token_id] = 0
        labels[labels == pad_token_id] = 0

        sample: Dict[str, torch.Tensor] = {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        if pixel_values is not None:
            sample["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            sample["image_grid_thw"] = image_grid_thw

        return sample


@dataclass(kw_only=True)
class MockVLMConversationProvider(DatasetProvider):
    """DatasetProvider for generic mock VLM conversation datasets.

    Builds train/valid/test datasets using a HF AutoProcessor and the
    `MockVLMConversationDataset` implementation. Intended to work across
    different VLM models whose processors support the conversation schema.
    """

    # Required to match model.seq_length
    sequence_length: int

    # HF processor/model ID (e.g., Qwen/Qwen2.5-VL-3B-Instruct or other VLMs)
    hf_processor_path: str

    # Sample generation options
    prompt: str = "Describe this image."
    random_seed: int = 0
    image_size: Tuple[int, int] = (256, 256)
    pad_to_max_length: bool = True
    create_attention_mask: bool = True

    # Keep parity with GPTDatasetConfig usage in batching utilities
    skip_getting_attention_mask_from_dataset: bool = True

    # Number of images per sample
    num_images: int = 1

    # HF AutoProcessor instance will be set during build
    _processor: Optional[Any] = None

    def _make_base_examples(self) -> List[Dict[str, Any]]:
        # Single minimal conversation example; dataset will repeat to target length
        num_images = max(0, int(getattr(self, "num_images", 1)))
        w, h = self.image_size
        rng = numpy.random.default_rng(seed=self.random_seed)
        images = None
        if num_images > 0:
            # Embed in-memory PIL images directly in the conversation so that
            # qwen_vl_utils.process_vision_info can discover them.
            images = [
                Image.fromarray(rng.integers(low=0, high=256, size=(h, w, 3), dtype=numpy.uint8), mode="RGB")
                for _ in range(num_images)
            ]

        content = [{"type": "image", "image": img} for img in images] if images is not None else []
        content.append({"type": "text", "text": self.prompt})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "dummy assistant response"},
        ]
        return [{"conversation": messages}]

    def build_datasets(self, context: DatasetBuildContext):
        from transformers import AutoProcessor

        # Initialize and store processor
        self._processor = AutoProcessor.from_pretrained(self.hf_processor_path, trust_remote_code=True)

        base_examples = self._make_base_examples()

        def _maybe_make(size: int) -> Optional[VLMConversationDataset]:
            if not size or size <= 0:
                return None
            return VLMConversationDataset(
                base_examples=base_examples,
                target_length=size,
                processor=self._processor,
                collate_impl=None,  # infer collate from processor type (qwen2_5_collate_fn)
            )

        train_ds = _maybe_make(context.train_samples)
        valid_ds = _maybe_make(context.valid_samples)
        test_ds = _maybe_make(context.test_samples)

        return train_ds, valid_ds, test_ds
