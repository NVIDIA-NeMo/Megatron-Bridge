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

from dataclasses import dataclass
from typing import Any

import numpy
import torch
from PIL import Image

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


class MockQwen25VLDataset(torch.utils.data.Dataset):
    """Mock vision-language dataset for Qwen2.5-VL that yields text+image samples.

    Each sample contains:
      - tokens: torch.LongTensor [L]
      - labels: torch.LongTensor [L]
      - attention_mask: torch.BoolTensor [L] (all ones by default)
      - loss_mask: torch.FloatTensor [L]
      - position_ids: torch.LongTensor [L]
      - pixel_values: torch.FloatTensor [num_images, C, H, W]
      - image_grid_thw: torch.LongTensor [num_images, 3]
    """

    def __init__(self, size: int, config: Any) -> None:
        if Image is None:
            raise ImportError("PIL is required for MockQwen25VLDataset. Please install pillow.")

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
        # Generate a simple RGB image with uniform noise
        array = self._rng.integers(low=0, high=256, size=(h, w, 3), dtype=numpy.uint8)
        return Image.fromarray(array, mode="RGB")

    def _build_inputs(self) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Build chat template with one image and a simple text prompt
        num_images = max(0, int(getattr(self.config, "num_images", 1)))
        content = [{"type": "image"} for _ in range(num_images)]
        content.append({"type": "text", "text": self.config.prompt})
        messages = [
            {
                "role": "user",
                "content": content,
            },
            {"role": "assistant", "content": "dummy assistant response"},
        ]

        # The chat template will insert appropriate placeholders for the image token(s)
        text = self.config._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        images: list[Image.Image] | None = None
        if num_images > 0:
            images = [self._generate_random_image() for _ in range(num_images)]

        processor_kwargs: dict[str, Any] = {
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
        pixel_values_t: torch.Tensor | None = None
        image_grid_thw_t: torch.Tensor | None = None
        if images is not None:
            # Ensure per-sample shapes without a leading batch dim
            pixel_values_t = inputs.pixel_values
            if pixel_values_t.dim() == 5 and pixel_values_t.size(0) == 1:
                pixel_values_t = pixel_values_t.squeeze(0)
            image_grid_thw_t = inputs.image_grid_thw
            if image_grid_thw_t.dim() == 3 and image_grid_thw_t.size(0) == 1:
                image_grid_thw_t = image_grid_thw_t.squeeze(0)
        return input_ids, pixel_values_t, image_grid_thw_t

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # noqa: ARG002 - idx unused randomness ok
        input_ids, pixel_values, image_grid_thw = self._build_inputs()

        # Ensure at least length 2 to form tokens/labels by shifting
        if input_ids.numel() < 2:
            # In rare cases of extremely short templates, pad minimally
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
        # Additionally mask special pad token ids if present
        loss_mask[labels == pad_token_id] = 0.0

        # For embedding lookup safety on padding
        tokens = tokens.clone()
        labels = labels.clone()
        tokens[tokens == pad_token_id] = 0
        labels[labels == pad_token_id] = 0

        sample: dict[str, torch.Tensor] = {
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
class MockQwen25VLDatasetProvider(DatasetProvider):
    """DatasetProvider for a mock Qwen2.5-VL vision-language dataset.

    Builds train/valid/test datasets using a HF AutoProcessor and the
    MockQwen25VLDataset implementation.
    """

    # Required to match model.seq_length
    sequence_length: int

    # HF processor/model ID for Qwen2.5-VL
    hf_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Sample generation options
    prompt: str = "Describe this image."
    random_seed: int = 0
    image_size: tuple[int, int] = (256, 256)
    pad_to_max_length: bool = True
    create_attention_mask: bool = True

    # Keep parity with GPTDatasetConfig usage in batching utilities
    skip_getting_attention_mask_from_dataset: bool = True

    # Number of images per sample
    num_images: int = 1

    # HF AutoProcessor instance will be set during build
    _processor: Any | None = None

    def build_datasets(self, context: DatasetBuildContext):
        """Create mock Qwen2.5-VL datasets for train/valid/test splits.

        Args:
            context: Provides sample counts and optional tokenizer.

        Returns:
            tuple[Dataset | None, Dataset | None, Dataset | None]
        """

        from transformers import AutoProcessor

        # Initialize and store processor on the provider so the dataset can use it
        self._processor = AutoProcessor.from_pretrained(self.hf_model_path, trust_remote_code=True)

        def _maybe_make(size: int) -> MockQwen25VLDataset | None:
            return MockQwen25VLDataset(size=size, config=self) if size and size > 0 else None

        train_ds = _maybe_make(context.train_samples)
        valid_ds = _maybe_make(context.valid_samples)
        test_ds = _maybe_make(context.test_samples)

        return train_ds, valid_ds, test_ds
