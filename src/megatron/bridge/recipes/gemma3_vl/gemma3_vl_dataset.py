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
from typing import Any, Dict, Optional, Tuple, List

import numpy
import torch
from PIL import Image

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider



class MockGemma3VLDataset(torch.utils.data.Dataset):
    """Mock vision-language dataset for Qwen2.5-VL that yields text+image samples.
    """
    def __init__(self, size: int, config: Any) -> None:
        if Image is None:
            raise ImportError("PIL is required for MockGemma3VLDataset. Please install pillow.")

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
    
    def _build_inputs(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Build chat template with one image and a simple text prompt
        num_images = max(0, int(getattr(self.config, "num_images", 1)))
        content = [{"type": "image"} for _ in range(num_images)]
        content.append({"type": "text", "text": self.config.prompt})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "dummy assistant response"},
        ]
        # The chat template will insert appropriate placeholders for the image token(s)
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
        if cur_len < target_len:
            input_ids = torch.nn.functional.pad(input_ids, (0, target_len - cur_len), value=self._hf_tokenizer.pad_token_id)
        elif cur_len > target_len:
            input_ids = input_ids[:target_len]
        return input_ids, inputs.pixel_values
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # noqa: ARG002 - idx unused randomness ok
        input_ids, pixel_values = self._build_inputs()
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
        sample: Dict[str, torch.Tensor] = {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        return sample

@dataclass(kw_only=True)
class MockGemma3VLDatasetProvider(DatasetProvider):
    """DatasetProvider for a mock Gemma3VL vision-language dataset.

    Builds train/valid/test datasets using a HF AutoProcessor and the
    MockGemma3VLDataset implementation.
    """

    # Required to match model.seq_length
    sequence_length: int

    # HF processor/model ID for Gemma3VL-4b
    hf_model_path: str = "google/gemma-3-4b-it"

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
        """Create mock Gemma3VL-4b datasets for train/valid/test splits.

        Args:
            context: Provides sample counts and optional tokenizer.

        Returns:
            Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]
        """

        from transformers import AutoProcessor

        # Initialize and store processor on the provider so the dataset can use it
        self._processor = AutoProcessor.from_pretrained(self.hf_model_path, trust_remote_code=True)

        def _maybe_make(size: int) -> Optional[MockGemma3VLDataset]:
            return MockGemma3VLDataset(size=size, config=self) if size and size > 0 else None

        train_ds = _maybe_make(context.train_samples)
        valid_ds = _maybe_make(context.valid_samples)
        test_ds = _maybe_make(context.test_samples)

        return train_ds, valid_ds, test_ds
