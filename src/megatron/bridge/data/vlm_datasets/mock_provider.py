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
from typing import Any, Literal

import numpy
from PIL import Image

from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


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
    image_size: tuple[int, int] = (256, 256)
    pad_to_max_length: bool = True
    create_attention_mask: bool = True

    # Keep parity with GPTDatasetConfig usage in batching utilities
    skip_getting_attention_mask_from_dataset: bool = True

    # Number of images per sample
    num_images: int = 1

    # Default dataloader type for VLM providers
    dataloader_type: Literal["single", "cyclic", "external"] | None = "single"

    # HF AutoProcessor instance will be set during build
    _processor: Any | None = None

    def _make_base_examples(self) -> list[dict[str, Any]]:
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
            {"role": "assistant", "content": [{"type": "text", "text": "dummy assistant response"}]},
        ]
        return [{"conversation": messages}]

    def build_datasets(self, context: DatasetBuildContext):
        from transformers import AutoProcessor

        # Initialize and store processor
        self._processor = AutoProcessor.from_pretrained(self.hf_processor_path, trust_remote_code=True)

        base_examples = self._make_base_examples()

        def _maybe_make(size: int) -> VLMConversationDataset | None:
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
