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
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy
from PIL import Image

from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


@dataclass(kw_only=True)
class MockVLMConversationProvider(DatasetProvider):
    """DatasetProvider for generic mock VLM conversation datasets.

    Builds train/valid/test datasets using a HF AutoProcessor and the
    `MockVLMConversationDataset` implementation. Intended to work across
    different VLM models whose processors support the conversation schema.
    """

    # Required to match model.seq_length
    seq_length: int

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

    # Default dataloader type for VLM providers
    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = "single"

    # HF AutoProcessor instance will be set during build
    _processor: Optional[Any] = None

    # Enable batch-level online sequence packing
    pack_sequences_in_batch: bool = False

    def _make_single_example(
        self, rng: numpy.random.Generator, prompt_text: str, response_text: str
    ) -> Dict[str, Any]:
        """Create a single mock conversation example with the given prompt and response text."""
        num_images = max(0, int(getattr(self, "num_images", 1)))
        w, h = self.image_size
        images = None
        if num_images > 0:
            images = [
                Image.fromarray(rng.integers(low=0, high=256, size=(h, w, 3), dtype=numpy.uint8), mode="RGB")
                for _ in range(num_images)
            ]

        content = [{"type": "image", "image": img} for img in images] if images is not None else []
        content.append({"type": "text", "text": prompt_text})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
        ]
        return {"conversation": messages}

    def _make_base_examples(self) -> List[Dict[str, Any]]:
        rng = numpy.random.default_rng(seed=self.random_seed)

        # Generate many diverse examples with random responses so the model
        # cannot memorize the data in a few iterations (keeps grad_norm non-zero).
        _VOCAB = (
            "the a is was are were have has had do does did will would could should "
            "may might can need to of in for on with at by from image shows depicts "
            "contains features displays large small red blue green bright dark light "
            "object scene background foreground color shape person animal building "
            "tree sky water ground left right top bottom center middle edge beautiful "
            "complex simple detailed abstract natural moving standing sitting running "
            "walking flying and or but so yet nor not very this that these those here "
            "there where when how what which who whom whose each every all both few "
            "many much some any no other another such"
        ).split()

        num_examples = 1000

        if self.pack_sequences_in_batch:
            # When packing is enabled, produce examples with varied response lengths
            # so that the packing logic concatenates sequences of different sizes.
            resp_len_range = (10, 100)
        else:
            # Without packing, keep responses short (10-30 words) to maintain similar
            # sequence lengths, since the collate pads to batch-max.
            resp_len_range = (10, 30)

        examples = []
        for _ in range(num_examples):
            resp_len = int(rng.integers(*resp_len_range))
            response = " ".join(rng.choice(_VOCAB, size=resp_len))
            examples.append(self._make_single_example(rng, self.prompt, response))

        return examples

    def build_datasets(self, context: DatasetBuildContext):
        from transformers import AutoProcessor

        # Initialize and store processor
        self._processor = AutoProcessor.from_pretrained(
            self.hf_processor_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=self.hf_processor_path,
            ),
        )

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
