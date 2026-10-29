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

"""Serializable config and runtime builder for synthetic VLM conversations."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy
from PIL import Image
from transformers import AutoProcessor

from megatron.bridge.data.base import DataloaderConfig, DatasetBuildContext
from megatron.bridge.data.datasets.direct_sft import DirectSFTDataset
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


_MOCK_RESPONSE_VOCABULARY = (
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


@dataclass(kw_only=True)
class MockVLMSFTDatasetConfig(DataloaderConfig):
    """Serializable settings for synthetic conversation-style VLM data."""

    seq_length: int
    hf_processor_path: str
    prompt: str = "Describe this image."
    random_seed: int = 0
    image_size: tuple[int, int] = (256, 256)
    num_images: int = 1
    num_base_examples: int = 1000
    skip_getting_attention_mask_from_dataset: bool = True
    dataloader_type: Literal["single", "cyclic"] | None = "single"
    enable_in_batch_packing: bool = False
    defer_in_batch_packing_to_step: bool = False
    pad_to_max_length: bool = False
    pad_to_multiple_of: int = 128
    in_batch_packing_pad_to_multiple_of: int = 1

    def validate(self) -> None:
        """Validate synthetic data settings."""
        if self.seq_length <= 0:
            raise ValueError("seq_length must be greater than 0.")
        if not isinstance(self.hf_processor_path, str) or not self.hf_processor_path.strip():
            raise ValueError("hf_processor_path must be a non-empty string.")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string.")
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError("image_size must contain two positive dimensions.")
        if self.num_images < 0:
            raise ValueError("num_images must be greater than or equal to 0.")
        if self.num_base_examples <= 0:
            raise ValueError("num_base_examples must be greater than 0.")
        if self.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be greater than 0.")
        if self.in_batch_packing_pad_to_multiple_of <= 0:
            raise ValueError("in_batch_packing_pad_to_multiple_of must be greater than 0.")

    def finalize(self) -> None:
        """Finalize dataloader settings and validate this config."""
        super().finalize()
        self.validate()


def make_mock_vlm_example(
    config: MockVLMSFTDatasetConfig,
    rng: numpy.random.Generator,
    response_text: str,
) -> dict[str, Any]:
    """Create one synthetic conversation with the configured number of images."""
    width, height = config.image_size
    images = [
        Image.fromarray(rng.integers(low=0, high=256, size=(height, width, 3), dtype=numpy.uint8), mode="RGB")
        for _ in range(config.num_images)
    ]
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": config.prompt})
    return {
        "conversation": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
        ]
    }


def make_mock_vlm_examples(config: MockVLMSFTDatasetConfig) -> list[dict[str, Any]]:
    """Generate the deterministic base examples reused by every requested split."""
    rng = numpy.random.default_rng(seed=config.random_seed)
    response_length_range = (10, 100) if config.enable_in_batch_packing else (10, 30)
    examples = []
    for _ in range(config.num_base_examples):
        response_length = int(rng.integers(*response_length_range))
        response = " ".join(rng.choice(_MOCK_RESPONSE_VOCABULARY, size=response_length))
        examples.append(make_mock_vlm_example(config, rng, response))
    return examples


def build_mock_vlm_sft_split(
    config: MockVLMSFTDatasetConfig,
    base_examples: list[dict[str, Any]],
    target_length: int,
    processor: Any,
) -> DirectSFTDataset | None:
    """Build one requested synthetic VLM split."""
    if target_length <= 0:
        return None
    return DirectSFTDataset(
        base_examples=base_examples,
        target_length=target_length,
        processor=processor,
        collate_impl=None,
        sequence_length=config.seq_length,
        pad_to_max_length=config.pad_to_max_length,
        pad_to_multiple_of=config.pad_to_multiple_of,
        enable_in_batch_packing=config.enable_in_batch_packing,
        defer_in_batch_packing_to_step=config.defer_in_batch_packing_to_step,
        in_batch_packing_pad_to_multiple_of=config.in_batch_packing_pad_to_multiple_of,
    )


class MockVLMSFTDatasetBuilder:
    """Build synthetic VLM datasets from declarative settings."""

    def __init__(self, config: MockVLMSFTDatasetConfig) -> None:
        config.validate()
        self.config = config

    def build(
        self,
        context: DatasetBuildContext,
    ) -> tuple[DirectSFTDataset | None, DirectSFTDataset | None, DirectSFTDataset | None]:
        """Build the train, validation, and test splits requested by the schedule."""
        processor = AutoProcessor.from_pretrained(
            self.config.hf_processor_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.config.trust_remote_code,
                hf_path=self.config.hf_processor_path,
            ),
        )
        base_examples = make_mock_vlm_examples(self.config)
        return (
            build_mock_vlm_sft_split(self.config, base_examples, context.train_samples, processor),
            build_mock_vlm_sft_split(self.config, base_examples, context.valid_samples, processor),
            build_mock_vlm_sft_split(self.config, base_examples, context.test_samples, processor),
        )


def mock_vlm_sft_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int],
    dataset_config: MockVLMSFTDatasetConfig,
    tokenizer: Any | None = None,
    pg_collection: Any | None = None,
) -> tuple[DirectSFTDataset | None, DirectSFTDataset | None, DirectSFTDataset | None]:
    """Build synthetic VLM splits through the canonical runtime builder."""
    context = DatasetBuildContext(
        train_samples=train_val_test_num_samples[0],
        valid_samples=train_val_test_num_samples[1],
        test_samples=train_val_test_num_samples[2],
        tokenizer=tokenizer,
        pg_collection=pg_collection,
    )
    return MockVLMSFTDatasetBuilder(dataset_config).build(context)
