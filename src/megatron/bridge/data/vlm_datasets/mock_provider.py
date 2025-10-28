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
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.state import TrainState
from megatron.bridge.training.utils.sig_utils import DistributedSignalHandler
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.core import mpu
from torch.utils.data import DataLoader


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

    # Default dataloader type for VLM providers
    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = "single"

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
            {"role": "assistant", "content": [{"type": "text", "text": "dummy assistant response"}]},
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

    def provide_dataloaders(
        self,
        context: DatasetBuildContext,
        cfg: ConfigContainer,
        train_state: TrainState,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        train_ds, valid_ds, test_ds = self.build_datasets(context)

        def worker_init_fn(_):
            DistributedSignalHandler(cfg.train.exit_signal).__enter__()

        maybe_worker_init_fn = worker_init_fn if cfg.train.exit_signal_handler_for_dataloader else None

        def _make_loader(
            ds: Optional[VLMConversationDataset],
            consumed_samples: int,
            dataloader_type: str,
        ) -> Optional[DataLoader]:
            if ds is None:
                return None
            return build_pretraining_data_loader(
                ds,
                consumed_samples,
                dataloader_type,
                cfg.train.micro_batch_size,
                cfg.dataset.num_workers,
                cfg.dataset.data_sharding,
                worker_init_fn=maybe_worker_init_fn,
                collate_fn=ds.collate_fn if hasattr(ds, "collate_fn") else None,
                pin_memory=cfg.dataset.pin_memory,
                persistent_workers=cfg.dataset.persistent_workers,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
                global_batch_size=cfg.train.global_batch_size,
            )

        train_loader = _make_loader(
            train_ds,
            train_state.consumed_train_samples,
            cfg.dataset.dataloader_type,
        )

        valid_loader: Optional[DataLoader] = None
        if cfg.train.eval_iters > 0:
            valid_dl_type = cfg.dataset.dataloader_type if cfg.train.skip_train else "cyclic"
            valid_loader = _make_loader(
                valid_ds,
                train_state.consumed_valid_samples if not cfg.train.skip_train else 0,
                valid_dl_type,
            )

        test_loader: Optional[DataLoader] = None
        if cfg.train.eval_iters > 0:
            test_loader = _make_loader(
                test_ds,
                0,
                cfg.dataset.dataloader_type,
            )

        return train_loader, valid_loader, test_loader
