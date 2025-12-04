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
Energon-backed provider for VLM training using Megatron-Energon.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

from megatron.energon import WorkerConfig, get_savable_loader, get_train_dataset
from transformers import AutoProcessor

from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider

from .task_encoder import Qwen2VLTaskEncoder


@dataclass(kw_only=True)
class EnergonVLMConversationProvider(DatasetProvider):
    """DatasetProvider that builds Energon dataloaders for VLM training.

    This provider mirrors the intent of EnergonMultiModalDataModule but conforms to
    Megatron-Bridge's DatasetProvider interface. It constructs Energon datasets and
    wraps them in savable loaders, returning them as "external" dataloaders to the
    training pipeline.
    """

    # Required to match model.seq_length (enforced by ConfigContainer.validate)
    seq_length: int

    # Path to Energon dataset (e.g., directory or manifest understood by Energon)
    path: str

    # HF processor/model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
    hf_processor_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Dataloader type must be "external" since Energon provides its own loader
    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = "external"

    # Energon dataset/runtime options
    shuffle_buffer_size: int = 100
    max_samples_per_sequence: Optional[int] = None
    packing_buffer_size: Optional[int] = None

    # Optional: pass through any additional Energon kwargs
    energon_kwargs: Optional[dict] = None

    # Keep parity with GPTDatasetConfig usage in batching utilities
    skip_getting_attention_mask_from_dataset: bool = True

    # Optional micro batch size for Energon loader (defaults to 1 if unset)
    micro_batch_size: Optional[int] = None

    def _build_split_loader(
        self,
        processor: Any,
        split: Literal["train", "val"],
        num_workers: int,
    ):
        # Worker configuration (non-distributed default; Energon handles distributed when applicable)
        worker_config = WorkerConfig.default_worker_config(num_workers)

        # Task encoder uses HF processor to encode text+vision into model-ready tensors
        task_encoder = Qwen2VLTaskEncoder(processor=processor, max_padding_length=self.seq_length)

        dataset = get_train_dataset(
            self.path,
            batch_size=int(self.micro_batch_size or 1),
            task_encoder=task_encoder,
            worker_config=worker_config,
            packing_buffer_size=self.packing_buffer_size,
            split_part=split,
            shuffle_buffer_size=self.shuffle_buffer_size,
            max_samples_per_sequence=self.max_samples_per_sequence,
            **(self.energon_kwargs or {}),
        )
        loader = get_savable_loader(dataset, worker_config=worker_config)
        return loader

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        # Bind processor for the requested model
        processor = AutoProcessor.from_pretrained(
            self.hf_processor_path,
            trust_remote_code=is_safe_repo(
                hf_path=self.hf_processor_path,
                trust_remote_code=self.trust_remote_code,
            ),
        )

        # For Energon-backed provider we expose loaders directly as "datasets" with dataloader_type="external".
        # The higher-level pipeline will pass them through unchanged.
        train_loader = None
        valid_loader = None
        test_loader = None

        if context.train_samples and context.train_samples > 0:
            # micro_batch_size is determined by training loop; we do not have access here.
            # Energon requires a batch size up-front, so we default to 1 and let the external loader
            # produce one sample at a time. Model-side global batching is controlled by the training loop.
            train_loader = self._build_split_loader(processor, "train", num_workers=self.num_workers)

        if context.valid_samples and context.valid_samples > 0:
            valid_loader = self._build_split_loader(processor, "val", num_workers=self.num_workers)

        # Energon path typically doesn't provide a separate "test" split. Return None by default.
        return train_loader, valid_loader, test_loader
