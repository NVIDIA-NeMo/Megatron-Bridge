# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
from dataclasses import dataclass
from typing import Any, Optional

from torch import int_repr

from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule
from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider


@dataclass(kw_only=True)
class EnergonProvider(DatasetProvider):
    """Energon Provider."""

    path: str
    image_processor: Optional[Any] = None
    seq_length: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int_repr
    dataloader_type: str = "external"
    task_encoder: Optional[Any] = None
    # Legacy in-batch packing switch. Keep for compatibility with existing recipes.
    pack_sequences_in_batch: bool = False
    # THD dataloader switch for batch-level online packing.
    # This is independent from pack_sequences_in_batch.
    batch_level_packing: bool = False
    packing_buffer_size: Optional[int] = None
    shuffle_buffer_size: int = 100
    # Optional bin selector for datasets split into bin directories.
    # Used to pin data selection and align comparisons with Energon BSHD.
    cord_bins_root: Optional[str] = None
    cord_bin_prefix: str = "cord_bin_"
    cord_bin_id: Optional[str] = None

    def build_datasets(self, context: DatasetBuildContext):
        resolved_path = self.path
        if self.cord_bin_id is not None and self.cord_bin_id != "":
            assert (
                self.cord_bins_root
            ), "EnergonProvider.cord_bins_root must be set when dataset.cord_bin_id is provided."
            resolved_path = os.path.join(self.cord_bins_root, f"{self.cord_bin_prefix}{self.cord_bin_id}")

        assert resolved_path, "EnergonProvider.path must be set. Use CLI override: dataset.path=<path>"
        if self.task_encoder is not None and hasattr(self.task_encoder, "seq_len"):
            self.task_encoder.seq_len = self.seq_length
            self.task_encoder.seq_length = self.seq_length
        effective_packing_buffer_size = self.packing_buffer_size if self.batch_level_packing else None
        dataset = EnergonMultiModalDataModule(
            path=resolved_path,
            tokenizer=context.tokenizer if context.tokenizer is not None else self.tokenizer,
            image_processor=self.image_processor,
            seq_length=self.seq_length,
            task_encoder=self.task_encoder,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
            packing_buffer_size=effective_packing_buffer_size,
            shuffle_buffer_size=self.shuffle_buffer_size,
            pg_collection=context.pg_collection,
        )
        return (
            iter(dataset.train_dataloader()),
            iter(dataset.val_dataloader()),
            iter(dataset.val_dataloader()),
        )
