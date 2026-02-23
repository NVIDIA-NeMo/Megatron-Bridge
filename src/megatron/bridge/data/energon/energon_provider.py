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
    # Enable batch-level online sequence packing
    pack_sequences_in_batch: bool = False

    def build_datasets(self, context: DatasetBuildContext):
        dataset = EnergonMultiModalDataModule(
            path=self.path,
            tokenizer=context.tokenizer if context.tokenizer is not None else self.tokenizer,
            image_processor=self.image_processor,
            seq_length=self.seq_length,
            task_encoder=self.task_encoder,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
        )
        return (
            iter(dataset.train_dataloader()),
            iter(dataset.val_dataloader()),
            iter(dataset.val_dataloader()),
        )
