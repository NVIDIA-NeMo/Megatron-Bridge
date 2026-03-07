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

# pylint: disable=C0115,C0116,C0301

from dataclasses import dataclass

from torch import int_repr

from megatron.bridge.data.utils import DatasetBuildContext
from megatron.bridge.diffusion.data.common.diffusion_energon_datamodule import (
    DiffusionDataModule,
    DiffusionDataModuleConfig,
)
from megatron.bridge.diffusion.data.wan.wan_taskencoder import WanTaskEncoder


@dataclass(kw_only=True)
class WanDataModuleConfig(DiffusionDataModuleConfig):  # noqa: D101
    path: str
    seq_length: int
    packing_buffer_size: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int_repr
    dataloader_type: str = "external"

    def __post_init__(self):
        self.dataset = DiffusionDataModule(
            path=self.path,
            seq_length=self.seq_length,
            packing_buffer_size=self.packing_buffer_size,
            task_encoder=WanTaskEncoder(seq_length=self.seq_length, packing_buffer_size=self.packing_buffer_size),
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
        )
        self.sequence_length = self.dataset.seq_length

    def build_datasets(self, context: DatasetBuildContext):
        return self.dataset.train_dataloader(), self.dataset.train_dataloader(), self.dataset.train_dataloader()
