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

import logging
from dataclasses import dataclass
from typing import Optional, Union

from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider
from megatron.bridge.diffusion.data.common.diffusion_energon_datamodule import (
    DiffusionDataModule,
    DiffusionDataModuleConfig,
)
from megatron.bridge.diffusion.data.wan.wan_taskencoder import WanTaskEncoder


logger = logging.getLogger(__name__)


def _is_empty_path(path: Optional[Union[str, list]]) -> bool:
    """Return True if path should be treated as empty (use mock data)."""
    if path is None:
        return True
    if isinstance(path, str):
        return not path.strip()
    if isinstance(path, list):
        return len(path) == 0
    return True


@dataclass(kw_only=True)
class WanDatasetConfig(DatasetProvider):
    """
    Unified WAN dataset config: mock vs real is decided at runtime in build_datasets()
    based on whether `path` is set (same pattern as FLUX recipe with dataset.path).

    Use this in the recipe with path=None by default. Override with dataset.path=/path/to/wds
    to load real data; no separate --mock flag needed.
    """

    path: Optional[Union[str, list]] = None
    seq_length: int = 1024
    packing_buffer_size: Optional[int] = None
    micro_batch_size: int = 1
    global_batch_size: int = 4
    num_workers: int = 16
    dataloader_type: str = "external"
    # Mock-only params (used when path is empty)
    F_latents: int = 24
    H_latents: int = 104
    W_latents: int = 60
    patch_spatial: int = 2
    patch_temporal: int = 1
    number_packed_samples: int = 1
    context_seq_len: int = 512
    context_embeddings_dim: int = 4096

    def __post_init__(self):
        self.sequence_length = self.seq_length

    def build_datasets(self, context: DatasetBuildContext):
        if _is_empty_path(self.path):
            logger.info(
                "WAN dataset: path is None or empty; using mock/synthetic data. "
                "Set dataset.path=/path/to/wds to use real data."
            )
            from megatron.bridge.diffusion.data.wan.wan_mock_datamodule import WanMockDataModuleConfig

            mock_cfg = WanMockDataModuleConfig(
                seq_length=self.seq_length,
                packing_buffer_size=self.packing_buffer_size,
                micro_batch_size=self.micro_batch_size,
                global_batch_size=self.global_batch_size,
                num_workers=self.num_workers,
                F_latents=self.F_latents,
                H_latents=self.H_latents,
                W_latents=self.W_latents,
                patch_spatial=self.patch_spatial,
                patch_temporal=self.patch_temporal,
                number_packed_samples=self.number_packed_samples,
                context_seq_len=self.context_seq_len,
                context_embeddings_dim=self.context_embeddings_dim,
            )
            return mock_cfg.build_datasets(context)

        # Real data: path is set (string or list)
        real_cfg = WanDataModuleConfig(
            path=self.path,
            seq_length=self.seq_length,
            packing_buffer_size=self.packing_buffer_size,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
        )
        return real_cfg.build_datasets(context)


@dataclass(kw_only=True)
class WanDataModuleConfig(DiffusionDataModuleConfig):  # noqa: D101
    path: str
    seq_length: int
    packing_buffer_size: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int
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
        if context is not None and context.pg_collection is not None:
            self.dataset.pg_collection = context.pg_collection
        return self.dataset.train_dataloader(), self.dataset.val_dataloader(), self.dataset.val_dataloader()
