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
from typing import Optional

from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider
from megatron.bridge.diffusion.data.common.diffusion_energon_datamodule import (
    DiffusionDataModule,
    DiffusionDataModuleConfig,
)
from megatron.bridge.diffusion.data.flux.flux_taskencoder import FluxTaskEncoder


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class FluxDatasetConfig(DatasetProvider):
    """
    Unified FLUX dataset config: mock vs real is decided at runtime in build_datasets()
    based on whether `path` is set (same pattern as Gemma/LLM recipes with dataset.blend).

    Use this in the recipe with path=None by default. Override with dataset.path=/path/to/wds
    to load real data; no separate --data_paths or mock flag needed.
    """

    path: Optional[str] = None
    seq_length: int = 1024
    packing_buffer_size: Optional[int] = None
    micro_batch_size: int = 1
    global_batch_size: int = 4
    num_workers: int = 16
    dataloader_type: str = "external"
    vae_scale_factor: int = 8
    latent_channels: int = 16
    # Mock-only params (used when path is empty)
    image_H: int = 1024
    image_W: int = 1024
    prompt_seq_len: int = 512
    context_dim: int = 4096
    pooled_prompt_dim: int = 768

    def __post_init__(self):
        self.sequence_length = self.seq_length

    def build_datasets(self, context: DatasetBuildContext):
        if not (self.path or "").strip():
            logger.info(
                "FLUX dataset: path is None or empty; using mock/synthetic data. "
                "Set dataset.path=/path/to/wds to use real data."
            )
            from megatron.bridge.diffusion.data.flux.flux_mock_datamodule import (
                FluxMockDataModuleConfig,
            )

            mock_cfg = FluxMockDataModuleConfig(
                path="",
                seq_length=self.seq_length,
                packing_buffer_size=self.packing_buffer_size,
                micro_batch_size=self.micro_batch_size,
                global_batch_size=self.global_batch_size,
                num_workers=self.num_workers,
                dataloader_type=self.dataloader_type,
                image_H=self.image_H,
                image_W=self.image_W,
                vae_channels=self.latent_channels,
                vae_scale_factor=self.vae_scale_factor,
                prompt_seq_len=self.prompt_seq_len,
                context_dim=self.context_dim,
                pooled_prompt_dim=self.pooled_prompt_dim,
            )
            return mock_cfg.build_datasets(context)

        real_cfg = FluxDataModuleConfig(
            path=self.path,
            seq_length=self.seq_length,
            packing_buffer_size=self.packing_buffer_size,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
            vae_scale_factor=self.vae_scale_factor,
            latent_channels=self.latent_channels,
            task_encoder_seq_length=None,
        )
        return real_cfg.build_datasets(context)


@dataclass(kw_only=True)
class FluxDataModuleConfig(DiffusionDataModuleConfig):  # noqa: D101
    path: str
    seq_length: int
    packing_buffer_size: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int
    dataloader_type: str = "external"
    vae_scale_factor: int = 8
    latent_channels: int = 16

    def __post_init__(self):
        self.dataset = DiffusionDataModule(
            path=self.path,
            seq_length=self.seq_length,
            packing_buffer_size=self.packing_buffer_size,
            task_encoder=FluxTaskEncoder(
                seq_length=self.seq_length,
                packing_buffer_size=self.packing_buffer_size,
                vae_scale_factor=self.vae_scale_factor,
                latent_channels=self.latent_channels,
            ),
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
            use_train_split_for_val=True,
        )
        self.sequence_length = self.dataset.seq_length

    def build_datasets(self, context: DatasetBuildContext):
        return (
            iter(self.dataset.train_dataloader()),
            iter(self.dataset.val_dataloader()),
            iter(self.dataset.val_dataloader()),
        )
