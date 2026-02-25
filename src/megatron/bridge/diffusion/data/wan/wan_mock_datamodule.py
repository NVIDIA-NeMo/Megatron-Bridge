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

import torch
from torch.utils.data import DataLoader, Dataset

from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider
from megatron.bridge.diffusion.models.wan.utils import patchify


class _MockDataset(Dataset):
    def __init__(self, length: int):
        self.length = max(int(length), 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        return {}


def mock_batch(  # noqa: D103
    F_latents: int,
    H_latents: int,
    W_latents: int,
    patch_temporal: int,
    patch_spatial: int,
    number_packed_samples: int,
    context_seq_len: int,
    context_embeddings_dim: int,
) -> dict:
    # set mock values for one video sample
    video_latent = torch.randn(16, F_latents, H_latents, W_latents, dtype=torch.float32)
    grid_size = torch.tensor(
        [
            video_latent.shape[1] // patch_temporal,
            video_latent.shape[2] // patch_spatial,
            video_latent.shape[3] // patch_spatial,
        ],
        dtype=torch.int32,
    )
    video_latent = patchify([video_latent], (patch_temporal, patch_spatial, patch_spatial))[0]
    video_latent = torch.as_tensor(video_latent, dtype=torch.float32)
    seq_len_q = video_latent.shape[0]
    seq_len_q_padded = seq_len_q
    loss_mask = torch.ones(seq_len_q, dtype=torch.bfloat16)
    context_embeddings = torch.randn(context_seq_len, context_embeddings_dim, dtype=torch.float32)
    seq_len_kv = context_embeddings.shape[0]
    seq_len_kv_padded = seq_len_kv
    video_metadata = {}

    # set mock values for packed video samples
    video_latents_packed = [video_latent for _ in range(number_packed_samples)]
    video_latents_packed = torch.cat(video_latents_packed, dim=0)
    loss_masks_packed = [loss_mask for _ in range(number_packed_samples)]
    loss_masks_packed = torch.cat(loss_masks_packed, dim=0)
    seq_len_q_packed = torch.tensor([seq_len_q for _ in range(number_packed_samples)], dtype=torch.int32)
    seq_len_q_padded_packed = torch.tensor([seq_len_q_padded for _ in range(number_packed_samples)], dtype=torch.int32)
    seq_len_kv_packed = torch.tensor([seq_len_kv for _ in range(number_packed_samples)], dtype=torch.int32)
    seq_len_kv_padded_packed = torch.tensor(
        [seq_len_kv_padded for _ in range(number_packed_samples)], dtype=torch.int32
    )
    grid_sizes_packed = torch.stack([grid_size for _ in range(number_packed_samples)], dim=0)
    context_embeddings_packed = [context_embeddings for _ in range(number_packed_samples)]
    context_embeddings_packed = torch.cat(context_embeddings_packed, dim=0)

    ### Note: shape of sample's values
    # video_latent: [num_patches, latents_channels * pF * pH * pW]
    # grid_size: [F_patches, W_patches, H_patches]
    # context_embeddings: [context_seq_len, text_embedding_dim]

    batch = dict(
        video_latents=video_latents_packed.unsqueeze(1),
        context_embeddings=context_embeddings_packed.unsqueeze(1),
        loss_mask=loss_masks_packed.unsqueeze(1),
        seq_len_q=seq_len_q_packed,
        seq_len_q_padded=seq_len_q_padded_packed,
        seq_len_kv=seq_len_kv_packed,
        seq_len_kv_padded=seq_len_kv_padded_packed,
        grid_sizes=grid_sizes_packed,
        video_metadata=video_metadata,
    )

    return batch


@dataclass(kw_only=True)
class WanMockDataModuleConfig(DatasetProvider):  # noqa: D101
    path: str = ""
    seq_length: int
    packing_buffer_size: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int
    dataloader_type: str = "external"
    F_latents: int = 24
    H_latents: int = 104
    W_latents: int = 60
    patch_spatial: int = 2
    patch_temporal: int = 1
    number_packed_samples: int = 1
    context_seq_len: int = 512
    context_embeddings_dim: int = 4096

    def __post_init__(self):
        mock_ds = _MockDataset(length=1024)
        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 8
        self._train_dl = DataLoader(
            mock_ds,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda samples: mock_batch(
                F_latents=self.F_latents,
                H_latents=self.H_latents,
                W_latents=self.W_latents,
                patch_temporal=self.patch_temporal,
                patch_spatial=self.patch_spatial,
                number_packed_samples=self.number_packed_samples,
                context_seq_len=self.context_seq_len,
                context_embeddings_dim=self.context_embeddings_dim,
            ),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            **kwargs,
        )
        self._train_dl = iter(self._train_dl)
        self.sequence_length = self.seq_length

    def build_datasets(self, _context: DatasetBuildContext):
        if hasattr(self, "dataset"):
            return self.dataset.train_dataloader(), self.dataset.train_dataloader(), self.dataset.train_dataloader()
        return self._train_dl, self._train_dl, self._train_dl
