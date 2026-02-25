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
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider
from megatron.bridge.diffusion.data.dit.dit_taskencoder import PosID3D


pos_id_3d = PosID3D()


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
    C, T, H, W = video_latent.shape
    video_latent = rearrange(
        video_latent,
        "C (T pt) (H ph) (W pw) -> (T H W) (ph pw pt C)",
        ph=patch_spatial,
        pw=patch_spatial,
        pt=patch_temporal,
    )
    video_latent = torch.as_tensor(video_latent, dtype=torch.bfloat16)

    context_embeddings = torch.randn(context_seq_len, context_embeddings_dim, dtype=torch.bfloat16)
    context_embeddings_seq_length = context_embeddings.shape[0]
    context_embeddings_mask = torch.ones(context_embeddings_seq_length, dtype=torch.bfloat16)

    pos_ids = rearrange(
        pos_id_3d.get_pos_id_3d(t=T // patch_temporal, h=H // patch_spatial, w=W // patch_spatial),
        "T H W d -> (T H W) d",
    )

    seq_len_q = video_latent.shape[0]
    seq_len_q_padded = seq_len_q

    loss_mask = torch.ones(seq_len_q, dtype=torch.bfloat16)

    seq_len_kv = context_embeddings.shape[0]
    seq_len_kv_padded = seq_len_kv

    video_latents_packed = [video_latent for _ in range(number_packed_samples)]
    video_latents_packed = torch.cat(video_latents_packed, dim=0)

    context_embeddings_packed = [context_embeddings for _ in range(number_packed_samples)]
    context_embeddings_packed = torch.cat(context_embeddings_packed, dim=0)

    context_embeddings_mask_packed = [context_embeddings_mask for _ in range(number_packed_samples)]
    context_embeddings_mask_packed = torch.cat(context_embeddings_mask_packed, dim=0)

    loss_masks_packed = [loss_mask for _ in range(number_packed_samples)]
    loss_masks_packed = torch.cat(loss_masks_packed, dim=0)

    seq_len_q_packed = torch.tensor([seq_len_q for _ in range(number_packed_samples)], dtype=torch.int32)
    seq_len_q_padded_packed = torch.tensor([seq_len_q_padded for _ in range(number_packed_samples)], dtype=torch.int32)

    seq_len_kv_packed = torch.tensor([seq_len_kv for _ in range(number_packed_samples)], dtype=torch.int32)
    seq_len_kv_padded_packed = torch.tensor(
        [seq_len_kv_padded for _ in range(number_packed_samples)], dtype=torch.int32
    )

    pos_ids_packed = [pos_ids for _ in range(number_packed_samples)]
    pos_ids_packed = torch.cat(pos_ids_packed, dim=0)

    context_embeddings_packed = [context_embeddings for _ in range(number_packed_samples)]
    context_embeddings_packed = torch.cat(context_embeddings_packed, dim=0)

    batch = dict(
        video=video_latents_packed.unsqueeze(0),
        context_embeddings=context_embeddings_packed.unsqueeze(0),
        context_mask=context_embeddings_mask_packed.unsqueeze(0),
        loss_mask=loss_masks_packed.unsqueeze(0),
        seq_len_q=seq_len_q_packed,
        seq_len_q_padded=seq_len_q_padded_packed,
        seq_len_kv=seq_len_kv_packed,
        seq_len_kv_padded=seq_len_kv_padded_packed,
        latent_shape=torch.tensor([[C, T, H, W] for _ in range(number_packed_samples)], dtype=torch.int32),
        pos_ids=pos_ids_packed.unsqueeze(0),
        video_metadata=[{"caption": f"Mock video sample {i}"} for i in range(number_packed_samples)],
    )

    return batch


@dataclass(kw_only=True)
class DiTMockDataModuleConfig(DatasetProvider):  # noqa: D101
    path: str = ""
    seq_length: int
    packing_buffer_size: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int
    dataloader_type: str = "external"
    task_encoder_seq_length: int = None
    F_latents: int = 1
    H_latents: int = 256
    W_latents: int = 512
    patch_spatial: int = 2
    patch_temporal: int = 1
    number_packed_samples: int = 1
    context_seq_len: int = 512
    context_embeddings_dim: int = 1024

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
