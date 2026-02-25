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

from typing import List

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.energon import SkipSample
from megatron.energon.task_encoder.base import stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys

from megatron.bridge.diffusion.data.common.diffusion_sample import DiffusionSample
from megatron.bridge.diffusion.data.common.diffusion_task_encoder_with_sp import DiffusionTaskEncoderWithSequencePacking
from megatron.bridge.diffusion.models.wan.utils import grid_sizes_calculation, patchify


def cook(sample: dict) -> dict:
    """
    Processes a raw sample dictionary from energon dataset and returns a new dictionary with specific keys.

    Args:
        sample (dict): The input dictionary containing the raw sample data.

    Returns:
        dict: A new dictionary containing the processed sample data with the following keys:
            - All keys from the result of `basic_sample_keys(sample)`
            - 'json': The contains meta data like resolution, aspect ratio, fps, etc.
            - 'pth': contains video latent tensor
            - 'pickle': contains text embeddings
    """
    return dict(
        **basic_sample_keys(sample),
        json=sample["json"],
        pth=sample["pth"],
        pickle=sample["pickle"],
    )


class WanTaskEncoder(DiffusionTaskEncoderWithSequencePacking):
    """
    Task encoder for Wan dataset.
    Attributes:
        cookers (list): A list of Cooker objects used for processing.
        patch_spatial (int): The spatial patch size. Defaults to 2.
        patch_temporal (int): The temporal patch size. Defaults to 1.
        seq_length (int): The sequence length. Defaults to 1024.
    """

    cookers = [
        Cooker(cook),
    ]

    def __init__(
        self,
        *args,
        max_frames: int = None,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        seq_length: int = 1024,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.seq_length = seq_length

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: dict) -> dict:
        video_latent = sample["pth"]
        context_embeddings = sample["pickle"]
        video_metadata = sample["json"]

        # sanity quality check
        if torch.isnan(video_latent).any() or torch.isinf(video_latent).any():
            raise SkipSample()
        if torch.max(torch.abs(video_latent)) > 1e3:
            raise SkipSample()

        # calculate grid size
        grid_size = grid_sizes_calculation(
            input_shape=video_latent.shape[1:],
            patch_size=(self.patch_temporal, self.patch_spatial, self.patch_spatial),
        )

        # patchify video_latent
        video_latent = patchify([video_latent], (self.patch_temporal, self.patch_spatial, self.patch_spatial))[0]

        # process text embeddings
        # pad here for text embeddings
        context_max_len = 512
        context_embeddings = F.pad(context_embeddings, (0, 0, 0, context_max_len - context_embeddings.shape[0]))

        # calculate sequence length
        seq_len_q = video_latent.shape[0]
        seq_len_kv = context_embeddings.shape[0]

        # loss mask
        loss_mask = torch.ones(seq_len_q, dtype=torch.bfloat16)

        # CAVEAT:
        #   when using context parallelism, we need to pad batch sequence length to be divisible by [cp_rank*2]
        #   (because TransformerEngine's context parallelism requires "AssertionError: Sequence length per GPU needs to be divisible by 2!")
        if parallel_state.get_context_parallel_world_size() > 1:
            sharding_factor = parallel_state.get_context_parallel_world_size() * 2
            seq_len_q_padded = ((seq_len_q + sharding_factor - 1) // sharding_factor) * sharding_factor
            seq_len_kv_padded = ((seq_len_kv + sharding_factor - 1) // sharding_factor) * sharding_factor
        else:
            seq_len_q_padded = seq_len_q
            seq_len_kv_padded = seq_len_kv

        # padding
        if seq_len_q < seq_len_q_padded:
            video_latent = F.pad(video_latent, (0, 0, 0, seq_len_q_padded - seq_len_q))
            loss_mask = F.pad(loss_mask, (0, seq_len_q_padded - seq_len_q))
            context_embeddings = F.pad(context_embeddings, (0, 0, 0, seq_len_kv_padded - seq_len_kv))

        ### Note: shape of sample's values
        # video_latent: [num_patches, latents_channels * pF * pH * pW]
        # grid_size: [F_patches, W_patches, H_patches]
        # context_embeddings: [context_seq_len, text_embedding_dim]

        return DiffusionSample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample["__subflavors__"],
            video=video_latent,
            context_embeddings=context_embeddings,
            latent_shape=torch.tensor(grid_size, dtype=torch.int32),
            loss_mask=loss_mask,
            seq_len_q=torch.tensor([seq_len_q], dtype=torch.int32),
            seq_len_q_padded=torch.tensor([seq_len_q_padded], dtype=torch.int32),
            seq_len_kv=torch.tensor([seq_len_kv], dtype=torch.int32),
            seq_len_kv_padded=torch.tensor([seq_len_kv_padded], dtype=torch.int32),
            pos_ids=torch.zeros(1, dtype=torch.bfloat16),  # dummy pos_ids
            video_metadata=video_metadata,
        )

    # NOTE:
    # the method select_samples_to_pack() and pack_selected_samples() are inherited from the parent
    #   class DiffusionTaskEncoderWithSequencePacking

    @stateless
    def batch(self, samples: List[DiffusionSample]) -> dict:
        """Return dictionary with data for batch."""
        if self.packing_buffer_size is None:
            # no packing
            return super().batch(samples).to_dict()

        # packing
        sample = samples[0]

        # # CAVEAT:
        # #   when using pipeline parallelism, we need to set batch sequence length to DataModule's seq_length because
        # #   because pipeline parallelism requires pre-specified sequence length to create buffer
        # if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        #     if sample.video.shape[0] > self.seq_length:
        #         raise ValueError(
        #             f"video sequence length {sample.video.shape[0]} is greater than DataModule's seq_length {self.seq_length}"
        #         )
        #     else:
        #         # set max_video_seq_len to DataModule's seq_length
        #         padded_seq_len = self.seq_length

        batch = dict(
            video_latents=sample.video.unsqueeze(1),
            context_embeddings=sample.context_embeddings.unsqueeze(1),
            loss_mask=sample.loss_mask.unsqueeze(1) if sample.loss_mask is not None else None,
            seq_len_q=sample.seq_len_q,
            seq_len_q_padded=sample.seq_len_q_padded,
            seq_len_kv=sample.seq_len_kv,
            seq_len_kv_padded=sample.seq_len_kv_padded,
            grid_sizes=sample.latent_shape,
            video_metadata=sample.video_metadata,
        )

        ### Note: shape of batch's values
        # video_latents: [seq_len, 1, latents_channels * pF * pH * pW]
        # context_embeddings: [seq_len, 1, text_embedding_dim]
        # loss_mask: [seq_len, 1]
        # seq_len_q: [num_samples]
        # seq_len_q_padded: [num_samples]
        # seq_len_kv: [num_samples]
        # seq_len_kv_padded: [num_samples]
        # grid_sizes: [num_samples, 3]
        # video_metadata: [num_samples]

        return batch
