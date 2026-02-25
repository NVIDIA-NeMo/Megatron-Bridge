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

import warnings
from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.core import parallel_state
from megatron.energon import SkipSample, stateless

from megatron.bridge.diffusion.data.common.diffusion_sample import DiffusionSample
from megatron.bridge.diffusion.data.common.diffusion_task_encoder_with_sp import DiffusionTaskEncoderWithSequencePacking


class DiTTaskEncoder(DiffusionTaskEncoderWithSequencePacking):
    """
    BasicDiffusionTaskEncoder is a class that encodes image/video samples for diffusion tasks.
    Attributes:
        cookers (list): A list of Cooker objects used for processing.
        max_frames (int, optional): The maximum number of frames to consider from the video. Defaults to None.
        text_embedding_max_length (int): The maximum length for text embeddings. Defaults to 512.
    Methods:
        __init__(*args, max_frames=None, text_embedding_max_size=512, **kwargs):
            Initializes the BasicDiffusionTaskEncoder with optional maximum frames and text embedding padding size.
        encode_sample(sample: dict) -> dict:
            Encodes a given sample dictionary containing video and text data.
            Args:
                sample (dict): A dictionary containing 'pth' for video latent and 'json' for additional info.
            Returns:
                dict: A dictionary containing encoded video, text embeddings, text mask, and loss mask.
            Raises:
                SkipSample: If the video latent contains NaNs, Infs, or is not divisible by the tensor parallel size.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: dict) -> DiffusionSample:
        video_latent = sample["pth"]

        if torch.isnan(video_latent).any() or torch.isinf(video_latent).any():
            raise SkipSample()
        if torch.max(torch.abs(video_latent)) > 1e3:
            raise SkipSample()

        info = sample["json"]
        video_latent = video_latent.squeeze(0)
        C, T, H, W = video_latent.shape
        seq_len = (
            video_latent.shape[-1]
            * video_latent.shape[-2]
            * video_latent.shape[-3]
            // self.patch_spatial**2
            // self.patch_temporal
        )

        if seq_len > self.seq_length:
            print(f"Skipping sample {sample['__key__']} because seq_len {seq_len} > self.seq_length {self.seq_length}")
            raise SkipSample()

        if self.max_frames is not None:
            video_latent = video_latent[:, : self.max_frames, :, :]

        tpcp_size = parallel_state.get_tensor_model_parallel_world_size()
        if parallel_state.get_context_parallel_world_size() > 1:
            tpcp_size *= parallel_state.get_context_parallel_world_size() * 2
        if (T * H * W) % tpcp_size != 0:
            warnings.warn(f"skipping {video_latent.shape=} not divisible by {tpcp_size=}")
            raise SkipSample()

        video_latent = rearrange(
            video_latent,
            "C (T pt) (H ph) (W pw) -> (T H W) (ph pw pt C)",
            ph=self.patch_spatial,
            pw=self.patch_spatial,
            pt=self.patch_temporal,
        )
        sample["pickle"] = sample["pickle"].cpu().float().numpy()
        if sample["pickle"].shape[0] == 1:
            sample["pickle"] = sample["pickle"][0]
        t5_text_embeddings = torch.from_numpy(sample["pickle"]).to(torch.bfloat16)
        t5_text_embeddings_seq_length = t5_text_embeddings.shape[0]

        if t5_text_embeddings_seq_length > self.text_embedding_max_length:
            t5_text_embeddings = t5_text_embeddings[: self.text_embedding_max_length]
        t5_text_mask = torch.ones(t5_text_embeddings_seq_length, dtype=torch.bfloat16)

        pos_ids = rearrange(
            pos_id_3d.get_pos_id_3d(t=T // self.patch_temporal, h=H // self.patch_spatial, w=W // self.patch_spatial),
            "T H W d -> (T H W) d",
        )

        loss_mask = torch.ones(seq_len, dtype=torch.bfloat16)
        sharding_factor = 64
        seq_len_q_padded = ((seq_len + sharding_factor - 1) // sharding_factor) * sharding_factor
        seq_len_kv_padded = (
            (t5_text_embeddings_seq_length + sharding_factor - 1) // sharding_factor
        ) * sharding_factor

        if seq_len < seq_len_q_padded:
            video_latent = F.pad(video_latent, (0, 0, 0, seq_len_q_padded - seq_len))
            loss_mask = F.pad(loss_mask, (0, seq_len_q_padded - seq_len))
            pos_ids = F.pad(pos_ids, (0, 0, 0, seq_len_q_padded - seq_len))

        if t5_text_embeddings_seq_length < seq_len_kv_padded:
            t5_text_embeddings = F.pad(
                t5_text_embeddings, (0, 0, 0, seq_len_kv_padded - t5_text_embeddings_seq_length)
            )
            t5_text_mask = F.pad(t5_text_mask, (0, seq_len_kv_padded - t5_text_embeddings_seq_length))

        return DiffusionSample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample["__subflavors__"],
            video=video_latent,
            context_embeddings=t5_text_embeddings,
            context_mask=t5_text_mask,
            loss_mask=loss_mask,
            seq_len_q=torch.tensor([seq_len], dtype=torch.int32),
            seq_len_q_padded=torch.tensor([seq_len_q_padded], dtype=torch.int32),
            seq_len_kv=torch.tensor([t5_text_embeddings_seq_length], dtype=torch.int32),
            seq_len_kv_padded=torch.tensor([seq_len_kv_padded], dtype=torch.int32),
            pos_ids=pos_ids,
            latent_shape=torch.tensor([C, T, H, W], dtype=torch.int32),
            video_metadata=info,
        )

    @stateless
    def batch(self, samples: List[DiffusionSample]) -> dict:
        """Return dictionary with data for batch."""
        if self.packing_buffer_size is None:
            # no packing
            return super().batch(samples).to_dict()

        # packing
        sample = samples[0]
        return dict(
            video=sample.video.unsqueeze_(0),
            context_embeddings=sample.context_embeddings.unsqueeze_(0),
            context_mask=sample.context_mask.unsqueeze_(0) if sample.context_mask is not None else None,
            loss_mask=sample.loss_mask.unsqueeze_(0) if sample.loss_mask is not None else None,
            seq_len_q=sample.seq_len_q,
            seq_len_q_padded=sample.seq_len_q_padded,
            seq_len_kv=sample.seq_len_kv,
            seq_len_kv_padded=sample.seq_len_kv_padded,
            pos_ids=sample.pos_ids.unsqueeze_(0) if sample.pos_ids is not None else None,
            latent_shape=sample.latent_shape,
            video_metadata=sample.video_metadata,
        )


class PosID3D:
    def __init__(self, *, max_t=32, max_h=128, max_w=128):
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.generate_pos_id()

    def generate_pos_id(self):
        self.grid = torch.stack(
            torch.meshgrid(
                torch.arange(self.max_t, device="cpu"),
                torch.arange(self.max_h, device="cpu"),
                torch.arange(self.max_w, device="cpu"),
            ),
            dim=-1,
        )

    def get_pos_id_3d(self, *, t, h, w):
        if t > self.max_t or h > self.max_h or w > self.max_w:
            self.max_t = max(self.max_t, t)
            self.max_h = max(self.max_h, h)
            self.max_w = max(self.max_w, w)
            self.generate_pos_id()
        return self.grid[:t, :h, :w]


pos_id_3d = PosID3D()
