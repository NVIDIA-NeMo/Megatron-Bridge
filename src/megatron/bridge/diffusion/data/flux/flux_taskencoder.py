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
from megatron.bridge.diffusion.data.common.diffusion_task_encoder_with_sp import (
    DiffusionTaskEncoderWithSequencePacking,
)


def cook(sample: dict) -> dict:
    """
    Processes a raw sample dictionary from energon dataset and returns a new dictionary with specific keys.

    Args:
        sample (dict): The input dictionary containing the raw sample data.

    Returns:
        dict: A new dictionary containing the processed sample data with the following keys:
            - All keys from the result of `basic_sample_keys(sample)`
            - 'json': The contains meta data like resolution, etc.
            - 'pth': contains image latent tensor
            - 'pickle': contains text embeddings (T5 and CLIP pooled)
    """
    return dict(
        **basic_sample_keys(sample),
        json=sample["json"],
        pth=sample["pth"],
        pickle=sample["pickle"],
    )


class FluxTaskEncoder(DiffusionTaskEncoderWithSequencePacking):
    """
    Task encoder for Flux dataset.
    Attributes:
        cookers (list): A list of Cooker objects used for processing.
        vae_scale_factor (int): The VAE downsampling factor. Defaults to 8.
        seq_length (int): The sequence length. Defaults to 1024.
        latent_channels (int): Number of latent channels from VAE. Defaults to 16.
    """

    cookers = [
        Cooker(cook),
    ]

    def __init__(
        self,
        *args,
        vae_scale_factor: int = 8,
        seq_length: int = 1024,
        latent_channels: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vae_scale_factor = vae_scale_factor
        self.seq_length = seq_length
        self.latent_channels = latent_channels

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: dict) -> dict:
        image_latent = sample["pth"]
        text_embeddings = sample["pickle"]
        image_metadata = sample["json"]

        # sanity quality check
        if torch.isnan(image_latent).any() or torch.isinf(image_latent).any():
            raise SkipSample()
        if torch.max(torch.abs(image_latent)) > 1e3:
            raise SkipSample()

        # image_latent shape: [C, H, W]
        # Keep latents unpacked - flux_step will pack them during forward pass
        C, H, W = image_latent.shape

        # Extract T5 embeddings and CLIP pooled embeddings
        # text_embeddings is expected to be a dict with keys:
        # - 'prompt_embeds': T5 embeddings [text_seq_len, context_dim]
        # - 'pooled_prompt_embeds': CLIP pooled embeddings [pooled_dim]
        if isinstance(text_embeddings, dict):
            prompt_embeds = text_embeddings.get("prompt_embeds", text_embeddings.get("t5_embeds"))
            pooled_prompt_embeds = text_embeddings.get("pooled_prompt_embeds", text_embeddings.get("clip_embeds"))

            # Ensure pooled_prompt_embeds is not None
            if pooled_prompt_embeds is None:
                pooled_prompt_embeds = torch.zeros(768, dtype=torch.bfloat16)
        else:
            # If it's a single tensor, assume it's T5 embeddings
            prompt_embeds = text_embeddings
            pooled_prompt_embeds = torch.zeros(768, dtype=torch.bfloat16)  # Default CLIP dim

        # pad text embeddings to fixed length
        text_max_len = 512
        if prompt_embeds.shape[0] < text_max_len:
            prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, text_max_len - prompt_embeds.shape[0]))
        else:
            prompt_embeds = prompt_embeds[:text_max_len]

        # calculate sequence lengths
        # For flux, seq_len_q is the number of patches after packing: (H/2)*(W/2)
        seq_len_q = (H // 2) * (W // 2)
        seq_len_kv = prompt_embeds.shape[0]  # text_seq_len

        # loss mask - covers all latent positions
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
            # Note: For unpacked latents [C, H, W], we need to pad H and W dimensions
            # But since we're padding sequence length, we pad the loss_mask only
            # The latent padding will be handled during packing in flux_step
            loss_mask = F.pad(loss_mask, (0, seq_len_q_padded - seq_len_q))
        if seq_len_kv < seq_len_kv_padded:
            prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, seq_len_kv_padded - seq_len_kv))

        ### Note: shape of sample's values
        # image_latent: [C, H, W] - unpacked format
        # latent_shape: [H, W]
        # prompt_embeds: [text_seq_len, text_embedding_dim]
        # pooled_prompt_embeds: [pooled_dim]
        # text_ids: [text_seq_len, 3]

        # Prepare text IDs for position encoding
        text_ids = torch.zeros(prompt_embeds.shape[0], 3, dtype=torch.bfloat16)
        text_ids[:, 0] = torch.arange(prompt_embeds.shape[0], dtype=torch.bfloat16)

        # Store pooled embeddings and text_ids in metadata
        metadata = {
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
            "original_metadata": image_metadata,
        }

        return DiffusionSample(
            __key__=sample["__key__"],
            __restore_key__=sample["__restore_key__"],
            __subflavor__=None,
            __subflavors__=sample["__subflavors__"],
            video=image_latent,  # Store unpacked latents [C, H, W]
            context_embeddings=prompt_embeds,
            latent_shape=torch.tensor([H, W], dtype=torch.int32),
            loss_mask=loss_mask,
            seq_len_q=torch.tensor([seq_len_q], dtype=torch.int32),
            seq_len_q_padded=torch.tensor([seq_len_q_padded], dtype=torch.int32),
            seq_len_kv=torch.tensor([seq_len_kv], dtype=torch.int32),
            seq_len_kv_padded=torch.tensor([seq_len_kv_padded], dtype=torch.int32),
            pos_ids=torch.zeros(1, dtype=torch.bfloat16),  # dummy pos_ids
            video_metadata=metadata,
        )

    # NOTE:
    # the method select_samples_to_pack() and pack_selected_samples() are inherited from the parent
    #   class DiffusionTaskEncoderWithSequencePacking

    @stateless
    def batch(self, samples: List[DiffusionSample]) -> dict:
        """Return dictionary with data for batch."""

        # Helper function to extract metadata
        def extract_metadata(sample):
            # Handle case where video_metadata is a list (from packed samples)
            metadata = sample.video_metadata
            if isinstance(metadata, list):
                metadata = metadata[0] if len(metadata) > 0 else {}

            if isinstance(metadata, dict) and "pooled_prompt_embeds" in metadata:
                pooled = metadata["pooled_prompt_embeds"]
                text_ids = metadata.get("text_ids", torch.zeros(512, 3, dtype=torch.bfloat16))
                orig_metadata = metadata.get("original_metadata", metadata)

                return (pooled, text_ids, orig_metadata)
            else:
                raise ValueError("Expected 'pooled_prompt_embeds' in metadata.")

        if self.packing_buffer_size is None:
            # No packing - batch multiple samples
            latents_list = []
            prompt_embeds_list = []
            pooled_embeds_list = []
            text_ids_list = []
            loss_mask_list = []
            seq_len_q_list = []
            seq_len_q_padded_list = []
            seq_len_kv_list = []
            seq_len_kv_padded_list = []
            latent_shape_list = []
            metadata_list = []

            for sample in samples:
                pooled, text_ids, metadata = extract_metadata(sample)

                latents_list.append(sample.video)
                prompt_embeds_list.append(sample.context_embeddings)
                pooled_embeds_list.append(pooled)
                text_ids_list.append(text_ids)
                loss_mask_list.append(sample.loss_mask if sample.loss_mask is not None else torch.ones(1))
                seq_len_q_list.append(sample.seq_len_q)
                seq_len_q_padded_list.append(sample.seq_len_q_padded)
                seq_len_kv_list.append(sample.seq_len_kv)
                seq_len_kv_padded_list.append(sample.seq_len_kv_padded)
                latent_shape_list.append(sample.latent_shape)
                metadata_list.append(metadata)

            return dict(
                latents=torch.stack(latents_list),
                prompt_embeds=torch.stack(prompt_embeds_list),
                pooled_prompt_embeds=torch.stack(pooled_embeds_list),
                text_ids=torch.stack(text_ids_list),
                loss_mask=torch.stack(loss_mask_list),
                seq_len_q=torch.cat(seq_len_q_list),
                seq_len_q_padded=torch.cat(seq_len_q_padded_list),
                seq_len_kv=torch.cat(seq_len_kv_list),
                seq_len_kv_padded=torch.cat(seq_len_kv_padded_list),
                latent_shape=torch.stack(latent_shape_list),
                image_metadata=metadata_list,
            )

        # Packing case - single packed sample
        sample = samples[0]
        pooled_prompt_embeds, text_ids, image_metadata = extract_metadata(sample)

        # Stack to create batch dimension
        # sample.video has shape [C, H, W] -> unsqueeze to [1, C, H, W] for batch
        latents = sample.video.unsqueeze(0)  # [1, C, H, W]

        # Prompt embeds: [text_seq_len, D] -> [1, text_seq_len, D] for batch
        prompt_embeds = sample.context_embeddings.unsqueeze(0)  # [1, text_seq_len, D]

        # Pooled embeds: [pooled_dim] -> [1, pooled_dim] for batch
        pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)  # [1, pooled_dim]

        # Text IDs: [text_seq_len, 3] -> [1, text_seq_len, 3] for batch
        text_ids = text_ids.unsqueeze(0)  # [1, text_seq_len, 3]

        # Loss mask: [seq_len_q] -> [1, seq_len_q] for batch
        loss_mask = sample.loss_mask.unsqueeze(0) if sample.loss_mask is not None else None

        batch = dict(
            latents=latents,  # [1, C, H, W] - unpacked format
            prompt_embeds=prompt_embeds,  # [1, text_seq_len, D]
            pooled_prompt_embeds=pooled_prompt_embeds,  # [1, pooled_dim]
            text_ids=text_ids,  # [1, text_seq_len, 3]
            loss_mask=loss_mask,  # [1, seq_len_q]
            seq_len_q=sample.seq_len_q,
            seq_len_q_padded=sample.seq_len_q_padded,
            seq_len_kv=sample.seq_len_kv,
            seq_len_kv_padded=sample.seq_len_kv_padded,
            latent_shape=sample.latent_shape,  # [H, W]
            image_metadata=image_metadata,
        )

        ### Note: shape of batch's values (with packing_buffer_size, batch size is 1)
        # latents: [1, C, H, W] - unpacked format
        # prompt_embeds: [1, text_seq_len, D]
        # pooled_prompt_embeds: [1, pooled_dim]
        # text_ids: [1, text_seq_len, 3]
        # loss_mask: [1, seq_len_q] where seq_len_q = (H/2)*(W/2)
        # seq_len_q: [num_samples]
        # seq_len_q_padded: [num_samples]
        # seq_len_kv: [num_samples]
        # seq_len_kv_padded: [num_samples]
        # latent_shape: [num_samples, 2]
        # image_metadata: [num_samples]

        return batch
