# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Convert validated BAGEL packed batches to PR #3635 MIMO inputs."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.attention.flex_attention import create_block_mask

from megatron.bridge.models.bagel.diffusion import BagelDiffusionScheduler


def _attention_metadata(
    nested_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten official masks without allocating a full packed S-by-S matrix."""
    lengths = torch.tensor([mask.shape[0] for mask in nested_masks])
    sample_ids = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    starts = torch.repeat_interleave(
        torch.cumsum(torch.nn.functional.pad(lengths, (1, 0)), dim=0)[:-1],
        lengths,
    )
    offsets = torch.repeat_interleave(
        torch.nn.functional.pad(torch.cumsum(lengths.square(), dim=0), (1, 0))[:-1],
        lengths,
    )
    lengths = torch.repeat_interleave(lengths, lengths)
    allowed = torch.cat([mask.eq(0).flatten() for mask in nested_masks])
    return allowed, sample_ids, starts, offsets, lengths


def _block_mask(nested_masks: list[torch.Tensor], num_heads: int) -> Any:
    """Build FlexAttention metadata from official BAGEL dense masks."""
    allowed, sample_ids, starts, offsets, lengths = [
        tensor.cuda(non_blocking=True) for tensor in _attention_metadata(nested_masks)
    ]

    def mask_mod(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        start = starts[query]
        length = lengths[query]
        local_key = torch.minimum(torch.maximum(key - start, torch.zeros_like(key)), length - 1)
        index = offsets[query] + (query - start) * length + local_key
        return (sample_ids[query] == sample_ids[key]) & allowed[index]

    sequence_length = len(sample_ids)
    return create_block_mask(
        mask_mod,
        B=1,
        H=num_heads,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=allowed.device,
        BLOCK_SIZE=128,
        _compile=True,
    )


def bagel_packed_batch_to_mimo(
    packed_batch: dict[str, object],
    scheduler: BagelDiffusionScheduler,
    *,
    num_attention_heads: int,
) -> dict[str, object]:
    """Create the CP=1 MIMO batch consumed by MCore BAGEL."""
    from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams

    sequence_length = packed_batch["sequence_length"]
    text_ids = packed_batch["packed_text_ids"].cuda(non_blocking=True)
    text_indexes = packed_batch["packed_text_indexes"].cuda(non_blocking=True)
    vit_indexes = packed_batch.get("packed_vit_token_indexes")
    if vit_indexes is not None:
        vit_indexes = vit_indexes.cuda(non_blocking=True)
        und_indexes = torch.cat((text_indexes, vit_indexes))
    else:
        und_indexes = text_indexes
    vae_indexes = packed_batch["packed_vae_token_indexes"].cuda(non_blocking=True)

    labels_full = torch.full((sequence_length,), -100, dtype=torch.long, device="cuda")
    loss_mask_full = torch.zeros(sequence_length, dtype=torch.float, device="cuda")
    if "ce_loss_indexes" in packed_batch:
        ce_indexes = packed_batch["ce_loss_indexes"].cuda(non_blocking=True)
        labels_full[ce_indexes] = packed_batch["packed_label_ids"].cuda(non_blocking=True)
        loss_mask_full[ce_indexes] = packed_batch["ce_loss_weights"].cuda(non_blocking=True)

    latents, shifted_timesteps, target = scheduler.prepare(packed_batch)
    mse_indexes = packed_batch["mse_loss_indexes"].cuda(non_blocking=True)
    gen_loss_mask = torch.isin(vae_indexes, mse_indexes).float()
    dense_target = torch.zeros(
        len(vae_indexes),
        target.shape[-1],
        dtype=target.dtype,
        device=target.device,
    )
    dense_target[gen_loss_mask.bool()] = target
    packed_positions = packed_batch["packed_position_ids"].cuda(non_blocking=True)

    modality_inputs: dict[str, object] = {
        "diffusion": {
            "latents": latents,
            "shifted_timesteps": shifted_timesteps,
            "latent_position_ids": packed_batch["packed_latent_position_ids"].cuda(non_blocking=True),
        }
    }
    if "packed_vit_tokens" in packed_batch:
        modality_inputs["images"] = {
            "vision_encoder": {
                "packed_vit_tokens": packed_batch["packed_vit_tokens"].cuda(non_blocking=True),
                "vit_token_seqlens": packed_batch["vit_token_seqlens"].cuda(non_blocking=True),
                "packed_vit_position_ids": packed_batch["packed_vit_position_ids"].cuda(non_blocking=True),
            }
        }

    packed_seq_params = MoTPackedSeqParams(
        packed_text_indexes=text_indexes,
        packed_vit_token_indexes=vit_indexes,
        packed_vae_token_indexes=vae_indexes,
        packed_und_token_indexes=und_indexes,
        packed_gen_token_indexes=vae_indexes,
        local_und_token_indexes=und_indexes,
        local_gen_token_indexes=vae_indexes,
        padded_und_seqlen=len(und_indexes),
        padded_gen_seqlen=len(vae_indexes),
        vit_tokens_encoded_per_cp=None,
    )
    return {
        "input_ids": text_ids.unsqueeze(0),
        "position_ids": torch.arange(len(text_ids), device="cuda").unsqueeze(0),
        "attention_mask": _block_mask(packed_batch["nested_attention_masks"], num_attention_heads),
        "labels": labels_full[und_indexes],
        "loss_mask": loss_mask_full[und_indexes],
        "modality_inputs": modality_inputs,
        "sample_lens": packed_batch["sample_lens"],
        "packed_position_ids": torch.cat((packed_positions[und_indexes], packed_positions[vae_indexes])),
        "sequence_length": sequence_length,
        "vis_gen_target": dense_target,
        "gen_loss_mask": gen_loss_mask,
        "packed_seq_params": packed_seq_params,
    }
