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

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch


ChunkSelectionStrategy = Literal["random", "first-valid"]


@dataclass(frozen=True)
class LongLiveChunk:
    """Token span selected for one packed WAN sample."""

    sample_index: int
    sample_start: int
    valid_end: int
    padded_end: int
    grid_size: Tuple[int, int, int]
    target_frame_start: int
    target_num_frames: int

    @property
    def tokens_per_frame(self) -> int:
        return self.grid_size[1] * self.grid_size[2]

    @property
    def target_start(self) -> int:
        return self.sample_start + self.target_frame_start * self.tokens_per_frame

    @property
    def target_end(self) -> int:
        return self.target_start + self.target_num_frames * self.tokens_per_frame


@dataclass(frozen=True)
class LongLivePairedChunk:
    """Clean/noisy token pair for one temporal chunk in the LongLive teacher-forcing layout."""

    sample_index: int
    original_start: int
    original_end: int
    original_padded_start: int
    original_padded_end: int
    frame_start: int
    frame_count: int
    grid_size: Tuple[int, int, int]
    clean_start: int
    noisy_start: int

    @property
    def tokens_per_frame(self) -> int:
        return self.grid_size[1] * self.grid_size[2]

    @property
    def token_count(self) -> int:
        return self.frame_count * self.tokens_per_frame

    @property
    def clean_end(self) -> int:
        return self.clean_start + self.token_count

    @property
    def noisy_end(self) -> int:
        return self.noisy_start + self.token_count


def temporal_frame_token_spans(
    grid_size: torch.Tensor | Tuple[int, int, int],
    sample_start: int = 0,
    valid_seq_len: Optional[int] = None,
) -> list[tuple[int, int]]:
    """Map a WAN `(F, H, W)` patch grid to per-temporal-frame token spans."""
    f, h, w = _as_grid_tuple(grid_size)
    tokens_per_frame = h * w
    max_frames = f
    if valid_seq_len is not None:
        max_frames = min(max_frames, valid_seq_len // tokens_per_frame)
    return [
        (sample_start + frame * tokens_per_frame, sample_start + (frame + 1) * tokens_per_frame)
        for frame in range(max_frames)
    ]


def select_longlive_chunks(
    grid_sizes: torch.Tensor,
    seq_len_q: torch.Tensor,
    seq_len_q_padded: torch.Tensor,
    target_chunk_frames: int = 1,
    strategy: ChunkSelectionStrategy = "random",
    generator: Optional[torch.Generator] = None,
) -> list[Optional[LongLiveChunk]]:
    """Select one target temporal chunk per packed sample."""
    if target_chunk_frames < 1:
        raise ValueError("target_chunk_frames must be >= 1")
    if strategy not in ("random", "first-valid"):
        raise ValueError("strategy must be 'random' or 'first-valid'")

    chunks: list[Optional[LongLiveChunk]] = []
    sample_start = 0
    for sample_idx, grid_size_tensor in enumerate(grid_sizes):
        grid_size = _as_grid_tuple(grid_size_tensor)
        valid_len = int(seq_len_q[sample_idx].item())
        padded_len = int(seq_len_q_padded[sample_idx].item())
        f, h, w = grid_size
        tokens_per_frame = h * w
        valid_frames = min(f, valid_len // tokens_per_frame)
        max_start_frame = valid_frames - target_chunk_frames

        if valid_frames < target_chunk_frames + 1 or max_start_frame < 1:
            chunks.append(None)
            sample_start += padded_len
            continue

        if strategy == "first-valid":
            target_frame_start = 1
        else:
            target_frame_start = int(
                torch.randint(
                    low=1,
                    high=max_start_frame + 1,
                    size=(1,),
                    generator=generator,
                    device=grid_size_tensor.device,
                ).item()
            )

        chunks.append(
            LongLiveChunk(
                sample_index=sample_idx,
                sample_start=sample_start,
                valid_end=sample_start + valid_len,
                padded_end=sample_start + padded_len,
                grid_size=grid_size,
                target_frame_start=target_frame_start,
                target_num_frames=target_chunk_frames,
            )
        )
        sample_start += padded_len

    return chunks


def select_longlive_paired_chunks(
    grid_sizes: torch.Tensor,
    seq_len_q: torch.Tensor,
    seq_len_q_padded: torch.Tensor,
    target_chunk_frames: int = 1,
) -> list[LongLivePairedChunk]:
    """Build natural clean/noisy paired chunks for all valid temporal chunks.

    The resulting sequence order is `[chunk0_clean, chunk0_noisy, chunk1_clean,
    chunk1_noisy, ...]`, matching the LongLive-2.0 natural Balanced-SP layout.
    """
    if target_chunk_frames < 1:
        raise ValueError("target_chunk_frames must be >= 1")

    chunks: list[LongLivePairedChunk] = []
    original_sample_start = 0
    paired_cursor = 0
    for sample_idx, grid_size_tensor in enumerate(grid_sizes):
        grid_size = _as_grid_tuple(grid_size_tensor)
        valid_len = int(seq_len_q[sample_idx].item())
        padded_len = int(seq_len_q_padded[sample_idx].item())
        f, h, w = grid_size
        tokens_per_frame = h * w
        valid_frames = min(f, valid_len // tokens_per_frame)

        for frame_start in range(0, valid_frames, target_chunk_frames):
            frame_count = min(target_chunk_frames, valid_frames - frame_start)
            original_start = original_sample_start + frame_start * tokens_per_frame
            original_end = original_start + frame_count * tokens_per_frame
            clean_start = paired_cursor
            noisy_start = clean_start + frame_count * tokens_per_frame
            chunks.append(
                LongLivePairedChunk(
                    sample_index=sample_idx,
                    original_start=original_start,
                    original_end=original_end,
                    original_padded_start=original_sample_start,
                    original_padded_end=original_sample_start + padded_len,
                    frame_start=frame_start,
                    frame_count=frame_count,
                    grid_size=grid_size,
                    clean_start=clean_start,
                    noisy_start=noisy_start,
                )
            )
            paired_cursor = noisy_start + frame_count * tokens_per_frame

        original_sample_start += padded_len

    return chunks


def build_longlive_paired_latents_and_masks(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    base_loss_mask: torch.Tensor,
    chunks: list[LongLivePairedChunk],
    noise_schedule,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return paired clean/noisy latents, velocity target, and noisy-only loss mask."""
    noisy_latents = noise_schedule.forward(clean_latents.float(), noise, sigma)
    target = noise - clean_latents.float()

    paired_latents = []
    paired_target = []
    paired_loss_mask = []
    for chunk in chunks:
        clean_slice = slice(chunk.original_start, chunk.original_end)
        paired_latents.append(clean_latents[:, clean_slice].float())
        paired_target.append(torch.zeros_like(target[:, clean_slice]))
        paired_loss_mask.append(torch.zeros_like(base_loss_mask[clean_slice]))

        paired_latents.append(noisy_latents[:, clean_slice])
        paired_target.append(target[:, clean_slice])
        paired_loss_mask.append(base_loss_mask[clean_slice])

    if not paired_latents:
        raise ValueError("LongLive paired teacher forcing requires at least one valid temporal chunk")

    return (
        torch.cat(paired_latents, dim=1),
        torch.cat(paired_target, dim=1),
        torch.cat(paired_loss_mask, dim=0),
    )


def build_longlive_paired_sequence_metadata(
    chunks: list[LongLivePairedChunk],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return chunk-level grid sizes, frame offsets, seq lengths, and padded seq lengths.

    The resulting sequence metadata is only safe for LongLive's dense-mask SBHD
    path. THD attention treats every `cu_seqlens` segment as a hard attention
    boundary, so `LongLiveWanFlowMatchingPipeline` rejects THD before using
    these chunk-level lengths for self-attention.
    """
    grid_sizes = []
    frame_offsets = []
    seq_lens = []
    for chunk in chunks:
        chunk_grid_size = (chunk.frame_count, chunk.grid_size[1], chunk.grid_size[2])
        chunk_len = chunk.token_count
        for _stream_idx in range(2):
            grid_sizes.append(chunk_grid_size)
            frame_offsets.append(chunk.frame_start)
            seq_lens.append(chunk_len)

    return (
        torch.tensor(grid_sizes, dtype=torch.int32, device=device),
        torch.tensor(frame_offsets, dtype=torch.int32, device=device),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
    )


def duplicate_context_for_longlive_paired_chunks(
    context_embeddings: torch.Tensor,
    seq_len_kv: torch.Tensor,
    seq_len_kv_padded: torch.Tensor,
    chunks: list[LongLivePairedChunk],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Duplicate each sample's text context for every clean/noisy paired chunk."""
    context_starts = [0]
    for length in seq_len_kv_padded[:-1]:
        context_starts.append(context_starts[-1] + int(length.item()))

    paired_context = []
    paired_seq_len_kv = []
    paired_seq_len_kv_padded = []
    for chunk in chunks:
        context_start = context_starts[chunk.sample_index]
        context_len = int(seq_len_kv[chunk.sample_index].item())
        context_padded_len = int(seq_len_kv_padded[chunk.sample_index].item())
        context_slice = slice(context_start, context_start + context_padded_len)
        for _stream_idx in range(2):
            paired_context.append(context_embeddings[context_slice])
            paired_seq_len_kv.append(context_len)
            paired_seq_len_kv_padded.append(context_padded_len)

    return (
        torch.cat(paired_context, dim=0),
        torch.tensor(paired_seq_len_kv, dtype=seq_len_kv.dtype, device=seq_len_kv.device),
        torch.tensor(paired_seq_len_kv_padded, dtype=seq_len_kv_padded.dtype, device=seq_len_kv_padded.device),
    )


def build_longlive_loss_mask(base_loss_mask: torch.Tensor, chunks: list[LongLiveChunk]) -> torch.Tensor:
    """Return a loss mask that only enables the selected target chunk tokens."""
    longlive_mask = torch.zeros_like(base_loss_mask)
    for chunk in chunks:
        longlive_mask[chunk.target_start : chunk.target_end] = base_loss_mask[chunk.target_start : chunk.target_end]
    return longlive_mask


def apply_longlive_noising(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    chunks: list[LongLiveChunk],
    noise_schedule,
) -> torch.Tensor:
    """Keep non-target tokens clean and replace target tokens with flow-matching noisy latents."""
    noisy_latents = noise_schedule.forward(clean_latents.float(), noise, sigma)
    mixed_latents = clean_latents.float().clone()
    for chunk in chunks:
        mixed_latents[:, chunk.target_start : chunk.target_end] = noisy_latents[
            :, chunk.target_start : chunk.target_end
        ]
    return mixed_latents


def build_teacher_forcing_self_attention_mask(
    total_seq_len: int,
    chunks: list[LongLiveChunk],
    device: torch.device,
) -> torch.Tensor:
    """Build a bool self-attention mask for LongLive teacher forcing."""
    mask = torch.ones(total_seq_len, total_seq_len, dtype=torch.bool, device=device)

    for chunk in chunks:
        history = slice(chunk.sample_start, chunk.target_start)
        target = slice(chunk.target_start, chunk.target_end)
        valid = slice(chunk.sample_start, chunk.valid_end)

        mask[history, history] = False
        mask[target, slice(chunk.sample_start, chunk.target_end)] = False
        mask[slice(chunk.target_end, chunk.valid_end), valid] = False

    return mask.unsqueeze(0).unsqueeze(0)


def build_paired_teacher_forcing_self_attention_mask(
    total_seq_len: int,
    chunks: list[LongLivePairedChunk],
    device: torch.device,
    local_attn_chunk_window: int | None = None,
) -> torch.Tensor:
    """Build a dense bool mask for LongLive paired clean/noisy teacher forcing.

    `False` means visible and `True` means masked, matching Transformer Engine
    padding mask semantics.
    """
    mask = torch.ones(total_seq_len, total_seq_len, dtype=torch.bool, device=device)
    if local_attn_chunk_window is not None and local_attn_chunk_window < 1:
        raise ValueError("local_attn_chunk_window must be >= 1 when provided")

    for query_idx, query_chunk in enumerate(chunks):
        clean_q = slice(query_chunk.clean_start, query_chunk.clean_end)
        noisy_q = slice(query_chunk.noisy_start, query_chunk.noisy_end)
        same_sample_previous = [
            (key_idx, key_chunk)
            for key_idx, key_chunk in enumerate(chunks[: query_idx + 1])
            if key_chunk.sample_index == query_chunk.sample_index
        ]
        if local_attn_chunk_window is None:
            clean_visible_chunks = [key_chunk for _key_idx, key_chunk in same_sample_previous]
            noisy_visible_chunks = [key_chunk for key_idx, key_chunk in same_sample_previous if key_idx < query_idx]
        else:
            clean_visible_chunks = [
                key_chunk for _key_idx, key_chunk in same_sample_previous[-local_attn_chunk_window:]
            ]
            noisy_visible_chunks = [
                key_chunk
                for key_idx, key_chunk in same_sample_previous[-local_attn_chunk_window:]
                if key_idx < query_idx
            ]

        for key_chunk in clean_visible_chunks:
            clean_k = slice(key_chunk.clean_start, key_chunk.clean_end)
            mask[clean_q, clean_k] = False

        for key_chunk in noisy_visible_chunks:
            clean_k = slice(key_chunk.clean_start, key_chunk.clean_end)
            mask[noisy_q, clean_k] = False

        same_noisy_k = slice(query_chunk.noisy_start, query_chunk.noisy_end)
        mask[noisy_q, same_noisy_k] = False

    return mask.unsqueeze(0).unsqueeze(0)


def split_self_attention_mask_rows(mask: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    """Select local query rows from a `[1, 1, S_q, S_k]` attention mask."""
    return mask.index_select(dim=-2, index=row_indices.to(device=mask.device, dtype=torch.long)).contiguous()


def _as_grid_tuple(grid_size: torch.Tensor | Tuple[int, int, int]) -> Tuple[int, int, int]:
    if torch.is_tensor(grid_size):
        return tuple(int(x) for x in grid_size.detach().cpu().tolist())
    return tuple(int(x) for x in grid_size)
