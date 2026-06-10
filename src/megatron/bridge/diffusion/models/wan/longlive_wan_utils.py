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


def split_self_attention_mask_rows(mask: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    """Select local query rows from a `[1, 1, S_q, S_k]` attention mask."""
    return mask.index_select(dim=-2, index=row_indices.to(device=mask.device, dtype=torch.long)).contiguous()


def _as_grid_tuple(grid_size: torch.Tensor | Tuple[int, int, int]) -> Tuple[int, int, int]:
    if torch.is_tensor(grid_size):
        return tuple(int(x) for x in grid_size.detach().cpu().tolist())
    return tuple(int(x) for x in grid_size)
