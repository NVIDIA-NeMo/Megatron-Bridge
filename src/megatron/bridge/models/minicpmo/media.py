# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-sample multimodal insertion without modifying embedding views in place."""

from collections.abc import Sequence

import torch


def insert_media_embeddings(
    embeddings: torch.Tensor,
    features: Sequence[torch.Tensor | Sequence[torch.Tensor]],
    bounds: Sequence[torch.Tensor],
    *,
    occupied: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert media features into half-open token spans, preserving gradients.

    Args:
        embeddings: Dense token embeddings of shape ``[batch, sequence, hidden]``.
        features: Per-sample features, each a tensor or a sequence of tensors.
        bounds: Per-sample integer ``[number_of_spans, 2]`` start/end positions.
        occupied: Slots already used by another modality, if any.

    Returns:
        New embeddings and the combined occupied-slot mask.

    Raises:
        ValueError: If batch sizes, spans, feature counts or hidden sizes differ.
    """
    if embeddings.ndim != 3:
        raise ValueError("Expected [batch, sequence, hidden] embeddings.")
    batch, sequence, hidden = embeddings.shape
    if len(features) != batch or len(bounds) != batch:
        raise ValueError("Media features and bounds must preserve the token batch size.")
    if occupied is None:
        occupied = torch.zeros((batch, sequence), dtype=torch.bool, device=embeddings.device)
    elif occupied.shape != (batch, sequence) or occupied.dtype != torch.bool:
        raise ValueError("Invalid occupied-slot mask.")
    occupied = occupied.to(embeddings.device).clone()
    if batch == 0:
        return embeddings.clone(), occupied
    rows = []
    for sample in range(batch):
        spans = bounds[sample]
        if spans.ndim != 2 or spans.shape[1] != 2 or spans.dtype not in (torch.int32, torch.int64):
            raise ValueError("Media bounds must be integer [number_of_spans, 2] tensors.")
        positions = []
        previous_end = 0
        for start, end in spans.tolist():
            if start < previous_end or end <= start or end > sequence:
                raise ValueError("Media spans must be ordered, non-overlapping and inside the sequence.")
            positions.append(torch.arange(start, end, device=embeddings.device))
            previous_end = end
        indices = torch.cat(positions) if positions else torch.empty(0, device=embeddings.device, dtype=torch.long)
        value = features[sample]
        if isinstance(value, torch.Tensor):
            if value.ndim == 2:
                chunks = [value] if value.shape[0] else []
            elif value.ndim == 3:
                chunks = list(value.unbind(0))
            else:
                raise ValueError("Media features must be [tokens, hidden] or [clips, tokens, hidden].")
        else:
            chunks = list(value)
        if any(v.ndim != 2 or v.shape[-1] != hidden for v in chunks):
            raise ValueError("Invalid per-sample media feature dimensions.")
        if len(chunks) != len(spans) or any(
            chunk.shape[0] != end - start for chunk, (start, end) in zip(chunks, spans.tolist())
        ):
            raise ValueError("Each media clip must match its own token span.")
        values = torch.cat(chunks) if chunks else embeddings.new_empty((0, hidden))
        if values.shape[0] != indices.numel():
            raise ValueError("Media feature count does not match its token spans.")
        if occupied[sample, indices].any():
            raise ValueError("Multiple modalities cannot overwrite the same token slots.")
        occupied[sample, indices] = True
        rows.append(embeddings[sample].index_copy(0, indices, values.to(embeddings)))
    return torch.stack(rows), occupied
