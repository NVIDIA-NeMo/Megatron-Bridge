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

"""``pack()`` transform for Step3.7 multimodal SFT.

Takes a list of ``MultimodalSFTSample`` (the output of
:class:`Step37Flickr8kDataset.__getitem__`) and produces a single packed
dict:

  - ``tokens``      : concat of ``s.tokens[:-1]`` for each sample
  - ``labels``      : concat of ``s.tokens[1:]``
  - ``loss_masks``  : concat of ``s.loss_mask[1:]``
  - ``cu_seqlens``  : prefix-sum of sample shifted-NTP lengths
  - ``position_id`` : per-sub-seq 0..len-1 (via shared helper)
  - ``image_paths`` : flat concat of all ``s.image_paths``

A zero-padding sample is appended if the total NTP length isn't a multiple
of ``seqlen_divisible_by`` (default 64). The padding sample is included in
``cu_seqlens`` so the padded tail forms its own sub-seq.
"""

from __future__ import annotations

from typing import Any

import torch

from megatron.bridge.models.stepfun.data.flickr8k.template import MultimodalSFTSample


def get_position_id_from_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Per-sub-seq 0..L-1 position ids.

    Given cu_seqlens = [0, 209, 418, ..., total], produces a 1-D tensor
    of length ``total`` where each sub-seq segment counts 0..L-1.
    """
    cu = cu_seqlens.to(torch.long)
    total = int(cu[-1].item())
    position_id = torch.zeros(total, dtype=torch.long)
    for i in range(len(cu) - 1):
        start = int(cu[i].item())
        end = int(cu[i + 1].item())
        position_id[start:end] = torch.arange(end - start, dtype=torch.long)
    return position_id


def pack_samples(
    pieces: list[MultimodalSFTSample],
    *,
    seqlen_divisible_by: int = 64,
) -> dict[str, Any]:
    """Pack a list of samples into a single next-token-prediction batch."""
    size = sum(len(sample) for sample in pieces)
    if size % seqlen_divisible_by != 0:
        padding_size = seqlen_divisible_by - size % seqlen_divisible_by
        padding_tensor = torch.zeros(padding_size + 1, dtype=torch.long)
        pieces.append(
            MultimodalSFTSample(
                tokens=padding_tensor,
                loss_mask=torch.zeros_like(padding_tensor, dtype=torch.float32),
                image_paths=[],
            )
        )

    sizes = torch.tensor([len(sample) for sample in pieces])
    tokens = torch.cat([sample["tokens"][:-1].to(torch.long) for sample in pieces])
    labels = torch.cat([sample["tokens"][1:].to(torch.long) for sample in pieces])
    loss_masks = torch.cat([sample["loss_mask"][1:].to(torch.float32) for sample in pieces])
    image_paths = [image for sample in pieces for image in sample.get("image_paths", [])]

    cu_seqlens = torch.cat([torch.zeros(1), torch.cumsum(sizes, 0)]).int()
    packed = {
        "tokens": tokens,
        "labels": labels,
        "loss_masks": loss_masks,
        "cu_seqlens": cu_seqlens,
        "max_seq_len": sizes.max(),
        "position_id": get_position_id_from_cu_seqlens(cu_seqlens),
        "image_paths": image_paths,
    }
    return packed
