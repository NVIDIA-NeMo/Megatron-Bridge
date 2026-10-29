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

from dataclasses import dataclass
from typing import Any, Optional

import torch
from megatron.energon import Sample


@dataclass
class DiffusionSample(Sample):
    """
    Data class representing a sample for diffusion tasks.

    Attributes:
        video (torch.Tensor): Video latents (C T H W).
        t5_text_embeddings (torch.Tensor): Text embeddings (S D).
        t5_text_mask (torch.Tensor): Mask for text embeddings.
        loss_mask (torch.Tensor): Mask indicating valid positions for loss computation.
        image_size (Optional[torch.Tensor]): Tensor containing image dimensions.
        fps (Optional[torch.Tensor]): Frame rate of the video.
        num_frames (Optional[torch.Tensor]): Number of frames in the video.
        padding_mask (Optional[torch.Tensor]): Mask indicating padding positions.
        seq_len_q (Optional[torch.Tensor]): Sequence length for query embeddings.
        seq_len_q_padded (Optional[torch.Tensor]): Sequence length for query embeddings after padding.
        seq_len_kv (Optional[torch.Tensor]): Sequence length for key/value embeddings.
        pos_ids (Optional[torch.Tensor]): Positional IDs.
        latent_shape (Optional[torch.Tensor]): Shape of the latent tensor.
        video_metadata (Optional[dict]): Metadata of the video.
    """

    video: torch.Tensor  # video latents (C T H W)
    context_embeddings: torch.Tensor  # (S D)
    context_mask: torch.Tensor = None  # 1
    image_size: Optional[torch.Tensor] = None
    loss_mask: torch.Tensor = None
    fps: Optional[torch.Tensor] = None
    num_frames: Optional[torch.Tensor] = None
    padding_mask: Optional[torch.Tensor] = None
    seq_len_q: Optional[torch.Tensor] = None
    seq_len_q_padded: Optional[torch.Tensor] = None
    seq_len_kv: Optional[torch.Tensor] = None
    seq_len_kv_padded: Optional[torch.Tensor] = None
    pos_ids: Optional[torch.Tensor] = None
    latent_shape: Optional[torch.Tensor] = None
    video_metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        """Converts the sample to a dictionary."""
        return dict(
            video=self.video,
            context_embeddings=self.context_embeddings,
            context_mask=self.context_mask,
            loss_mask=self.loss_mask,
            image_size=self.image_size,
            fps=self.fps,
            num_frames=self.num_frames,
            padding_mask=self.padding_mask,
            seq_len_q=self.seq_len_q,
            seq_len_q_padded=self.seq_len_q_padded,
            seq_len_kv=self.seq_len_kv,
            seq_len_kv_padded=self.seq_len_kv_padded,
            pos_ids=self.pos_ids,
            latent_shape=self.latent_shape,
            video_metadata=self.video_metadata,
        )

    def __add__(self, other: Any) -> int:
        """Adds the sequence length of this sample with another sample or integer."""
        if isinstance(other, DiffusionSample):
            # Use padded length if available (for CP), otherwise use unpadded
            self_len = self.seq_len_q_padded.item() if self.seq_len_q_padded is not None else self.seq_len_q.item()
            other_len = other.seq_len_q_padded.item() if other.seq_len_q_padded is not None else other.seq_len_q.item()
            return self_len + other_len
        elif isinstance(other, int):
            # Use padded length if available (for CP), otherwise use unpadded
            self_len = self.seq_len_q_padded.item() if self.seq_len_q_padded is not None else self.seq_len_q.item()
            return self_len + other
        raise NotImplementedError

    def __radd__(self, other: Any) -> int:
        """Handles reverse addition for summing with integers."""
        # This is called if sum or other operations start with a non-DiffusionSample object.
        # e.g., sum([DiffusionSample(1), DiffusionSample(2)]) -> the 0 + DiffusionSample(1) calls __radd__.
        if isinstance(other, int):
            # Use padded length if available (for CP), otherwise use unpadded
            self_len = self.seq_len_q_padded.item() if self.seq_len_q_padded is not None else self.seq_len_q.item()
            return self_len + other
        raise NotImplementedError

    def __lt__(self, other: Any) -> bool:
        """Compares this sample's sequence length with another sample or integer."""
        if isinstance(other, DiffusionSample):
            # Use padded length if available (for CP), otherwise use unpadded
            self_len = self.seq_len_q_padded.item() if self.seq_len_q_padded is not None else self.seq_len_q.item()
            other_len = other.seq_len_q_padded.item() if other.seq_len_q_padded is not None else other.seq_len_q.item()
            return self_len < other_len
        elif isinstance(other, int):
            # Use padded length if available (for CP), otherwise use unpadded
            self_len = self.seq_len_q_padded.item() if self.seq_len_q_padded is not None else self.seq_len_q.item()
            return self_len < other
        raise NotImplementedError
