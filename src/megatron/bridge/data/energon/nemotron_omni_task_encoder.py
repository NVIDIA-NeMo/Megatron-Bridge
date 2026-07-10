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

"""Nemotron Omni Energon adapter backed by the shared HF-style collator."""

from typing import Any, Sequence

from megatron.bridge.data.energon.hf_task_encoder import HFEnergonBatch, HFEnergonSample, HFTaskEncoder
from megatron.bridge.models.nemotron_omni.data.collate_fn import nemotron_omni_collate_fn


# Compatibility aliases for callers that imported the old model-specific
# sample and batch names. Their representation is now the generic HF-style
# Energon contract shared by all processor-backed VLMs.
NemotronOmniTaskSample = HFEnergonSample
NemotronOmniTaskBatch = HFEnergonBatch


class NemotronOmniTaskEncoder(HFTaskEncoder):
    """Normalize Energon samples and delegate all model processing to one collator.

    The task encoder owns only source adaptation and configuration. Tokenization,
    assistant masking, modality-token expansion, padding, and in-batch packing
    are performed by :func:`nemotron_omni_collate_fn` for both Direct-HF and
    Energon datasets.
    """

    def __init__(
        self,
        processor: Any,
        seq_length: int = 4096,
        max_audio_duration: float = 30.0,
        num_mel_bins: int = 128,
        visual_keys: Sequence[str] = ("pixel_values",),
        temporal_patch_size: int = 2,
        video_fps: float = 1.0,
        video_nframes: int = 8,
        use_temporal_video_embedder: bool = False,
        patch_dim: int = 16,
        pad_to_max_length: bool = False,
        pad_to_multiple_of: int = 128,
        enable_in_batch_packing: bool = False,
        in_batch_packing_pad_to_multiple_of: int = 1,
    ) -> None:
        super().__init__(
            processor=processor,
            seq_length=seq_length,
            visual_keys=visual_keys,
            collate_fn=nemotron_omni_collate_fn,
            pad_to_max_length=pad_to_max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            enable_in_batch_packing=enable_in_batch_packing,
            in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        )
        self.max_audio_duration = max_audio_duration
        self.num_mel_bins = num_mel_bins
        self.temporal_patch_size = temporal_patch_size
        self.video_fps = video_fps
        self.video_nframes = video_nframes
        self.use_temporal_video_embedder = use_temporal_video_embedder
        self.patch_dim = patch_dim

    def collate_fn(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate normalized Energon examples with the shared Omni path."""
        return self._collate_impl(
            examples,
            self.processor,
            visual_keys=self.visual_keys,
            sequence_length=self.seq_length,
            pad_to_max_length=self.pad_to_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            enable_in_batch_packing=self.enable_in_batch_packing,
            in_batch_packing_pad_to_multiple_of=self.in_batch_packing_pad_to_multiple_of,
            max_audio_duration=self.max_audio_duration,
            num_mel_bins=self.num_mel_bins,
            temporal_patch_size=self.temporal_patch_size,
            video_fps=self.video_fps,
            video_nframes=self.video_nframes,
            use_temporal_video_embedder=self.use_temporal_video_embedder,
            patch_dim=self.patch_dim,
        )
