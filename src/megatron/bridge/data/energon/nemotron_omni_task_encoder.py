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

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import torch

from megatron.bridge.data.energon.hf_task_encoder import HFEnergonBatch, HFEnergonSample, HFTaskEncoder
from megatron.bridge.models.nemotron_omni.data.collate_fn import (
    _validate_nemotron_omni_visual_keys,
    nemotron_omni_collate_fn,
)
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


# Import-name migration alias. The sample representation intentionally changed
# from model-ready tensors to the shared HF-style example contract; callers that
# directly constructed the old sample dataclass must migrate to ``example``.
NemotronOmniTaskSample = HFEnergonSample


_VISUAL_INPUT_FIELDS = frozenset(field.name for field in fields(GenericVisualInputs))


class _LegacyVisualTensorMapping(MutableMapping[str, Any]):
    """Live mapping view over a batch's ``GenericVisualInputs`` fields."""

    def __init__(self, batch: "NemotronOmniTaskBatch") -> None:
        self.batch = batch

    def __getitem__(self, key: str) -> Any:
        if key not in _VISUAL_INPUT_FIELDS or self.batch.visual_inputs is None:
            raise KeyError(key)
        value = getattr(self.batch.visual_inputs, key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in _VISUAL_INPUT_FIELDS:
            raise KeyError(key)
        if self.batch.visual_inputs is None:
            self.batch.visual_inputs = GenericVisualInputs()
        setattr(self.batch.visual_inputs, key, value)

    def __delitem__(self, key: str) -> None:
        self[key]
        assert self.batch.visual_inputs is not None
        setattr(self.batch.visual_inputs, key, None)

    def __iter__(self) -> Iterator[str]:
        if self.batch.visual_inputs is None:
            return iter(())
        return iter(self.batch.visual_inputs.as_model_kwargs())

    def __len__(self) -> int:
        return sum(1 for _ in self)


@dataclass(init=False)
class NemotronOmniTaskBatch(HFEnergonBatch):
    """HF-style batch with the legacy ``visual_tensors`` constructor/property."""

    num_patches: torch.Tensor | None = None
    sound_clips: torch.Tensor | None = None
    sound_length: torch.Tensor | None = None
    imgs_sizes: torch.Tensor | None = None
    num_frames: torch.Tensor | None = None
    num_image_tiles: torch.Tensor | None = None

    def __init__(
        self,
        *args: Any,
        visual_inputs: GenericVisualInputs | None = None,
        visual_tensors: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        omni_fields = {
            name: kwargs.pop(name, None)
            for name in ("num_patches", "sound_clips", "sound_length", "imgs_sizes", "num_frames", "num_image_tiles")
        }
        if visual_inputs is not None and visual_tensors is not None:
            raise ValueError("Specify only one of visual_inputs or legacy visual_tensors.")
        if visual_inputs is None and visual_tensors is not None:
            visual_inputs = GenericVisualInputs(
                **{key: value for key, value in visual_tensors.items() if value is not None}
            )
        super().__init__(*args, visual_inputs=visual_inputs, **kwargs)
        for name, value in omni_fields.items():
            setattr(self, name, value)
        self._visual_tensors_proxy = _LegacyVisualTensorMapping(self)

    @property
    def visual_tensors(self) -> MutableMapping[str, Any]:
        """Expose a live legacy visual tensor mapping for low-level batch consumers."""
        return self._visual_tensors_proxy

    @visual_tensors.setter
    def visual_tensors(self, value: Mapping[str, Any] | None) -> None:
        """Normalize assignment through the legacy visual tensor attribute."""
        self.visual_inputs = (
            GenericVisualInputs(**{key: item for key, item in value.items() if item is not None})
            if value is not None
            else None
        )


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
        _validate_nemotron_omni_visual_keys(visual_keys)
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

    def batch(self, samples: list[HFEnergonSample]) -> NemotronOmniTaskBatch:
        """Collate shared samples while retaining the model-specific batch type."""
        collated, batch_kwargs = self._collate_batch_kwargs(samples)
        batch_kwargs.update(
            num_patches=collated.get("num_patches"),
            sound_clips=collated.get("sound_clips"),
            sound_length=collated.get("sound_length"),
            imgs_sizes=collated.get("imgs_sizes"),
            num_frames=collated.get("num_frames"),
            num_image_tiles=collated.get("num_image_tiles"),
        )
        return NemotronOmniTaskBatch(**batch_kwargs)

    def encode_batch(self, batch: HFEnergonBatch) -> dict[str, Any]:
        """Return the shared batch plus the legacy ``tokens`` alias."""
        raw = super().encode_batch(batch)
        raw["tokens"] = raw["input_ids"]
        return raw
