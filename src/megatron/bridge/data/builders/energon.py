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

"""Serializable configuration and runtime builder for Energon VLM data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from transformers import AutoProcessor, AutoTokenizer, Qwen3VLProcessor

from megatron.bridge.data.base import DataloaderConfig, DatasetBuildContext, validate_declarative_mapping
from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


def _validate_hf_path(path: str, *, field_name: str) -> None:
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(kw_only=True)
class HFEnergonTaskEncoderConfig:
    """Serializable settings for the generic Hugging Face Energon task encoder.

    ``visual_keys`` names processor output tensors retained in the model batch.
    ``min_pixels`` and ``max_pixels`` are independent processor preprocessing
    bounds controlling visual resolution and token cost; they are not output keys.
    """

    hf_processor_path: str
    visual_keys: tuple[str, ...] = ("pixel_values",)
    min_pixels: int | None = None
    max_pixels: int | None = None
    trust_remote_code: bool | None = None

    def validate(self) -> None:
        """Validate generic Hugging Face task-encoder settings."""
        _validate_hf_path(self.hf_processor_path, field_name="hf_processor_path")
        if not self.visual_keys or not all(isinstance(key, str) and key for key in self.visual_keys):
            raise ValueError("visual_keys must contain non-empty strings.")


@dataclass(kw_only=True)
class QwenVLEnergonTaskEncoderConfig:
    """Serializable settings for the Qwen-VL Energon task encoder.

    Qwen's visual output keys are model-owned. ``min_pixels`` and
    ``max_pixels`` instead bound processor preprocessing and visual-token cost.
    """

    hf_processor_path: str
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    patch_size: int = 14
    min_pixels: int = 200704
    max_pixels: int = 1003520
    max_num_images: int | None = 10
    max_num_frames: int | None = 60
    max_visual_tokens: int | None = 16384
    trust_remote_code: bool | None = None

    def validate(self) -> None:
        """Validate Qwen-VL task-encoder settings."""
        _validate_hf_path(self.hf_processor_path, field_name="hf_processor_path")
        for field_name in ("temporal_patch_size", "spatial_merge_size", "patch_size", "min_pixels", "max_pixels"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than 0.")
        for field_name in ("max_num_images", "max_num_frames", "max_visual_tokens"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be greater than 0 when set.")


@dataclass(kw_only=True)
class NemotronOmniEnergonTaskEncoderConfig:
    """Serializable settings for the Nemotron Omni Energon task encoder.

    ``visual_keys`` is retained for configuration compatibility, but Omni owns
    its visual input contract and supports only ``("pixel_values",)``.
    """

    hf_processor_path: str
    max_audio_duration: float
    num_mel_bins: int
    visual_keys: tuple[str, ...]
    temporal_patch_size: int
    video_fps: float
    video_nframes: int
    use_temporal_video_embedder: bool
    patch_dim: int
    trust_remote_code: bool | None = None

    def validate(self) -> None:
        """Validate Nemotron Omni task-encoder settings."""
        _validate_hf_path(self.hf_processor_path, field_name="hf_processor_path")
        if self.max_audio_duration <= 0:
            raise ValueError("max_audio_duration must be greater than 0.")
        for field_name in ("num_mel_bins", "temporal_patch_size", "video_nframes", "patch_dim"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than 0.")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be greater than 0.")
        if not self.visual_keys or tuple(self.visual_keys) != ("pixel_values",):
            raise ValueError("Nemotron Omni visual_keys must be exactly ('pixel_values',).")


EnergonTaskEncoderConfig = (
    HFEnergonTaskEncoderConfig | QwenVLEnergonTaskEncoderConfig | NemotronOmniEnergonTaskEncoderConfig
)


@dataclass(kw_only=True)
class EnergonDatasetConfig(DataloaderConfig):
    """Serializable configuration for an Energon-backed multimodal dataset."""

    path: str | None = None
    seq_length: int
    micro_batch_size: int
    task_encoder: EnergonTaskEncoderConfig
    dataloader_type: Literal["external"] | None = "external"
    do_validation: bool = True
    do_test: bool = False
    num_val_workers: int | None = None
    shuffle_buffer_size: int = 100
    max_samples_per_sequence: int | None = None
    packing_buffer_size: int | None = None
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    enable_in_batch_packing: bool = False
    defer_in_batch_packing_to_step: bool = False
    pad_to_max_length: bool = False
    pad_to_multiple_of: int = 128
    in_batch_packing_pad_to_multiple_of: int = 1

    def validate(self) -> None:
        """Validate declarative Energon settings."""
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("EnergonDatasetConfig.path must be set to a non-empty dataset path.")
        if self.seq_length <= 0:
            raise ValueError("seq_length must be greater than 0.")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be greater than 0.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be greater than or equal to 0.")
        if self.num_val_workers is not None and self.num_val_workers < 0:
            raise ValueError("num_val_workers must be greater than or equal to 0 when set.")
        if self.shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be greater than 0.")
        if self.max_samples_per_sequence is not None and self.max_samples_per_sequence <= 0:
            raise ValueError("max_samples_per_sequence must be greater than 0 when set.")
        if self.packing_buffer_size is not None and self.packing_buffer_size <= 0:
            raise ValueError("packing_buffer_size must be greater than 0 when set.")
        if self.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be greater than 0.")
        if self.in_batch_packing_pad_to_multiple_of <= 0:
            raise ValueError("in_batch_packing_pad_to_multiple_of must be greater than 0.")
        if self.do_test:
            raise ValueError("EnergonDatasetConfig does not support a distinct test split.")
        if not isinstance(
            self.task_encoder,
            (HFEnergonTaskEncoderConfig, QwenVLEnergonTaskEncoderConfig, NemotronOmniEnergonTaskEncoderConfig),
        ):
            raise TypeError("task_encoder must be a supported declarative Energon task-encoder config.")
        validate_declarative_mapping(self.dataset_kwargs, field_name="dataset_kwargs")
        reserved_dataset_kwargs = {
            "batch_size",
            "max_samples_per_sequence",
            "packing_buffer_size",
            "shuffle_buffer_size",
            "split_part",
            "task_encoder",
            "worker_config",
        }
        conflicting_kwargs = reserved_dataset_kwargs.intersection(self.dataset_kwargs)
        if conflicting_kwargs:
            raise ValueError(
                "dataset_kwargs cannot override builder-owned arguments: " + ", ".join(sorted(conflicting_kwargs))
            )
        self.task_encoder.validate()

    def finalize(self) -> None:
        """Finalize dataloader fields and validate the config."""
        super().finalize()
        self.validate()


def build_energon_task_encoder(config: EnergonDatasetConfig) -> Any:
    """Construct the configured Energon task encoder at runtime."""
    task_config = config.task_encoder
    task_config.validate()
    effective_packing = config.enable_in_batch_packing and not config.defer_in_batch_packing_to_step

    trust_remote_code = is_safe_repo(
        trust_remote_code=(
            task_config.trust_remote_code if task_config.trust_remote_code is not None else config.trust_remote_code
        ),
        hf_path=task_config.hf_processor_path,
    )
    if isinstance(task_config, HFEnergonTaskEncoderConfig):
        from megatron.bridge.data.energon.hf_task_encoder import HFTaskEncoder

        processor = AutoProcessor.from_pretrained(
            task_config.hf_processor_path,
            trust_remote_code=trust_remote_code,
        )
        return HFTaskEncoder(
            processor=processor,
            seq_length=config.seq_length,
            visual_keys=task_config.visual_keys,
            min_pixels=task_config.min_pixels,
            max_pixels=task_config.max_pixels,
            pad_to_max_length=config.pad_to_max_length,
            pad_to_multiple_of=config.pad_to_multiple_of,
            enable_in_batch_packing=effective_packing,
            in_batch_packing_pad_to_multiple_of=config.in_batch_packing_pad_to_multiple_of,
        )

    if isinstance(task_config, QwenVLEnergonTaskEncoderConfig):
        from megatron.bridge.models.qwen_vl.data.energon import QwenVLTaskEncoder

        tokenizer = AutoTokenizer.from_pretrained(
            task_config.hf_processor_path,
            trust_remote_code=trust_remote_code,
        )
        image_processor = Qwen3VLProcessor.from_pretrained(
            task_config.hf_processor_path,
            trust_remote_code=trust_remote_code,
        )
        return QwenVLTaskEncoder(
            tokenizer=tokenizer,
            image_processor=image_processor,
            temporal_patch_size=task_config.temporal_patch_size,
            spatial_merge_size=task_config.spatial_merge_size,
            patch_size=task_config.patch_size,
            max_padding_length=config.seq_length,
            min_pixels=task_config.min_pixels,
            max_pixels=task_config.max_pixels,
            max_num_images=task_config.max_num_images,
            max_num_frames=task_config.max_num_frames,
            max_visual_tokens=task_config.max_visual_tokens,
            pad_to_max_length=config.pad_to_max_length,
            pad_to_multiple_of=config.pad_to_multiple_of,
            enable_in_batch_packing=effective_packing,
            in_batch_packing_pad_to_multiple_of=config.in_batch_packing_pad_to_multiple_of,
        )

    from megatron.bridge.data.energon.nemotron_omni_task_encoder import NemotronOmniTaskEncoder

    processor = AutoProcessor.from_pretrained(
        task_config.hf_processor_path,
        trust_remote_code=trust_remote_code,
    )
    return NemotronOmniTaskEncoder(
        processor=processor,
        seq_length=config.seq_length,
        max_audio_duration=task_config.max_audio_duration,
        num_mel_bins=task_config.num_mel_bins,
        visual_keys=task_config.visual_keys,
        temporal_patch_size=task_config.temporal_patch_size,
        video_fps=task_config.video_fps,
        video_nframes=task_config.video_nframes,
        use_temporal_video_embedder=task_config.use_temporal_video_embedder,
        patch_dim=task_config.patch_dim,
        pad_to_max_length=config.pad_to_max_length,
        pad_to_multiple_of=config.pad_to_multiple_of,
        enable_in_batch_packing=effective_packing,
        in_batch_packing_pad_to_multiple_of=config.in_batch_packing_pad_to_multiple_of,
    )


class EnergonDatasetBuilder:
    """Build Energon runtime dataloader iterators from declarative config."""

    def __init__(self, config: EnergonDatasetConfig) -> None:
        config.validate()
        self.config = config

    def build(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, None]:
        """Build requested Energon train and validation iterators."""
        assert self.config.path is not None
        build_train = context.train_samples > 0
        build_validation = self.config.do_validation and context.valid_samples > 0
        task_encoder = build_energon_task_encoder(self.config) if build_train or build_validation else None
        datamodule = EnergonMultiModalDataModule(
            path=self.config.path,
            tokenizer=context.tokenizer,
            seq_length=self.config.seq_length,
            task_encoder=task_encoder,
            validation_task_encoder=task_encoder,
            micro_batch_size=self.config.micro_batch_size,
            num_workers=self.config.num_workers,
            num_val_workers=self.config.num_val_workers,
            pin_memory=self.config.pin_memory,
            shuffle_buffer_size=self.config.shuffle_buffer_size,
            max_samples_per_sequence=self.config.max_samples_per_sequence,
            packing_buffer_size=self.config.packing_buffer_size,
            pg_collection=context.pg_collection,
            **self.config.dataset_kwargs,
        )
        train = iter(datamodule.train_dataloader()) if build_train else None
        validation = iter(datamodule.val_dataloader()) if build_validation else None
        return train, validation, None


def energon_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int],
    dataset_config: EnergonDatasetConfig,
    tokenizer: Any | None = None,
    pg_collection: Any | None = None,
) -> tuple[Any | None, Any | None, None]:
    """Build Energon iterators through the canonical runtime builder."""
    context = DatasetBuildContext(
        train_samples=train_val_test_num_samples[0],
        valid_samples=train_val_test_num_samples[1],
        test_samples=train_val_test_num_samples[2],
        tokenizer=tokenizer,
        pg_collection=pg_collection,
    )
    return EnergonDatasetBuilder(dataset_config).build(context)
