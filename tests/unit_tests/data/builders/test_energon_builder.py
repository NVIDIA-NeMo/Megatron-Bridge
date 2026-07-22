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

"""Focused coverage for declarative Energon config and runtime construction."""

from unittest.mock import MagicMock

import pytest
from megatron.training.config.instantiate_utils import instantiate

from megatron.bridge.data import DatasetBuildContext
from megatron.bridge.data.builders.energon import (
    EnergonDatasetBuilder,
    EnergonDatasetConfig,
    HFEnergonTaskEncoderConfig,
    NemotronOmniEnergonTaskEncoderConfig,
    QwenVLEnergonTaskEncoderConfig,
    build_energon_task_encoder,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


def _qwen_config(**overrides) -> EnergonDatasetConfig:
    values = {
        "path": "/data/shards",
        "seq_length": 4096,
        "micro_batch_size": 2,
        "num_workers": 0,
        "task_encoder": QwenVLEnergonTaskEncoderConfig(hf_processor_path="Qwen/model"),
        "defer_in_batch_packing_to_step": True,
    }
    values.update(overrides)
    return EnergonDatasetConfig(**values)


def test_config_round_trip_is_declarative_and_cli_overridable():
    config = _qwen_config(dataset_kwargs={"handler": "warn_and_continue"})

    serialized = ConfigContainer._convert_value_to_dict(config)
    restored = instantiate(serialized)

    assert isinstance(restored, EnergonDatasetConfig)
    assert isinstance(restored.task_encoder, QwenVLEnergonTaskEncoderConfig)
    assert restored.dataset_kwargs == {"handler": "warn_and_continue"}
    assert "processor" not in serialized
    assert "tokenizer" not in serialized

    process_config_with_overrides(
        restored,
        cli_overrides=[
            "task_encoder.max_num_images=4",
            "task_encoder.max_visual_tokens=8192",
        ],
    )
    assert restored.task_encoder.max_num_images == 4
    assert restored.task_encoder.max_visual_tokens == 8192


@pytest.mark.parametrize("reserved_key", ["batch_size", "task_encoder", "split_part", "worker_config"])
def test_config_rejects_builder_owned_dataset_kwargs(reserved_key: str):
    config = _qwen_config(dataset_kwargs={reserved_key: 1})

    with pytest.raises(ValueError, match="builder-owned"):
        config.validate()


def test_qwen_factory_preserves_limits_and_deferred_packing(monkeypatch: pytest.MonkeyPatch):
    config = _qwen_config(
        trust_remote_code=True,
        enable_in_batch_packing=True,
        pad_to_max_length=True,
        pad_to_multiple_of=64,
        in_batch_packing_pad_to_multiple_of=8,
        task_encoder=QwenVLEnergonTaskEncoderConfig(
            hf_processor_path="Qwen/model",
            min_pixels=123,
            max_pixels=456,
            max_num_images=4,
            max_num_frames=16,
            max_visual_tokens=999,
        ),
    )
    tokenizer = object()
    processor = object()
    encoder = object()
    encoder_cls = MagicMock(return_value=encoder)
    safe_repo = MagicMock(return_value=False)
    monkeypatch.setattr("megatron.bridge.data.builders.energon.is_safe_repo", safe_repo)
    monkeypatch.setattr(
        "megatron.bridge.data.builders.energon.AutoTokenizer.from_pretrained",
        lambda *_, **__: tokenizer,
    )
    monkeypatch.setattr(
        "megatron.bridge.data.builders.energon.Qwen3VLProcessor.from_pretrained",
        lambda *_, **__: processor,
    )
    monkeypatch.setattr("megatron.bridge.models.qwen_vl.data.energon.QwenVLTaskEncoder", encoder_cls)

    assert build_energon_task_encoder(config) is encoder

    safe_repo.assert_called_once_with(trust_remote_code=True, hf_path="Qwen/model")
    encoder_cls.assert_called_once_with(
        tokenizer=tokenizer,
        image_processor=processor,
        temporal_patch_size=2,
        spatial_merge_size=2,
        patch_size=14,
        max_padding_length=4096,
        min_pixels=123,
        max_pixels=456,
        max_num_images=4,
        max_num_frames=16,
        max_visual_tokens=999,
        pad_to_max_length=True,
        pad_to_multiple_of=64,
        enable_in_batch_packing=False,
        in_batch_packing_pad_to_multiple_of=8,
    )


def test_generic_hf_factory_uses_collate_time_packing(monkeypatch: pytest.MonkeyPatch):
    config = EnergonDatasetConfig(
        path="/data/shards",
        seq_length=2048,
        micro_batch_size=1,
        task_encoder=HFEnergonTaskEncoderConfig(
            hf_processor_path="org/model",
            visual_keys=("pixel_values", "image_sizes"),
            min_pixels=100,
            max_pixels=200,
        ),
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )
    processor = object()
    encoder = object()
    encoder_cls = MagicMock(return_value=encoder)
    monkeypatch.setattr("megatron.bridge.data.builders.energon.is_safe_repo", lambda **_: False)
    monkeypatch.setattr(
        "megatron.bridge.data.builders.energon.AutoProcessor.from_pretrained",
        lambda *_, **__: processor,
    )
    monkeypatch.setattr("megatron.bridge.data.energon.hf_task_encoder.HFTaskEncoder", encoder_cls)

    assert build_energon_task_encoder(config) is encoder
    assert encoder_cls.call_args.kwargs["enable_in_batch_packing"] is True
    assert encoder_cls.call_args.kwargs["in_batch_packing_pad_to_multiple_of"] == 4
    assert encoder_cls.call_args.kwargs["visual_keys"] == ("pixel_values", "image_sizes")
    assert encoder_cls.call_args.kwargs["min_pixels"] == 100
    assert encoder_cls.call_args.kwargs["max_pixels"] == 200


def test_nemotron_factory_preserves_omni_settings(monkeypatch: pytest.MonkeyPatch):
    config = EnergonDatasetConfig(
        path="/data/shards",
        seq_length=1024,
        micro_batch_size=1,
        task_encoder=NemotronOmniEnergonTaskEncoderConfig(
            hf_processor_path="nvidia/model",
            max_audio_duration=10.0,
            num_mel_bins=128,
            visual_keys=("pixel_values",),
            temporal_patch_size=2,
            video_fps=1.0,
            video_nframes=8,
            use_temporal_video_embedder=True,
            patch_dim=16,
        ),
        enable_in_batch_packing=True,
    )
    processor = object()
    encoder = object()
    encoder_cls = MagicMock(return_value=encoder)
    monkeypatch.setattr("megatron.bridge.data.builders.energon.is_safe_repo", lambda **_: False)
    monkeypatch.setattr(
        "megatron.bridge.data.builders.energon.AutoProcessor.from_pretrained",
        lambda *_, **__: processor,
    )
    monkeypatch.setattr(
        "megatron.bridge.data.energon.nemotron_omni_task_encoder.NemotronOmniTaskEncoder",
        encoder_cls,
    )

    assert build_energon_task_encoder(config) is encoder
    assert encoder_cls.call_args.kwargs["processor"] is processor
    assert encoder_cls.call_args.kwargs["max_audio_duration"] == 10.0
    assert encoder_cls.call_args.kwargs["use_temporal_video_embedder"] is True
    assert encoder_cls.call_args.kwargs["enable_in_batch_packing"] is True


def test_nemotron_config_rejects_unsupported_visual_keys():
    config = NemotronOmniEnergonTaskEncoderConfig(
        hf_processor_path="nvidia/model",
        max_audio_duration=10.0,
        num_mel_bins=128,
        visual_keys=("pixel_values", "image_sizes"),
        temporal_patch_size=2,
        video_fps=1.0,
        video_nframes=8,
        use_temporal_video_embedder=True,
        patch_dim=16,
    )

    with pytest.raises(ValueError, match=r"visual_keys must be exactly \('pixel_values',\)"):
        config.validate()


def test_builder_honors_requested_splits_and_reuses_runtime_encoder(monkeypatch: pytest.MonkeyPatch):
    config = _qwen_config(num_val_workers=0, packing_buffer_size=32)
    task_encoder = object()
    build_encoder = MagicMock(return_value=task_encoder)
    datamodule = MagicMock()
    datamodule.train_dataloader.return_value = ["train"]
    datamodule.val_dataloader.return_value = ["validation"]
    datamodule_cls = MagicMock(return_value=datamodule)
    monkeypatch.setattr("megatron.bridge.data.builders.energon.build_energon_task_encoder", build_encoder)
    monkeypatch.setattr("megatron.bridge.data.builders.energon.EnergonMultiModalDataModule", datamodule_cls)

    train, validation, test = EnergonDatasetBuilder(config).build(
        DatasetBuildContext(train_samples=10, valid_samples=5, test_samples=3)
    )

    assert list(train) == ["train"]
    assert list(validation) == ["validation"]
    assert test is None
    build_encoder.assert_called_once_with(config)
    assert datamodule_cls.call_args.kwargs["task_encoder"] is task_encoder
    assert datamodule_cls.call_args.kwargs["validation_task_encoder"] is task_encoder
    assert datamodule_cls.call_args.kwargs["num_val_workers"] == 0
    assert datamodule_cls.call_args.kwargs["packing_buffer_size"] == 32


def test_builder_skips_unrequested_validation(monkeypatch: pytest.MonkeyPatch):
    config = _qwen_config(do_validation=False)
    datamodule = MagicMock()
    datamodule.train_dataloader.return_value = []
    monkeypatch.setattr("megatron.bridge.data.builders.energon.build_energon_task_encoder", lambda _: object())
    monkeypatch.setattr(
        "megatron.bridge.data.builders.energon.EnergonMultiModalDataModule",
        MagicMock(return_value=datamodule),
    )

    _, validation, test = EnergonDatasetBuilder(config).build(
        DatasetBuildContext(train_samples=1, valid_samples=1, test_samples=1)
    )

    assert validation is None
    assert test is None
    datamodule.val_dataloader.assert_not_called()
