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

from unittest.mock import MagicMock

import pytest
from megatron.training.config.instantiate_utils import instantiate

import megatron.bridge.models.nemotron_omni.data.collate_fn as nemotron_omni_collate
from megatron.bridge.data import DatasetBuildContext
from megatron.bridge.data.builders import MockVLMSFTDatasetBuilder, MockVLMSFTDatasetConfig
from megatron.bridge.data.builders import mock_vlm_sft as builder_module
from megatron.bridge.data.collators.registry import resolve_model_collate
from megatron.bridge.training.config import ConfigContainer


def _config(**overrides) -> MockVLMSFTDatasetConfig:
    values = {
        "seq_length": 128,
        "hf_processor_path": "org/model",
        "num_images": 0,
        "num_base_examples": 4,
    }
    values.update(overrides)
    return MockVLMSFTDatasetConfig(**values)


def test_mock_config_round_trip_contains_no_runtime_processor():
    config = _config(enable_in_batch_packing=True, defer_in_batch_packing_to_step=True)

    serialized = ConfigContainer._convert_value_to_dict(config)
    restored = instantiate(serialized)

    assert isinstance(restored, MockVLMSFTDatasetConfig)
    assert restored.enable_in_batch_packing is True
    assert restored.defer_in_batch_packing_to_step is True
    assert "processor" not in serialized


def test_mock_examples_are_deterministic_and_follow_conversation_schema():
    config = _config(random_seed=42, num_images=1, image_size=(8, 6))

    first = builder_module.make_mock_vlm_examples(config)
    second = builder_module.make_mock_vlm_examples(config)

    assert len(first) == 4
    assert first[0]["conversation"][0]["content"][-1] == {
        "type": "text",
        "text": "Describe this image.",
    }
    assert first[0]["conversation"][1] == second[0]["conversation"][1]
    assert first[0]["conversation"][0]["content"][0]["image"].size == (8, 6)


def test_builder_loads_processor_at_runtime_and_builds_requested_splits(monkeypatch: pytest.MonkeyPatch):
    config = _config(
        enable_in_batch_packing=True,
        defer_in_batch_packing_to_step=True,
        pad_to_max_length=True,
        pad_to_multiple_of=64,
        in_batch_packing_pad_to_multiple_of=8,
    )
    processor = object()
    processor_loader = MagicMock(return_value=processor)
    captured_kwargs = []

    class CapturingDirectSFTDataset:
        def __init__(self, **kwargs):
            captured_kwargs.append(kwargs)
            self.target_length = kwargs["target_length"]

        def __len__(self):
            return self.target_length

    monkeypatch.setattr(builder_module.AutoProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(builder_module, "is_safe_repo", lambda **_: False)
    monkeypatch.setattr(builder_module, "DirectSFTDataset", CapturingDirectSFTDataset)

    train, validation, test = MockVLMSFTDatasetBuilder(config).build(
        DatasetBuildContext(train_samples=3, valid_samples=2, test_samples=0)
    )

    processor_loader.assert_called_once_with("org/model", trust_remote_code=False)
    assert len(train) == 3
    assert len(validation) == 2
    assert test is None
    assert len(captured_kwargs) == 2
    for kwargs in captured_kwargs:
        assert kwargs["processor"] is processor
        assert kwargs["enable_in_batch_packing"] is True
        assert kwargs["defer_in_batch_packing_to_step"] is True
        assert kwargs["pad_to_max_length"] is True
        assert kwargs["pad_to_multiple_of"] == 64
        assert kwargs["in_batch_packing_pad_to_multiple_of"] == 8


def test_builder_collates_super_vl_processor_with_nemotron_omni_collator(monkeypatch: pytest.MonkeyPatch):
    processor = type("NemotronH_Omni_Reasoning_V3Processor", (), {})()
    calls = []

    def expanded_collate(examples, processor, **kwargs):
        calls.append((examples, processor))
        return {}

    monkeypatch.setattr(builder_module.AutoProcessor, "from_pretrained", MagicMock(return_value=processor))
    monkeypatch.setattr(nemotron_omni_collate, "nemotron_omni_expanded_collate_fn", expanded_collate)
    resolve_model_collate.cache_clear()
    try:
        train, _, _ = MockVLMSFTDatasetBuilder(_config(trust_remote_code=True)).build(
            DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0)
        )
        examples = [train[0]]
        train.collate_fn(examples)
    finally:
        # Drop the cached test double before monkeypatch restores the real collator.
        resolve_model_collate.cache_clear()

    assert calls == [(examples, processor)]


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"num_images": -1}, "num_images"),
        ({"image_size": (0, 16)}, "image_size"),
        ({"pad_to_multiple_of": 0}, "pad_to_multiple_of"),
    ],
)
def test_mock_config_rejects_invalid_values(overrides, error):
    with pytest.raises(ValueError, match=error):
        _config(**overrides).validate()
