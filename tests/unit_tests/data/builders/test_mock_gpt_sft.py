import numpy
import pytest
from megatron.training.config.instantiate_utils import instantiate

from megatron.bridge.data.builders import (
    MockGPTSFTDataset,
    MockGPTSFTDatasetConfig,
    generate_lognormal_sequence_lengths,
    mock_gpt_sft_train_valid_test_datasets_provider,
)
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.training.config import ConfigContainer


pytestmark = pytest.mark.unit


def _config(**overrides) -> MockGPTSFTDatasetConfig:
    values = {
        "seq_length": 128,
        "min_sequence_length": 16,
        "max_sequence_length": 128,
        "mean_sequence_length": 64,
        "lognormal_sigma": 0.7,
        "num_base_samples": 32,
        "in_batch_packing_pad_to_multiple_of": 4,
        "num_workers": 0,
        "persistent_workers": False,
    }
    values.update(overrides)
    return MockGPTSFTDatasetConfig(**values)


def test_mock_config_round_trip_preserves_distribution_settings():
    config = _config(defer_in_batch_packing_to_step=True)

    serialized = ConfigContainer._convert_value_to_dict(config)
    restored = instantiate(serialized)

    assert isinstance(restored, MockGPTSFTDatasetConfig)
    assert restored.mean_sequence_length == 64
    assert restored.lognormal_sigma == 0.7
    assert restored.defer_in_batch_packing_to_step is True


def test_lognormal_lengths_are_deterministic_clipped_and_variable():
    config = _config(num_base_samples=10_000, seed=7)

    first = generate_lognormal_sequence_lengths(config)
    second = generate_lognormal_sequence_lengths(config)

    numpy.testing.assert_array_equal(first, second)
    assert first.dtype == numpy.int64
    assert first.min() == config.min_sequence_length
    assert first.max() == config.max_sequence_length
    assert numpy.unique(first).size > 100
    assert numpy.median(first) < config.resolved_mean_sequence_length


def test_dataset_materializes_exact_length_shifted_tokens():
    dataset = MockGPTSFTDataset(
        config=_config(),
        sequence_lengths=numpy.array([3, 5], dtype=numpy.int64),
        target_length=4,
    )

    sample = dataset[1]

    assert len(dataset) == 4
    assert sample["tokens"].tolist() == [0, 1, 2, 3, 4]
    assert sample["labels"].tolist() == [1, 2, 3, 4, 5]
    assert sample["position_ids"].tolist() == [0, 1, 2, 3, 4]
    assert sample["loss_mask"].sum().item() == 5


def test_deferred_collate_preserves_rows_and_sequence_lengths():
    config = _config(defer_in_batch_packing_to_step=True)
    dataset = MockGPTSFTDataset(
        config=config,
        sequence_lengths=numpy.array([3, 5], dtype=numpy.int64),
        target_length=2,
    )

    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert batch["tokens"].shape == (2, 5)
    assert batch["sequence_lengths"].tolist() == [3, 5]
    assert batch["loss_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert batch["attention_mask"] is None
    assert "cu_seqlens_q" not in batch


def test_static_collate_builds_aligned_thd_pack():
    config = _config(defer_in_batch_packing_to_step=False)
    dataset = MockGPTSFTDataset(
        config=config,
        sequence_lengths=numpy.array([3, 5], dtype=numpy.int64),
        target_length=2,
    )

    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert batch["tokens"].shape == (1, 12)
    assert batch["cu_seqlens_q"].tolist() == [0, 3, 8]
    assert batch["cu_seqlens_q_padded"].tolist() == [0, 4, 12]
    assert batch["padding_mask"].sum().item() == 4
    assert batch["max_seqlen_q"].item() == 8


def test_provider_builds_only_requested_splits_and_has_specific_registry_entry():
    config = _config(num_base_samples=8)

    train, validation, test = mock_gpt_sft_train_valid_test_datasets_provider([3, 0, 2], config)

    assert len(train) == 3
    assert validation is None
    assert len(test) == 2
    assert train.sequence_lengths is test.sequence_lengths
    assert get_dataset_provider(config) is mock_gpt_sft_train_valid_test_datasets_provider


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"min_sequence_length": 0}, "min_sequence_length"),
        ({"max_sequence_length": 256}, "max_sequence_length"),
        ({"mean_sequence_length": 8}, "mean_sequence_length"),
        ({"lognormal_sigma": 0}, "lognormal_sigma"),
        ({"num_base_samples": 0}, "num_base_samples"),
        ({"enable_in_batch_packing": False}, "enable_in_batch_packing"),
        ({"dataloader_type": "batch"}, "dataloader_type"),
        ({"dataset_root": "/tmp/data"}, "does not accept"),
    ],
)
def test_mock_config_rejects_invalid_values(overrides, error):
    with pytest.raises(ValueError, match=error):
        _config(**overrides).validate()


def test_padding_sample_has_zero_loss():
    dataset = MockGPTSFTDataset(
        config=_config(),
        sequence_lengths=numpy.array([3], dtype=numpy.int64),
        target_length=1,
    )

    assert dataset[-1]["loss_mask"].sum().item() == 0
