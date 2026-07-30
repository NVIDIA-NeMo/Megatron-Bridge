from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.bridge.data import utils as data_utils
from megatron.bridge.data.loaders import _get_collate_fn
from megatron.bridge.training.config import MockVarlenDatasetConfig


pytestmark = pytest.mark.unit


def _mock_varlen_config() -> MockVarlenDatasetConfig:
    return MockVarlenDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        seq_length=65_536,
        varlen_mock_dataset_config_json=(
            '{"mode":"distribution","type":"lognormal","format":"thd",'
            '"min_seq_len":65534,"max_seq_len":65534,"mean_seq_len":65534,'
            '"lognormal_sigma":1.1}'
        ),
    )


def test_mock_varlen_loader_preserves_samples_as_a_list() -> None:
    cfg = SimpleNamespace(dataset=_mock_varlen_config())
    collate_fn = _get_collate_fn(object(), cfg)
    samples = [{"tokens": object()}, {"tokens": object()}]

    assert collate_fn is not None
    assert collate_fn(samples) is samples


def test_mock_varlen_provider_builds_tp_zero_dataset_on_every_pp_stage(monkeypatch) -> None:
    varlen_dataset = pytest.importorskip("megatron.training.datasets.varlen_dataset")
    builder = Mock()
    builder.return_value.build.return_value = ("train", "valid", "test")
    owns_dataset = Mock(return_value=True)
    monkeypatch.setattr(data_utils, "BlendedMegatronDatasetBuilder", builder)
    monkeypatch.setattr(data_utils, "is_dataset_built_on_rank", owns_dataset)
    pg_collection = object()
    dataset_config = _mock_varlen_config()

    result = data_utils.pretrain_train_valid_test_datasets_provider(
        [256, 0, 0],
        dataset_config,
        pg_collection=pg_collection,
    )

    assert result == ("train", "valid", "test")
    dataset_type, sample_counts, rank_predicate, captured_config = builder.call_args.args
    assert dataset_type is varlen_dataset.MockVarlenDataset
    assert sample_counts == [256, 0, 0]
    assert captured_config is dataset_config
    assert rank_predicate() is True
    owns_dataset.assert_called_once_with(pg_collection, include_middle_pipeline_stages=True)
