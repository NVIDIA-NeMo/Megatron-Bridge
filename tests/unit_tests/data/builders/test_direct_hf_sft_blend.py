# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from collections import Counter

import pytest
from megatron.training.config.instantiate_utils import instantiate

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.builders import (
    DirectHFSFTDatasetBuilder,
    DirectHFSFTDatasetConfig,
    HFDatasetSourceConfig,
    PromptCompletionSFTPreprocessingConfig,
)
from megatron.bridge.data.builders import direct_hf_sft as builder_module
from megatron.bridge.data.sources import hf as hf_sources
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


pytestmark = pytest.mark.unit


class _Tokenizer:
    added_tokens_decoder = {}
    pad_token_id = 0
    eos_token_id = 3


def _rows_by_dataset(rows_by_name):
    """Serve a distinct row list per dataset preset name."""

    def _load(source):
        return rows_by_name[source.dataset_name]

    return _load


def _row(tag, index):
    return {"prompt": f"{tag} question {index}", "completion": f"answer {index}"}


def _config(source, weights=None, **kwargs):
    kwargs.setdefault("do_validation", False)
    kwargs.setdefault("do_test", False)
    return DirectHFSFTDatasetConfig(
        seq_length=16,
        source=source,
        source_weights=weights,
        preprocessing=PromptCompletionSFTPreprocessingConfig(
            prompt_column="prompt",
            completion_column="completion",
        ),
        pad_to_multiple_of=1,
        **kwargs,
    )


def _blend():
    return [HFDatasetSourceConfig(dataset_name="squad"), HFDatasetSourceConfig(dataset_name="gsm8k")]


def test_builder_draws_training_rows_from_every_blend_source(monkeypatch):
    monkeypatch.setattr(
        hf_sources,
        "load_and_adapt_hf_dataset",
        _rows_by_dataset(
            {"squad": [_row("squad", i) for i in range(4)], "gsm8k": [_row("gsm8k", i) for i in range(4)]}
        ),
    )
    config = _config(_blend(), [1.0, 3.0])

    train, validation, test = DirectHFSFTDatasetBuilder(config).build(
        DatasetBuildContext(16, 0, 0, tokenizer=_Tokenizer())
    )

    # Weights are epoch counts: squad contributes its 4 rows once, gsm8k three times.
    tags = Counter(train[index]["prompt"].split()[0] for index in range(16))
    assert tags == {"gsm8k": 12, "squad": 4}
    assert (validation, test) == (None, None)


def test_builder_without_a_blend_reads_the_single_source(monkeypatch):
    # The single-source path resolves the loader through the builder module.
    monkeypatch.setattr(builder_module, "load_and_adapt_hf_dataset", _rows_by_dataset({"squad": [_row("squad", 0)]}))
    config = _config(HFDatasetSourceConfig(dataset_name="squad"))

    train, _, _ = DirectHFSFTDatasetBuilder(config).build(DatasetBuildContext(2, 0, 0, tokenizer=_Tokenizer()))

    assert train[0]["prompt"].startswith("squad")


def test_config_rejects_a_weight_count_that_does_not_match(monkeypatch):
    with pytest.raises(ValueError, match="equal in number"):
        _config(_blend(), [1.0]).validate()


def test_config_rejects_a_non_positive_weight():
    with pytest.raises(ValueError, match="must be positive"):
        _config(_blend(), [1.0, 0.0]).validate()


def test_config_rejects_an_empty_blend():
    with pytest.raises(ValueError, match="at least one source"):
        _config([]).validate()


def test_config_validates_every_blend_source():
    with pytest.raises(ValueError, match="Unknown Hugging Face dataset preset"):
        _config([HFDatasetSourceConfig(dataset_name="not-a-preset")]).validate()


def test_blend_config_round_trip_is_declarative():
    config = _config(_blend(), [1.0, 2.0], blend_seed=7)

    restored = instantiate(ConfigContainer._convert_value_to_dict(config))

    assert isinstance(restored, DirectHFSFTDatasetConfig)
    assert [entry.dataset_name for entry in restored.source] == ["squad", "gsm8k"]
    assert restored.source_weights == [1.0, 2.0]
    assert restored.blend_seed == 7
    restored.validate()


def test_blend_weights_can_be_supplied_by_cli_override():
    config = _config(_blend(), [1.0, 1.0])

    process_config_with_overrides(config, cli_overrides=["source_weights=[1.0,4.0]"])
    config.validate()

    assert config.source_weights == [1.0, 4.0]


def test_blend_survives_an_unrelated_cli_override():
    # Any override re-serializes the whole config, so the blend has to come back
    # as source configs even when the override never mentions it.
    config = _config(_blend(), [1.0, 2.0])

    process_config_with_overrides(config, cli_overrides=["seq_length=32"])
    config.validate()

    assert config.seq_length == 32
    assert all(isinstance(entry, HFDatasetSourceConfig) for entry in config.source)
    assert [entry.dataset_name for entry in config.source] == ["squad", "gsm8k"]


@pytest.mark.parametrize("seed", [None, 1.5, "1234", True])
def test_config_rejects_a_non_integer_blend_seed(seed):
    # random.Random(None) reseeds from entropy, so ranks would disagree on order.
    with pytest.raises(ValueError, match="blend_seed must be an integer"):
        _config(_blend(), [1.0, 2.0], blend_seed=seed).validate()


def test_config_rejects_a_non_finite_blend_weight():
    with pytest.raises(ValueError, match="positive finite"):
        _config(_blend(), [1.0, float("nan")]).validate()


def test_blend_introduced_by_override_keeps_split_sources_usable():
    # A scalar-source config gains a blend through Hydra, and the split sources
    # the blend now requires arrive in the same override as plain mappings.
    config = _config(HFDatasetSourceConfig(dataset_name="squad"), do_validation=True, do_test=True)

    process_config_with_overrides(
        config,
        cli_overrides=[
            "~source",
            "+source=[{dataset_name: squad}, {dataset_name: gsm8k}]",
            "+validation_source={dataset_name: squad, split: validation}",
            "+test_source={dataset_name: gsm8k, split: test}",
        ],
    )
    config.validate()

    assert [entry.dataset_name for entry in config.source] == ["squad", "gsm8k"]
    assert isinstance(config.validation_source, HFDatasetSourceConfig)
    assert isinstance(config.test_source, HFDatasetSourceConfig)
    assert config.validation_source.split == "validation"
