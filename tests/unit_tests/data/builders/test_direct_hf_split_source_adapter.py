# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from megatron.bridge.data.builders.direct_hf_sft import DirectHFSFTDatasetConfig
from megatron.bridge.data.sources import hf_adapters as hf_adapter_module
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig
from megatron.bridge.data.sources.hf_adapters import adapt_hf_dataset


pytestmark = pytest.mark.unit


def test_explicit_custom_validation_source_keeps_own_adapter_columns(monkeypatch):
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(
            path_or_dataset="org/train",
            schema_adapter="default_audio",
            adapter_kwargs={"audio_column": "train_audio", "text_column": "train_text"},
        ),
        validation_source=HFDatasetSourceConfig(
            path_or_dataset="org/validation",
            schema_adapter="default_audio",
        ),
        do_test=False,
    )
    config.validate()
    monkeypatch.setattr(hf_adapter_module, "_decode_audio", lambda _audio: ([0.0], 16_000))

    adapted = adapt_hf_dataset(
        [{"audio": {}, "text": "validation"}],
        adapter_name=config.validation_source.schema_adapter,
        adapter_kwargs=config.validation_source.adapter_kwargs,
    )

    assert adapted[0]["conversation"][-1]["content"][0]["text"] == "validation"


def test_custom_validation_split_of_training_source_inherits_adapter_columns():
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(
            path_or_dataset="org/dataset",
            schema_adapter="default_audio",
            adapter_kwargs={"audio_column": "recording", "text_column": "transcript"},
        ),
        validation_source=HFDatasetSourceConfig(
            path_or_dataset="org/dataset",
            split="validation",
            schema_adapter="default_audio",
        ),
        do_test=False,
    )

    config.validate()

    assert config.validation_source.adapter_kwargs == {
        "audio_column": "recording",
        "text_column": "transcript",
    }
