# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.data.hf_datasets import text_sft_provider as provider_mod
from megatron.bridge.data.hf_datasets.text_sft_provider import HFTextSFTDatasetProvider


pytestmark = pytest.mark.unit


class _FakeBuilder:
    config = None

    def __init__(self, *, config, tokenizer):
        type(self).config = config
        assert tokenizer is not None

    def build(self):
        return "train", "validation", "test"


def test_deprecated_provider_delegates_to_config_and_builder(monkeypatch, tmp_path):
    monkeypatch.setattr(provider_mod, "_get_gpt_sft_dataset_builder", lambda: _FakeBuilder)

    with pytest.warns(DeprecationWarning, match="GPTSFTDatasetConfig"):
        provider = HFTextSFTDatasetProvider(
            seq_length=128,
            maker_name="squad",
            maker_kwargs={"path_or_dataset": "mock/squad"},
            dataset_root=tmp_path,
            do_test=False,
        )

    assert provider.build_datasets(DatasetBuildContext(1, 1, 0, tokenizer=object())) == (
        "train",
        "validation",
        "test",
    )
    assert isinstance(_FakeBuilder.config, GPTSFTDatasetConfig)
    assert _FakeBuilder.config.hf_dataset.schema_adapter == "squad"
    assert _FakeBuilder.config.hf_output_root == tmp_path


def test_deprecated_provider_requires_tokenizer():
    with pytest.warns(DeprecationWarning):
        provider = HFTextSFTDatasetProvider(seq_length=128, maker_name="squad")

    with pytest.raises(ValueError, match="requires a tokenizer"):
        provider.build_datasets(DatasetBuildContext(1, 0, 0))


@pytest.mark.parametrize(
    ("maker_name", "path", "subset", "split"),
    [
        ("squad", "rajpurkar/squad", None, "train"),
        ("gsm8k", "openai/gsm8k", "main", "train"),
        ("openmathinstruct2", "nvidia/OpenMathInstruct-2", None, "train_1M"),
    ],
)
def test_deprecated_provider_preserves_builtin_maker_source_defaults(maker_name, path, subset, split):
    with pytest.warns(DeprecationWarning):
        provider = HFTextSFTDatasetProvider(seq_length=128, maker_name=maker_name)

    config = provider._to_config()  # noqa: SLF001

    assert config.hf_dataset.path_or_dataset == path
    assert config.hf_dataset.subset == subset
    assert config.hf_dataset.split == split


def test_deprecated_provider_partial_validation_override_uses_validation_split():
    with pytest.warns(DeprecationWarning):
        provider = HFTextSFTDatasetProvider(
            seq_length=128,
            maker_name="squad",
            val_maker_kwargs={"revision": "main"},
            do_test=False,
        )

    config = provider._to_config()  # noqa: SLF001

    assert config.hf_validation_dataset.path_or_dataset == "rajpurkar/squad"
    assert config.hf_validation_dataset.split == "validation"
    assert config.hf_validation_dataset.load_kwargs == {"revision": "main"}


def test_deprecated_provider_preserves_explicit_base_split_for_validation():
    with pytest.warns(DeprecationWarning):
        provider = HFTextSFTDatasetProvider(
            seq_length=128,
            maker_name="squad",
            maker_kwargs={"split": "custom_validation"},
            val_maker_kwargs={},
            do_test=False,
        )

    config = provider._to_config()  # noqa: SLF001

    assert config.hf_validation_dataset.split == "custom_validation"


def test_deprecated_provider_explicit_validation_source_takes_precedence_over_proportion():
    with pytest.warns(DeprecationWarning):
        provider = HFTextSFTDatasetProvider(
            seq_length=128,
            maker_name="squad",
            val_maker_kwargs={"split": "validation"},
            val_proportion=0.1,
            do_test=False,
        )

    config = provider._to_config()  # noqa: SLF001

    assert config.hf_validation_dataset is not None
    assert config.hf_validation_proportion is None
    config.validate()


def test_deprecated_provider_ignores_disabled_split_kwargs():
    with pytest.warns(DeprecationWarning):
        provider = HFTextSFTDatasetProvider(
            seq_length=128,
            maker_name="squad",
            val_maker_kwargs={"split": "validation"},
            test_maker_kwargs={"split": "test"},
            val_proportion=0.1,
            do_validation=False,
            do_test=False,
        )

    config = provider._to_config()  # noqa: SLF001

    assert config.hf_validation_dataset is None
    assert config.hf_test_dataset is None
    assert config.hf_validation_proportion is None
    config.validate()
