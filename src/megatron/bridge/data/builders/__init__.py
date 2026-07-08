"""Canonical dataset Config + Builder APIs."""

from megatron.bridge.data.builders import gpt_sft_dataset as _gpt_sft_dataset
from megatron.bridge.data.builders.gpt_sft_dataset import (
    GPTSFTDatasetBuilder,
    GPTSFTDatasetConfig,
    gpt_sft_train_valid_test_datasets_provider,
)
from megatron.bridge.data.builders.hf_sft_dataset import (
    HFSFTDatasetBuilder,
    HFSFTDatasetConfig,
    hf_sft_train_valid_test_datasets_provider,
)
from megatron.bridge.data.hf_source import HFDatasetSourceConfig
from megatron.bridge.data.sft_processing import (
    ChatSFTPreprocessingConfig,
    PromptCompletionSFTPreprocessingConfig,
    SFTPreprocessingConfig,
)


# =============================================================================
# Deprecated compatibility exports
# =============================================================================
FinetuningDatasetBuilder = _gpt_sft_dataset.FinetuningDatasetBuilder
FinetuningDatasetConfig = _gpt_sft_dataset.FinetuningDatasetConfig


__all__ = [
    "GPTSFTDatasetBuilder",
    "GPTSFTDatasetConfig",
    "ChatSFTPreprocessingConfig",
    "HFDatasetSourceConfig",
    "HFSFTDatasetBuilder",
    "HFSFTDatasetConfig",
    "PromptCompletionSFTPreprocessingConfig",
    "SFTPreprocessingConfig",
    "gpt_sft_train_valid_test_datasets_provider",
    "hf_sft_train_valid_test_datasets_provider",
    # Deprecated compatibility exports.
    "FinetuningDatasetBuilder",
    "FinetuningDatasetConfig",
]
