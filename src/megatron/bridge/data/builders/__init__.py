"""Canonical dataset Config + Builder APIs."""

from megatron.bridge.data.builders import gpt_sft as _gpt_sft
from megatron.bridge.data.builders.direct_hf_sft import (
    DirectHFSFTDatasetBuilder,
    DirectHFSFTDatasetConfig,
    direct_hf_sft_train_valid_test_datasets_provider,
)
from megatron.bridge.data.builders.energon import (
    EnergonDatasetBuilder,
    EnergonDatasetConfig,
    HFEnergonTaskEncoderConfig,
    NemotronOmniEnergonTaskEncoderConfig,
    QwenVLEnergonTaskEncoderConfig,
    energon_train_valid_test_datasets_provider,
)
from megatron.bridge.data.builders.gpt_sft import (
    GPTSFTDatasetBuilder,
    GPTSFTDatasetConfig,
    gpt_sft_train_valid_test_datasets_provider,
)
from megatron.bridge.data.builders.mock_vlm_sft import (
    MockVLMSFTDatasetBuilder,
    MockVLMSFTDatasetConfig,
    mock_vlm_sft_train_valid_test_datasets_provider,
)
from megatron.bridge.data.sft_processing import (
    ChatSFTPreprocessingConfig,
    PromptCompletionSFTPreprocessingConfig,
    SFTPreprocessingConfig,
)
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig


# =============================================================================
# Deprecated compatibility exports
# =============================================================================
FinetuningDatasetBuilder = _gpt_sft.FinetuningDatasetBuilder
FinetuningDatasetConfig = _gpt_sft.FinetuningDatasetConfig


__all__ = [
    "GPTSFTDatasetBuilder",
    "GPTSFTDatasetConfig",
    "ChatSFTPreprocessingConfig",
    "HFDatasetSourceConfig",
    "DirectHFSFTDatasetBuilder",
    "DirectHFSFTDatasetConfig",
    "EnergonDatasetBuilder",
    "EnergonDatasetConfig",
    "HFEnergonTaskEncoderConfig",
    "QwenVLEnergonTaskEncoderConfig",
    "NemotronOmniEnergonTaskEncoderConfig",
    "MockVLMSFTDatasetBuilder",
    "MockVLMSFTDatasetConfig",
    "PromptCompletionSFTPreprocessingConfig",
    "SFTPreprocessingConfig",
    "gpt_sft_train_valid_test_datasets_provider",
    "direct_hf_sft_train_valid_test_datasets_provider",
    "energon_train_valid_test_datasets_provider",
    "mock_vlm_sft_train_valid_test_datasets_provider",
    # Deprecated compatibility exports.
    "FinetuningDatasetBuilder",
    "FinetuningDatasetConfig",
]
