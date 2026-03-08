# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MIMO multi-encoder data loading utilities."""

# Providers
from megatron.bridge.data.mimo.base_provider import MimoDatasetProvider
from megatron.bridge.data.mimo.collate import mimo_collate_fn
from megatron.bridge.data.mimo.dataset import MimoDataset
from megatron.bridge.data.mimo.dp_utils import get_mimo_dp_info
from megatron.bridge.data.mimo.hf_provider import HFMimoDatasetProvider
from megatron.bridge.data.mimo.loaders import build_mimo_data_loaders
from megatron.bridge.data.mimo.mock_provider import MockMimoProvider


__all__ = [
    # Core
    "MimoDataset",
    "mimo_collate_fn",
    # Providers (base + implementations)
    "MimoDatasetProvider",
    "HFMimoDatasetProvider",
    "MockMimoProvider",
    # Utilities
    "get_mimo_dp_info",
    "build_mimo_data_loaders",
]
