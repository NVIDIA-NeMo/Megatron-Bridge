# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MegatronMIMO multi-encoder data loading utilities."""

# Providers
from megatron.bridge.data.megatron_mimo.base_provider import MegatronMIMODatasetProvider
from megatron.bridge.data.megatron_mimo.collate import megatron_mimo_collate_fn
from megatron.bridge.data.megatron_mimo.dataset import MegatronMIMODataset
from megatron.bridge.data.megatron_mimo.dp_utils import (
    get_megatron_mimo_dp_info,
    get_megatron_mimo_sampling_info,
    slice_batch_for_megatron_mimo,
)
from megatron.bridge.data.megatron_mimo.hf_provider import HFMegatronMIMODatasetProvider
from megatron.bridge.data.megatron_mimo.loaders import build_megatron_mimo_data_loaders
from megatron.bridge.data.megatron_mimo.mock_provider import MockMegatronMIMOProvider


__all__ = [
    # Core
    "MegatronMIMODataset",
    "megatron_mimo_collate_fn",
    # Providers (base + implementations)
    "MegatronMIMODatasetProvider",
    "HFMegatronMIMODatasetProvider",
    "MockMegatronMIMOProvider",
    # Utilities
    "get_megatron_mimo_dp_info",
    "get_megatron_mimo_sampling_info",
    "slice_batch_for_megatron_mimo",
    "build_megatron_mimo_data_loaders",
]
