# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""OmniModal multi-encoder data loading utilities."""

# Providers
from megatron.bridge.data.omni_modal.base_provider import OmniModalDatasetProvider
from megatron.bridge.data.omni_modal.collate import omni_modal_collate_fn
from megatron.bridge.data.omni_modal.dataset import OmniModalDataset
from megatron.bridge.data.omni_modal.dp_utils import (
    get_omni_modal_dp_info,
    get_omni_modal_sampling_info,
    slice_batch_for_omni_modal,
)
from megatron.bridge.data.omni_modal.hf_provider import HFOmniModalDatasetProvider
from megatron.bridge.data.omni_modal.loaders import build_omni_modal_data_loaders
from megatron.bridge.data.omni_modal.mock_provider import MockOmniModalProvider


__all__ = [
    # Core
    "OmniModalDataset",
    "omni_modal_collate_fn",
    # Providers (base + implementations)
    "OmniModalDatasetProvider",
    "HFOmniModalDatasetProvider",
    "MockOmniModalProvider",
    # Utilities
    "get_omni_modal_dp_info",
    "get_omni_modal_sampling_info",
    "slice_batch_for_omni_modal",
    "build_omni_modal_data_loaders",
]
