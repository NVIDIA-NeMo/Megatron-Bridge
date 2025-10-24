# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VLM dataset utilities.

Public API re-exports:
- Makers: functions to build conversation examples from HF datasets
- Providers: classes that build PyTorch datasets bound to HF processors
- Collate fns: model-specific batch builders
"""

from megatron.bridge.data.vlm_datasets.collate import (
    COLLATE_FNS,
    default_collate_fn,
    phi4_mm_collate_fn,
    qwen2_5_collate_fn,
)
from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.data.vlm_datasets.hf_dataset_makers import (
    make_cord_v2_dataset,
    make_cv17_dataset,
    make_medpix_dataset,
    make_rdr_dataset,
)
from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider
from megatron.bridge.data.vlm_datasets.mock_provider import MockVLMConversationProvider
from megatron.bridge.data.vlm_datasets.preloaded_provider import PreloadedVLMConversationProvider


__all__ = [
    # Makers
    "make_rdr_dataset",
    "make_cord_v2_dataset",
    "make_medpix_dataset",
    "make_cv17_dataset",
    # Dataset types/providers
    "VLMConversationDataset",
    "HFDatasetConversationProvider",
    "PreloadedVLMConversationProvider",
    "MockVLMConversationProvider",
    # Collation utilities
    "COLLATE_FNS",
    "default_collate_fn",
    "qwen2_5_collate_fn",
    "phi4_mm_collate_fn",
]
