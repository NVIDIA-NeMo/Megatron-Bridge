# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Hugging Face conversation dataset providers, makers, and text collators."""

from typing import Any

from megatron.bridge.data.hf_datasets.conversation_dataset import ConversationDataset
from megatron.bridge.data.hf_datasets.makers import (
    make_cord_v2_dataset,
    make_cv17_dataset,
    make_default_audio_dataset,
    make_gsm8k_dataset,
    make_llava_video_178k_dataset,
    make_medpix_dataset,
    make_openmathinstruct2_dataset,
    make_openmathinstruct2_thinking_dataset,
    make_raven_dataset,
    make_rdr_dataset,
    make_squad_dataset,
    make_text_chat_dataset,
    make_valor32k_avqa_dataset,
)
from megatron.bridge.data.hf_datasets.text_collate import text_chat_collate_fn


# =============================================================================
# Deprecated provider compatibility exports
# =============================================================================
def __getattr__(name: str) -> Any:
    """Load deprecated provider exports lazily to avoid source/builder import cycles."""
    if name == "HFConversationDatasetProvider":
        from megatron.bridge.data.hf_datasets.provider import HFConversationDatasetProvider

        return HFConversationDatasetProvider
    if name == "HFTextSFTDatasetProvider":
        from megatron.bridge.data.hf_datasets.text_sft_provider import HFTextSFTDatasetProvider

        return HFTextSFTDatasetProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConversationDataset",
    "make_cord_v2_dataset",
    "make_cv17_dataset",
    "make_default_audio_dataset",
    "make_gsm8k_dataset",
    "make_llava_video_178k_dataset",
    "make_medpix_dataset",
    "make_openmathinstruct2_dataset",
    "make_openmathinstruct2_thinking_dataset",
    "make_raven_dataset",
    "make_rdr_dataset",
    "make_squad_dataset",
    "make_text_chat_dataset",
    "make_valor32k_avqa_dataset",
    "text_chat_collate_fn",
    # Deprecated provider compatibility exports.
    "HFConversationDatasetProvider",
    "HFTextSFTDatasetProvider",
]
