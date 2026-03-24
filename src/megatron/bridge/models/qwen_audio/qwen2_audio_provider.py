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
Qwen2-Audio Model Provider configurations for Megatron-Core.

This module provides configuration classes for Qwen2-Audio models,
compatible with HuggingFace's Qwen2-Audio model configurations.

Reference: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct

Qwen2-Audio Key Features:
- Audio-language capabilities with separate language model and audio encoder
- Whisper-like audio encoder for processing mel spectrograms
- Based on Qwen2 language model architecture
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from megatron.core.models.gpt import GPTModel as MCoreGPTModel

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen.qwen_provider import Qwen2ModelProvider


if TYPE_CHECKING:
    from megatron.bridge.models.qwen_audio.modeling_qwen2_audio import Qwen2AudioModel


# =============================================================================
# Qwen2-Audio Model Provider
# =============================================================================


@dataclass
class Qwen2AudioModelProvider(Qwen2ModelProvider):
    """
    Base model provider for Qwen2-Audio Models.

    Qwen2-Audio is a multimodal model combining a Whisper-like audio encoder
    with a Qwen2 language model for audio understanding tasks.

    Reference:
    - https://huggingface.co/Qwen/Qwen2-Audio-7B
    - https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct

    Key Features:
    - Audio encoder based on Whisper architecture
    - Supports variable-length audio inputs via mel spectrograms
    - Multi-turn conversation with audio context
    """

    # Audio-Language models shouldn't scatter embeddings across sequence parallel regions
    # because audio embeddings are inserted into language embeddings
    scatter_embedding_sequence_parallel: bool = False

    # HuggingFace config containing audio_config and text_config
    hf_config: Optional[Any] = None

    # Audio-specific token IDs (defaults from Qwen2-Audio)
    audio_token_id: int = 151646  # <|AUDIO|> token

    # Token IDs
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643

    # Freeze options for fine-tuning
    freeze_language_model: bool = False
    freeze_audio_model: bool = False
    freeze_audio_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "Qwen2AudioModel":
        """
        Provide a Qwen2AudioModel instance with audio and language components.

        Args:
            pre_process: Whether this is the first stage in pipeline parallelism
            post_process: Whether this is the last stage in pipeline parallelism
            vp_stage: Virtual pipeline stage number

        Returns:
            Qwen2AudioModel instance with HF audio encoder and Megatron language model
        """
        from megatron.bridge.models.qwen_audio.modeling_qwen2_audio import Qwen2AudioModel

        model = Qwen2AudioModel(
            config=self,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

        # Apply freeze options if any are enabled for fine-tuning
        if self.freeze_language_model or self.freeze_audio_model or self.freeze_audio_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_audio_model=self.freeze_audio_model,
                freeze_audio_projection=self.freeze_audio_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """
        Provide just the language model component without audio.

        Args:
            pre_process: Whether this is the first stage in pipeline parallelism
            post_process: Whether this is the last stage in pipeline parallelism
            vp_stage: Virtual pipeline stage number

        Returns:
            MCoreGPTModel instance (language model only)
        """
        return GPTModelProvider.provide(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
