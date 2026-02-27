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
Qwen2-Audio Model for Megatron.

This module provides the Qwen2AudioModel class that combines:
- HuggingFace's audio encoder (audio_tower) for processing mel spectrograms
- HuggingFace's multimodal projector for audio-to-language projection
- Megatron's language model for text generation

Reference: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
"""

import types
from typing import TYPE_CHECKING, Optional

import torch
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


# Import HuggingFace Qwen2Audio model classes with fallback
try:
    from transformers import Qwen2AudioForConditionalGeneration
    from transformers.models.qwen2_audio.modeling_qwen2_audio import (
        Qwen2AudioEncoder,
        Qwen2AudioMultiModalProjector,
    )

    HAS_QWEN2_AUDIO = True
except ImportError:
    Qwen2AudioForConditionalGeneration = None
    Qwen2AudioEncoder = None
    Qwen2AudioMultiModalProjector = None
    HAS_QWEN2_AUDIO = False


class Qwen2AudioModel(MegatronModule):
    """
    Qwen2-Audio Model wrapper for Megatron.

    This class combines HuggingFace's audio components with Megatron's language model:
    - Audio tower (HF): Processes mel spectrograms through Whisper-like encoder
    - Multimodal projector (HF): Projects audio features to language model space
    - Language model (Megatron): Generates text conditioned on audio and text inputs

    The audio encoder forward pass uses HuggingFace implementation,
    while the language model forward pass uses Megatron's optimized implementation.

    Args:
        config (GPTModelProvider): Model provider containing configuration for language and audio modules.
        pre_process (bool, optional): Whether to construct the audio tower and projector. Default: True.
        post_process (bool, optional): Whether to apply post-processing. Default: True.
        vp_stage (Optional[int], optional): Pipeline stage for model parallelism. Default: None.

    Attributes:
        pre_process (bool): If True, enables audio and multimodal components.
        post_process (bool): If True, enables post-processing.
        vp_stage (Optional[int]): Pipeline stage for model parallelism.
        audio_tower (nn.Module): Audio encoder from HuggingFace (Whisper-like).
        multi_modal_projector (nn.Module): Projects audio features to language model space.
        language_model (nn.Module): Megatron language model.

    Forward Inputs:
        input_ids (torch.LongTensor, optional): Tokenized input ids for the language model.
        attention_mask (torch.Tensor, optional): Attention mask for the language model.
        position_ids (torch.LongTensor, optional): Position ids for the language model.
        inputs_embeds (torch.FloatTensor, optional): Precomputed input embeddings.
        input_features (torch.Tensor, optional): Mel spectrogram features for audio.
        feature_attention_mask (torch.Tensor, optional): Attention mask for audio features.
        labels (torch.Tensor, optional): Target labels for supervised training.
        runtime_gather_output (bool, optional): If True, gather outputs across pipeline stages.
        loss_mask (Tensor, optional): Mask for loss computation.

    Returns:
        Tensor: Model output (e.g., logits or loss, depending on mode).

    Note:
        - If `pre_process` is False, only the language model is constructed.
        - The audio tower and projector are only active if `pre_process` is True.
        - This class is intended for use within the Megatron-LM framework.
    """

    def __init__(
        self,
        config: GPTModelProvider,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        if pre_process:
            if not HAS_QWEN2_AUDIO:
                raise ImportError(
                    "Qwen2Audio model requires transformers with Qwen2Audio support. "
                    "Please upgrade: pip install 'transformers>=4.40.0'"
                )

            # Initialize audio tower from HuggingFace config
            # The audio_tower is a Whisper-like encoder that processes mel spectrograms
            self.audio_tower = Qwen2AudioEncoder(config.hf_config.audio_config)

            # Initialize multimodal projector from HuggingFace config
            # Projects audio encoder output dimension to language model hidden size
            self.multi_modal_projector = Qwen2AudioMultiModalProjector(config.hf_config)

            # Ensure HF audio tower params are marked for TP grad sync
            hook_hf_module_setattr_for_tp_grad_sync(self.audio_tower)
            hook_hf_module_setattr_for_tp_grad_sync(self.multi_modal_projector)

        # Initialize Megatron language model
        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Finalize grad requires these to be bound with module
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        # Monkey-patch methods from HuggingFace Qwen2AudioForConditionalGeneration
        if HAS_QWEN2_AUDIO and Qwen2AudioForConditionalGeneration is not None:
            self._merge_input_ids_with_audio_features = types.MethodType(
                Qwen2AudioForConditionalGeneration._merge_input_ids_with_audio_features, self
            )

        # Store audio token id from config
        self.audio_token_id = getattr(config, "audio_token_id", 151646)
        self.pad_token_id = getattr(config.hf_config, "pad_token_id", -1)

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional["PackedSeqParams"] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass combining HuggingFace audio encoder with Megatron language model.

        Args:
            input_ids: Tokenized input ids for the language model.
            attention_mask: Attention mask for the language model.
            position_ids: Position ids for the language model.
            inputs_embeds: Precomputed input embeddings.
            input_features: Mel spectrogram features for audio input.
            feature_attention_mask: Attention mask for audio features.
            labels: Target labels for supervised training.
            runtime_gather_output: If True, gather outputs across pipeline stages.
            loss_mask: Mask for loss computation.

        Returns:
            Tensor: Model output containing logits or loss.
        """
        if self.pre_process:
            if inputs_embeds is None:
                # Get text embeddings from Megatron language model
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # [seq_len, batch, hidden]

                # Transpose to HF format [batch, seq_len, hidden]
                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()

            if input_features is not None and input_ids.shape[1] != 1:
                # Process audio features
                target_device = self.audio_tower.conv1.weight.device

                input_features = input_features.to(target_device)
                if feature_attention_mask is not None:
                    feature_attention_mask = feature_attention_mask.to(target_device)

                # Compute audio feature lengths from attention mask
                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    feature_attention_mask.sum(-1)
                )

                batch_size, _, max_mel_seq_len = input_features.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1

                # Create attention mask for audio encoder
                seq_range = (
                    torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
                lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
                padding_mask = seq_range >= lengths_expand

                audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                    batch_size, 1, max_seq_len, max_seq_len
                )
                audio_attention_mask = audio_attention_mask_.to(
                    dtype=self.audio_tower.conv1.weight.dtype, device=target_device
                )
                audio_attention_mask[audio_attention_mask_] = float("-inf")

                # Forward through audio encoder
                audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask)
                selected_audio_feature = audio_outputs.last_hidden_state

                # Project audio features to language model dimension
                audio_features = self.multi_modal_projector(selected_audio_feature)

                # Check if we need legacy processing (non-expanded audio tokens)
                audio_tokens = input_ids == self.audio_token_id
                legacy_processing = (audio_tokens[:, :-1] & audio_tokens[:, 1:]).sum() == 0

                if legacy_processing:
                    # Use HF's merge function for legacy processing
                    inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                        audio_features, audio_output_lengths, inputs_embeds, input_ids, attention_mask, labels
                    )
                else:
                    # Modern processing: audio tokens are already expanded
                    num_audios, max_audio_tokens, embed_dim = audio_features.shape
                    audio_features_mask = torch.arange(max_audio_tokens, device=audio_output_lengths.device)[None, :]
                    audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
                    audio_features = audio_features[audio_features_mask]

                    n_audio_tokens = (input_ids == self.audio_token_id).sum().item()
                    n_audio_features = audio_features.shape[0]

                    if n_audio_tokens != n_audio_features:
                        raise ValueError(
                            f"Audio features and audio tokens do not match: tokens: {n_audio_tokens}, features {n_audio_features}"
                        )

                    special_audio_mask = (input_ids == self.audio_token_id).to(inputs_embeds.device)
                    special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
                    audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

            # Transpose back to Megatron format [seq_len, batch, hidden]
            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()

        # Forward through Megatron language model
        outputs = self.language_model.forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )

        return outputs

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_audio_model: bool,
        freeze_audio_projection: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_audio_model (bool): Freeze the audio model module (audio_tower).
            freeze_audio_projection (bool): Freeze the audio projection module (multi_modal_projector).
        """
        modules = []

        if freeze_language_model and hasattr(self, "language_model") and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_audio_model and hasattr(self, "audio_tower") and self.audio_tower is not None:
            modules.append(self.audio_tower)

        if (
            freeze_audio_projection
            and hasattr(self, "multi_modal_projector")
            and self.multi_modal_projector is not None
        ):
            modules.append(self.multi_modal_projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
