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

"""Native Omni language/head integration with the published HF perception modules."""

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from torch.nn import functional as F
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_outputs import SequenceClassifierOutput

from megatron.bridge.models.d1.modeling_d1.backbone import D1Backbone
from megatron.bridge.models.d1.modeling_d1.decision_head import D1DecisionHead


OMNI_REPOSITORY = "LiquidAI/d1-omni-600M"
OMNI_REVISION = "02b55d7076f15129e59ab3f94783f32c4b088674"


class D1OmniModel(MegatronModule):
    """Trainable native decision model with shared text, image and audio paths.

    Args:
        config: Native language transformer configuration.
        vocab_size: Published unpadded vocabulary size.
        layer_types: Published language-layer schedule.
        rotary_base: Published rotary base.
        vision_config: Published HF vision configuration.
        audio_config: Published HF audio configuration.
        projector_hidden_size: Vision projector intermediate width.
        head_layers: Number of marker-head transformer layers.
        max_length: Maximum combined prefix and text length.
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        vocab_size: int,
        layer_types: tuple[str, ...],
        rotary_base: float,
        vision_config: dict[str, object],
        audio_config: dict[str, object],
        projector_hidden_size: int,
        head_layers: int,
        max_length: int,
    ) -> None:
        super().__init__(config)
        if max_length <= 0 or projector_hidden_size <= 0 or not vision_config or not audio_config:
            raise ValueError("Omni requires positive lengths and widths and both perception configurations")
        self.max_length = max_length
        self.pre_process = True
        self.post_process = True
        self.encoder = D1Backbone(
            config, vocab_size=vocab_size, layer_types=layer_types, causal=False, rotary_base=rotary_base
        )
        self.head = D1DecisionHead(config, layers=head_layers)
        vision_class = get_class_from_dynamic_module(
            "vision.Vision", OMNI_REPOSITORY, revision=OMNI_REVISION, code_revision=OMNI_REVISION
        )
        audio_class = get_class_from_dynamic_module(
            "audio.Audio", OMNI_REPOSITORY, revision=OMNI_REVISION, code_revision=OMNI_REVISION
        )
        device = self.encoder.embed_tokens.weight.device
        self.vision = vision_class(vision_config, projector_hidden_size, config.hidden_size).to(
            device=device, dtype=config.params_dtype
        )
        self.audio = audio_class(audio_config, config.hidden_size).to(device=device, dtype=config.params_dtype)
        # Perception modules are replicated in the currently supported TP=1 path.
        for parameter in list(self.vision.parameters()) + list(self.audio.parameters()):
            parameter.allreduce = True
            parameter.tensor_model_parallel = False

    def encode_images(self, images: list[object]) -> torch.Tensor:
        """Return a media prefix [1, prefix_length, width] for ordered PIL images."""
        return self.vision(images)

    def encode_audio(self, audio: object) -> torch.Tensor:
        """Return a media prefix [1, prefix_length, width] for mono 16 kHz audio."""
        return self.audio(audio)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        media_prefix: torch.Tensor | None = None,
        media_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack variable-length prefixes, then return only normalized text states.

        Args:
            input_ids: Long text IDs [batch, padded_text_length].
            attention_mask: Boolean real-token mask, with right padding.
            media_prefix: Optional padded media embeddings [batch, prefix_length, width].
            media_lengths: Optional long prefix lengths [batch].

        Returns:
            Text states [batch, longest_real_text, width] and their boolean mask.
        """
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be long with shape [batch, text_length]")
        if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be boolean with the input_ids shape")
        if attention_mask.device != input_ids.device:
            raise ValueError("text IDs and their mask must share a device")
        batch, length = input_ids.shape
        lengths = attention_mask.sum(dim=1)
        positions = torch.arange(length, device=input_ids.device)
        if torch.any(lengths == 0) or not torch.equal(attention_mask, positions[None, :] < lengths[:, None]):
            raise ValueError("each row requires nonempty, right-padded text")
        embeddings = self.encoder.embed_tokens(input_ids)
        if media_lengths is not None:
            if media_lengths.shape != (batch,) or media_lengths.dtype != torch.long:
                raise ValueError("media_lengths must be long with shape [batch]")
            if media_lengths.device != input_ids.device:
                raise ValueError("prefix lengths and text IDs must share a device")
        if media_prefix is None:
            if media_lengths is not None and torch.any(media_lengths != 0):
                raise ValueError("nonzero media_lengths require media_prefix")
            media_lengths = torch.zeros(batch, device=input_ids.device, dtype=torch.long)
            sequences = [embeddings[row, :count] for row, count in enumerate(lengths.tolist())]
        else:
            if media_lengths is None:
                raise ValueError("media_prefix requires explicit media_lengths")
            if (
                media_prefix.ndim != 3
                or media_prefix.shape[0] != batch
                or media_prefix.shape[2] != self.config.hidden_size
            ):
                raise ValueError("media_prefix must have shape [batch, prefix_length, hidden_size]")
            if media_prefix.device != input_ids.device:
                raise ValueError("media embeddings and text IDs must share a device")
            if torch.any((media_lengths < 0) | (media_lengths > media_prefix.shape[1])):
                raise ValueError("media lengths must index the supplied prefixes")
            sequences = [
                torch.cat((media_prefix[row, :prefix].to(embeddings.dtype), embeddings[row, :count]))
                for row, (prefix, count) in enumerate(zip(media_lengths.tolist(), lengths.tolist()))
            ]
        total_lengths = lengths + media_lengths
        if torch.any(total_lengths > self.max_length):
            raise ValueError("combined prefix and text exceed the model maximum length")
        packed = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        valid = torch.arange(packed.shape[1], device=input_ids.device)[None, :] < total_lengths[:, None]
        encoded = self.encoder(packed, valid_tokens=valid, media_lengths=media_lengths)
        text = nn.utils.rnn.pad_sequence(
            [
                encoded[row, prefix : prefix + count]
                for row, (prefix, count) in enumerate(zip(media_lengths.tolist(), lengths.tolist()))
            ],
            batch_first=True,
        )
        valid_text = torch.arange(text.shape[1], device=input_ids.device)[None, :] < lengths[:, None]
        return text, valid_text

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        marker_positions: torch.Tensor,
        marker_mask: torch.Tensor,
        question_types: torch.Tensor,
        media_prefix: torch.Tensor | None = None,
        media_lengths: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        """Return trainable option logits, optional CE loss and normalized text states.

        Args:
            input_ids: Long text IDs [batch, padded_text_length].
            attention_mask: Boolean real-token mask [batch, padded_text_length].
            marker_positions: Long option-marker indices relative to text [batch, options].
            marker_mask: Boolean real-option mask [batch, options].
            question_types: Long choice=0, score=1, noul=2 IDs [batch].
            media_prefix: Optional media prefix embeddings [batch, padded_prefix_length, width].
            media_lengths: Optional long prefix lengths [batch].
            labels: Optional long internal option indices [batch].

        Returns:
            Native differentiable logits and loss; calibration and public noul
            ordering remain caller responsibilities.
        """
        text, valid = self.encode_text(
            input_ids, attention_mask=attention_mask, media_prefix=media_prefix, media_lengths=media_lengths
        )
        logits = self.head(
            text,
            valid_tokens=valid,
            marker_positions=marker_positions,
            marker_mask=marker_mask,
            question_types=question_types,
        )
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape[:1] or labels.dtype != torch.long or labels.device != logits.device:
                raise ValueError("labels must be long with shape [batch] on the logits device")
            if torch.any((labels < 0) | (labels >= logits.shape[1])):
                raise ValueError("labels must index the declared options")
            if torch.any(~marker_mask.gather(1, labels[:, None])):
                raise ValueError("labels cannot select absent options")
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=(text,))
