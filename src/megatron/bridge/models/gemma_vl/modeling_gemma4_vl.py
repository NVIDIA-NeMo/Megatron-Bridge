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

"""
Gemma 4 Vision-Language model.

Vision-Language model (Gemma4VLModel):
- HuggingFace Gemma4 vision tower + multimodal embedder
- Megatron-Core GPT language model (Dense or MoE)

Text-only (Dense/MoE) layer specs and providers live in:
- megatron.bridge.models.gemma.modeling_gemma4
- megatron.bridge.models.gemma.gemma4_provider
"""

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from torch import Tensor
from transformers import AutoModel

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import (
    hook_hf_module_setattr_for_tp_grad_sync,
    slice_batch_for_context_parallel,
)


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


def _keep_hf_precision_buffers_in_fp32(module: nn.Module) -> None:
    """Keep HF non-persistent precision-sensitive buffers in fp32 after casts.

    HF Gemma4 registers buffers such as vision RoPE ``inv_freq`` and audio
    ``inv_timescales`` as non-persistent fp32 buffers. A plain
    ``module.to(dtype=bf16)`` casts them to bf16, but
    ``from_pretrained(torch_dtype=bf16)`` keeps them in fp32.
    """

    for submodule in module.modules():
        if "inv_freq" in submodule._buffers and hasattr(submodule, "compute_default_rope_parameters"):
            device = submodule._buffers["inv_freq"].device
            rope_type = getattr(submodule, "rope_type", "default")
            if isinstance(rope_type, str):
                if rope_type == "default":
                    inv_freq, attention_scaling = submodule.compute_default_rope_parameters(
                        submodule.config,
                        device=device,
                    )
                else:
                    from transformers.models.gemma4.modeling_gemma4 import ROPE_INIT_FUNCTIONS

                    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](
                        submodule.config,
                        device=device,
                    )
                submodule._buffers["inv_freq"] = inv_freq.float()
                if "original_inv_freq" in submodule._buffers:
                    submodule._buffers["original_inv_freq"] = inv_freq.clone().float()
                submodule.attention_scaling = attention_scaling

        if "inv_timescales" in submodule._buffers and hasattr(submodule, "hidden_size"):
            device = submodule._buffers["inv_timescales"].device
            min_timescale = 1.0
            max_timescale = 10000.0
            num_timescales = submodule.hidden_size // 2
            log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
            inv_timescales = min_timescale * torch.exp(
                torch.arange(num_timescales, device=device, dtype=torch.float32) * -log_timescale_increment
            )
            submodule._buffers["inv_timescales"] = inv_timescales.unsqueeze(0).unsqueeze(0)

        for name in ("softcap",):
            buffer = submodule._buffers.get(name)
            if torch.is_tensor(buffer) and buffer.is_floating_point():
                submodule._buffers[name] = buffer.float()


class _SimpleVisionEmbedder(nn.Module):
    """Fallback Gemma4 vision projector for transformers versions without the HF class."""

    def __init__(self, vision_hidden: int, text_hidden: int, eps: float):
        super().__init__()
        self.embedding_projection = nn.Linear(vision_hidden, text_hidden, bias=False)
        self._eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self._eps).sqrt()
        x = (x.float() / rms).to(x.dtype)
        return self.embedding_projection(x)


class _SimpleAudioEmbedder(nn.Module):
    """Fallback Gemma4 audio projector for transformers versions without the HF class."""

    def __init__(self, audio_proj_dim: int, text_hidden: int, eps: float):
        super().__init__()
        self.embedding_projection = nn.Linear(audio_proj_dim, text_hidden, bias=False)
        self._eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self._eps).sqrt()
        x = (x.float() / rms).to(x.dtype)
        return self.embedding_projection(x)


# ---------------------------------------------------------------------------
# Gemma 4 Vision-Language model
# ---------------------------------------------------------------------------


class Gemma4VLModel(MegatronModule):
    """Gemma 4 Vision-Language-Audio model.

    Wraps HF vision/audio towers + multimodal projectors with a Megatron-Core
    GPT language model (Dense or MoE).

    Forward flow:
        1. Embed text tokens via language model embedding
        2. If pixel_values: vision_tower → embed_vision → scatter at image_token_id positions
        3. If input_features: audio_tower → embed_audio → scatter at audio_token_id positions
        4. Forward through language model decoder
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
            # Vision encoder
            self.vision_tower = AutoModel.from_config(config.vision_config)
            self._init_embed_vision(config)
            target_dtype = getattr(config, "params_dtype", None)
            if target_dtype is not None:
                self.vision_tower.to(dtype=target_dtype)
                _keep_hf_precision_buffers_in_fp32(self.vision_tower)
                self.embed_vision.to(dtype=target_dtype)
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)

            # Audio encoder (optional — only when audio_config is provided)
            if getattr(config, "audio_config", None) is not None:
                self.audio_tower = AutoModel.from_config(config.audio_config)
                self._init_embed_audio(config)
                if target_dtype is not None:
                    self.audio_tower.to(dtype=target_dtype)
                    _keep_hf_precision_buffers_in_fp32(self.audio_tower)
                    self.embed_audio.to(dtype=target_dtype)
                hook_hf_module_setattr_for_tp_grad_sync(self.audio_tower)

        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

    def _init_embed_vision(self, config):
        """Initialize the multimodal embedder (vision → language projection)."""
        try:
            from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder

            self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
        except (ImportError, AttributeError):
            vision_hidden = config.vision_config.hidden_size
            text_hidden = config.text_config.hidden_size
            eps = config.vision_config.rms_norm_eps
            self.embed_vision = _SimpleVisionEmbedder(vision_hidden, text_hidden, eps)

    def _init_embed_audio(self, config):
        """Initialize the audio projector (audio encoder output → language space).

        Gemma4's embed_audio mirrors embed_vision: parameter-free RMSNorm followed
        by a linear projection from audio_config.output_proj_dims to text hidden_size.
        """
        try:
            from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioEmbedder

            self.embed_audio = Gemma4AudioEmbedder(config.audio_config, config.text_config)
        except (ImportError, AttributeError):
            audio_proj_dim = config.audio_config.output_proj_dims
            text_hidden = config.text_config.hidden_size
            eps = getattr(config.audio_config, "rms_norm_eps", 1e-6)
            self.embed_audio = _SimpleAudioEmbedder(audio_proj_dim, text_hidden, eps)

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor)

    def get_image_features(self, pixel_values, image_position_ids=None, **kwargs):
        """Extract and project image features using HF vision tower + embedder."""
        _keep_hf_precision_buffers_in_fp32(self.vision_tower)
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
            **kwargs,
        )
        return self.embed_vision(vision_outputs.last_hidden_state)

    def get_audio_features(self, input_features, **kwargs):
        """Extract and project audio features using HF audio tower + embedder."""
        _keep_hf_precision_buffers_in_fp32(self.audio_tower)
        audio_outputs = self.audio_tower(input_features=input_features, **kwargs)
        return self.embed_audio(audio_outputs.last_hidden_state)

    def _scatter_modality_features(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        features: torch.Tensor,
        token_id: int,
        modality_name: str,
    ) -> torch.Tensor:
        """Scatter projected modality features into the embedding at special token positions."""
        mask = (input_ids == token_id).unsqueeze(-1)
        mask = mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        n_slots = mask[:, :, 0].sum().item()
        n_feats = features.numel() // inputs_embeds.shape[-1]
        if n_slots != n_feats:
            raise ValueError(
                f"{modality_name} token count mismatch: "
                f"{n_slots} {modality_name}_token_id positions vs "
                f"{n_feats} tokens from {modality_name} encoder."
            )
        return inputs_embeds.masked_scatter(mask, features.to(inputs_embeds.device, inputs_embeds.dtype))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional["PackedSeqParams"] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """Forward pass combining HF vision/audio encoders with Megatron language model."""
        lm_input_ids = input_ids
        if self.pre_process:
            if input_ids is not None:
                multimodal_mask = input_ids == self.config.image_token_id
                if hasattr(self.config, "audio_token_id"):
                    multimodal_mask = torch.logical_or(
                        multimodal_mask,
                        input_ids == self.config.audio_token_id,
                    )
                if multimodal_mask.any():
                    lm_input_ids = input_ids.clone()
                    lm_input_ids[multimodal_mask] = self.config.text_config.pad_token_id

            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(input_ids=lm_input_ids, position_ids=None)
                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [B, S, H]
                if getattr(self.language_model.config, "scale_embeddings_by_hidden_size", False):
                    inputs_embeds = inputs_embeds * (self.language_model.config.hidden_size**0.5)

            # Vision: scatter image features at image_token_id positions
            if pixel_values is not None:
                image_features = self.get_image_features(pixel_values, image_position_ids=image_position_ids)
                inputs_embeds = self._scatter_modality_features(
                    inputs_embeds,
                    input_ids,
                    image_features,
                    self.config.image_token_id,
                    "image",
                )

            # Audio: scatter audio features at audio_token_id positions
            if input_features is not None and hasattr(self, "audio_tower"):
                audio_features = self.get_audio_features(input_features)
                inputs_embeds = self._scatter_modality_features(
                    inputs_embeds,
                    input_ids,
                    audio_features,
                    self.config.audio_token_id,
                    "audio",
                )

            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [S, B, H]

        attention_mask = self._compute_attention_mask(input_ids) if input_ids is not None else attention_mask

        pg_coll = getattr(self.config, "_pg_collection", None)
        if pg_coll is not None:
            inputs_embeds, labels, loss_mask, position_ids, attention_mask = slice_batch_for_context_parallel(
                inputs_embeds=inputs_embeds,
                labels=labels,
                loss_mask=loss_mask,
                position_ids=position_ids,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
                pg_collection=pg_coll,
            )

        if self.config.sequence_parallel and inputs_embeds is not None:
            tp_group = self.config._pg_collection.tp if self.config._pg_collection is not None else None
            inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds, group=tp_group)

        outputs = self.language_model.forward(
            input_ids=lm_input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )
        # Return just logits in inference mode (no training objective supplied).
        # Training callers always provide labels or loss_mask and receive (logits, loss_mask).
        if labels is None and loss_mask is None:
            return outputs
        return (outputs, loss_mask)

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
        freeze_audio_model: bool = False,
        freeze_audio_projection: bool = False,
    ):
        """Freeze model modules for fine-tuning."""
        pairs = [
            (freeze_language_model, "language_model"),
            (freeze_vision_model, "vision_tower"),
            (freeze_vision_projection, "embed_vision"),
            (freeze_audio_model, "audio_tower"),
            (freeze_audio_projection, "embed_audio"),
        ]
        for should_freeze, attr in pairs:
            if should_freeze and hasattr(self, attr):
                for param in getattr(self, attr).parameters():
                    param.requires_grad = False

    def _compute_attention_mask(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute HF-style attention masks for full and sliding Gemma4 layers."""
        if not self.pre_process:
            return None
        batch_size, seq_len = input_ids.shape
        causal_mask = torch.tril(
            torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool, device=input_ids.device)
        )

        def _bidirectional_block_mask(token_mask: torch.Tensor) -> torch.Tensor:
            padded = F.pad(token_mask, (1, 0), value=0)
            boundary = padded[:, 1:] > padded[:, :-1]
            block_ids = token_mask * torch.cumsum(boundary, dim=-1)
            return torch.logical_and(
                block_ids[:, None, :] == block_ids.unsqueeze(-1),
                block_ids.unsqueeze(-1) > 0,
            )

        bidir = _bidirectional_block_mask(input_ids == self.config.image_token_id)

        # blocked[b, 0, i, j] = True where attention is prevented:
        # causal blocks j > i; image tokens within the same block override this
        # (bidirectional). Audio tokens intentionally follow the causal text mask.
        return ~torch.logical_or(causal_mask, bidir.unsqueeze(1))
