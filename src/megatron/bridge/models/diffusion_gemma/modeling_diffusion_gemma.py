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

"""Megatron DiffusionGemma encoder/decoder model."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from torch import nn
from torch.nn import functional as F

from megatron.bridge.models.gemma.gemma3_provider import Gemma3LanguageModelEmbedding
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel


class DiffusionGemmaLanguageModelEmbedding(Gemma3LanguageModelEmbedding):
    """Gemma token embedding with HF DiffusionGemma's scale and padding semantics."""

    padding_idx: Optional[int] = None

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, tokentype_ids: int = None) -> torch.Tensor:
        embeddings = super(Gemma3LanguageModelEmbedding, self).forward(input_ids, position_ids, tokentype_ids)
        if self.padding_idx is not None:
            # HF uses nn.Embedding(padding_idx=...), so pad lookups never update
            # the pad row. The tied output-head contribution is unaffected.
            is_padding = (input_ids == self.padding_idx).transpose(0, 1)[..., None]
            embeddings = torch.where(is_padding, embeddings.detach(), embeddings)
        scale = torch.tensor(self.config.hidden_size**0.5, device=embeddings.device, dtype=embeddings.dtype)
        return embeddings * scale


class DiffusionGemmaRMSNorm(nn.Module):
    """RMSNorm with DiffusionGemma's FP32 normalization and optional scale."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states.float()
        output = output * torch.pow(output.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            output = output * self.weight.float()
        return output.type_as(hidden_states)


class DiffusionGemmaSelfConditioning(nn.Module):
    """Decoder self-conditioning block from the pinned HF implementation."""

    def __init__(self, hidden_size: int, intermediate_size: int, eps: float) -> None:
        super().__init__()
        self.pre_norm = DiffusionGemmaRMSNorm(hidden_size, eps=eps)
        self.post_norm = DiffusionGemmaRMSNorm(hidden_size, eps=eps, with_scale=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, inputs_embeds: torch.Tensor, self_conditioning_signal: torch.Tensor) -> torch.Tensor:
        normed = self.pre_norm(self_conditioning_signal)
        signal = self.down_proj(F.gelu(self.gate_proj(normed), approximate="tanh") * self.up_proj(normed))
        return self.post_norm(inputs_embeds + signal)


@dataclass
class DiffusionGemmaEncoderOutput:
    """Encoder outputs and optional per-layer attention K/V."""

    last_hidden_state: torch.Tensor
    key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None


@dataclass
class DiffusionGemmaBlockDiffusionOutput:
    """Decoder logits plus the encoder outputs used to produce them."""

    logits: torch.Tensor
    encoder_last_hidden_state: torch.Tensor
    encoder_loss: torch.Tensor | None = None


class DiffusionGemmaModel(Gemma4VLModel):
    """Gemma 4 VL multimodal encoder plus DiffusionGemma decoder-only modules.

    The Megatron language model is the single tied text stack. The HF checkpoint
    stores its tied tensors under ``model.decoder.*``; the encoder reuses them.
    """

    def __init__(self, config, pre_process: bool = True, post_process: bool = True, vp_stage: Optional[int] = None):
        super().__init__(config=config, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        self._encoder_padding_mask: Optional[torch.Tensor] = None
        embedding = getattr(self.language_model, "embedding", None)
        if isinstance(embedding, Gemma3LanguageModelEmbedding):
            embedding.__class__ = DiffusionGemmaLanguageModelEmbedding
            embedding.padding_idx = getattr(config.text_config, "pad_token_id", None)
        if pre_process:
            # DiffusionGemma's pinned HF reference uses eager vision attention
            # for parity with the model's custom vision masking/cache behavior.
            for module in self.vision_tower.modules():
                module_config = getattr(module, "config", None)
                if module_config is not None and hasattr(module_config, "_attn_implementation"):
                    module_config._attn_implementation = "eager"
            self.self_conditioning = DiffusionGemmaSelfConditioning(
                config.text_config.hidden_size,
                config.text_config.intermediate_size,
                config.text_config.rms_norm_eps,
            )
            target_dtype = getattr(config, "params_dtype", None)
            if target_dtype is not None:
                self.self_conditioning.to(dtype=target_dtype)

    @contextmanager
    def _without_output_head(self):
        original = self.language_model.post_process
        self.language_model.post_process = False
        try:
            yield
        finally:
            self.language_model.post_process = original

    def _tp_group(self):
        pg_collection = getattr(self.config, "_pg_collection", None)
        return pg_collection.tp if pg_collection is not None else None

    def _compute_attention_mask(
        self, input_ids: torch.Tensor, mm_token_type_ids: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        blocked = super()._compute_attention_mask(input_ids, mm_token_type_ids=mm_token_type_ids)
        padding = self._encoder_padding_mask
        if blocked is None or padding is None:
            return blocked
        valid = padding.to(device=blocked.device, dtype=torch.bool)
        if valid.shape != input_ids.shape:
            raise ValueError(
                f"encoder attention_mask shape {tuple(valid.shape)} != input_ids {tuple(input_ids.shape)}"
            )
        if bool(valid.all()):
            return blocked
        blocked = blocked | ~valid[:, None, None, :]
        # Keep pad-query rows finite. They are never attended by real tokens and
        # are masked out again as decoder K/V prefix positions.
        diagonal = torch.eye(input_ids.shape[1], dtype=torch.bool, device=blocked.device)[None, None]
        return blocked & ~(diagonal & ~valid[:, None, :, None])

    def encode(
        self,
        *,
        return_key_values: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> DiffusionGemmaEncoderOutput:
        """Run the multimodal encoder without materializing vocabulary logits."""
        if kwargs.get("labels") is not None or kwargs.get("loss_mask") is not None:
            raise ValueError("encode() does not accept labels or loss_mask")
        key_values = [None] * len(self.language_model.decoder.layers) if return_key_values else None
        hooks = []
        if return_key_values:
            for index, layer in enumerate(self.language_model.decoder.layers):

                def save_key_values(_module, args, call_kwargs, index=index):
                    values = _attention_qkv(args, call_kwargs)
                    key_values[index] = (values[1], values[2])

                hooks.append(
                    layer.self_attention.core_attention.register_forward_pre_hook(save_key_values, with_kwargs=True)
                )
        try:
            self._encoder_padding_mask = attention_mask
            with self._without_output_head():
                hidden = super().forward(**kwargs)
            if self.config.sequence_parallel:
                hidden = gather_from_sequence_parallel_region(
                    hidden, tensor_parallel_output_grad=False, group=self._tp_group()
                )
        finally:
            self._encoder_padding_mask = None
            for hook in hooks:
                hook.remove()
        if return_key_values and any(value is None for value in key_values):
            raise RuntimeError("Failed to capture encoder K/V for every DiffusionGemma layer")
        return DiffusionGemmaEncoderOutput(
            last_hidden_state=hidden.transpose(0, 1).contiguous(), key_values=key_values
        )

    def _decoder_embeddings(
        self,
        decoder_input_ids: torch.Tensor,
        self_conditioning_logits: Optional[torch.Tensor],
        self_conditioning_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        embedding = self.language_model.embedding
        inputs = embedding(input_ids=decoder_input_ids, position_ids=None).transpose(0, 1).contiguous()
        if self_conditioning_logits is None:
            signal = torch.zeros_like(inputs)
        else:
            word_embeddings = embedding.word_embeddings
            weight = word_embeddings.weight
            vocab_size = int(getattr(word_embeddings, "num_embeddings", weight.shape[0]))
            probabilities = self_conditioning_logits[..., :vocab_size].softmax(dim=-1, dtype=torch.float32)
            start = int(getattr(word_embeddings, "vocab_start_index", 0))
            probabilities = probabilities[..., start : start + weight.shape[0]].to(weight.dtype)
            scale = torch.tensor(self.config.hidden_size**0.5, device=inputs.device, dtype=inputs.dtype)
            signal = torch.matmul(probabilities, weight) * scale
            if weight.shape[0] != vocab_size:
                signal = reduce_from_tensor_model_parallel_region(
                    signal, group=getattr(word_embeddings, "tp_group", None)
                )
            if self_conditioning_mask is not None:
                signal = signal * self_conditioning_mask.to(signal.dtype)[:, None, None]
        return self.self_conditioning(inputs, signal).transpose(0, 1).contiguous()

    @contextmanager
    def _decoder_attention(self, key_values, encoder_attention_mask: Optional[torch.Tensor], canvas_length: int):
        hooks = []
        for index, layer in enumerate(self.language_model.decoder.layers):
            core = layer.self_attention.core_attention
            old_image_mask = getattr(core, "_image_bidirectional_attention", None)
            if old_image_mask is not None:
                core._image_bidirectional_attention = False
            encoder_key, encoder_value = key_values[index]
            window = getattr(core, "_image_attention_window", None)
            is_local = window is not None

            def replace_qkv(
                _module,
                args,
                call_kwargs,
                encoder_key=encoder_key,
                encoder_value=encoder_value,
                is_local=is_local,
                window=window,
            ):
                query, key, value = _attention_qkv(args, call_kwargs)
                prefix_key, prefix_value = encoder_key, encoder_value
                prefix_mask = encoder_attention_mask
                if is_local and window > 1 and prefix_key.shape[0] > window - 1:
                    prefix_key = prefix_key[-(window - 1) :]
                    prefix_value = prefix_value[-(window - 1) :]
                    if prefix_mask is not None:
                        prefix_mask = prefix_mask[:, -(window - 1) :]
                key = torch.cat((prefix_key, key), dim=0)
                value = torch.cat((prefix_value, value), dim=0)
                batch = query.shape[1]
                if prefix_mask is None:
                    prefix_mask = torch.ones((batch, prefix_key.shape[0]), dtype=torch.bool, device=query.device)
                allowed = torch.cat(
                    (prefix_mask.bool(), torch.ones((batch, canvas_length), dtype=torch.bool, device=query.device)),
                    dim=1,
                )
                blocked = ~allowed[:, None, None, :].expand(batch, 1, query.shape[0], key.shape[0])
                return _replace_attention_args(args, call_kwargs, query, key, value, blocked)

            hooks.append((core, old_image_mask, core.register_forward_pre_hook(replace_qkv, with_kwargs=True)))
        try:
            yield
        finally:
            for core, old_image_mask, hook in hooks:
                hook.remove()
                if old_image_mask is not None:
                    core._image_bidirectional_attention = old_image_mask

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        key_values: list[tuple[torch.Tensor, torch.Tensor]],
        encoder_length: int,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        self_conditioning_logits: Optional[torch.Tensor] = None,
        self_conditioning_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise one canvas from captured encoder K/V and return logits [B, S, V]."""
        hidden = self._decoder_embeddings(decoder_input_ids, self_conditioning_logits, self_conditioning_mask)
        if self.config.sequence_parallel:
            hidden = scatter_to_sequence_parallel_region(hidden, group=self._tp_group())
        canvas_length = decoder_input_ids.shape[1]
        rotary = self.language_model.rotary_pos_emb(canvas_length, offset=encoder_length)
        with self._decoder_attention(key_values, encoder_attention_mask, canvas_length):
            hidden = self.language_model.decoder(hidden_states=hidden, attention_mask=None, rotary_pos_emb=rotary)
        weight = self.language_model.shared_embedding_or_output_weight()
        logits, _ = self.language_model.output_layer(hidden, weight=weight, runtime_gather_output=True)
        return logits.transpose(0, 1).float().contiguous()

    def forward(
        self,
        input_ids: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        self_conditioning_logits: Optional[torch.Tensor] = None,
        self_conditioning_mask: Optional[torch.Tensor] = None,
        encoder_target_ids: Optional[torch.Tensor] = None,
        encoder_target_mask: Optional[torch.Tensor] = None,
    ) -> DiffusionGemmaBlockDiffusionOutput:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None].expand_as(input_ids)
        encoded = self.encode(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            mm_token_type_ids=mm_token_type_ids,
            attention_mask=encoder_attention_mask,
            return_key_values=True,
        )
        logits = self.decode(
            decoder_input_ids,
            encoded.key_values,
            encoder_length=input_ids.shape[1],
            encoder_attention_mask=encoder_attention_mask,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
        )
        encoder_loss = None
        if encoder_target_ids is not None:
            if encoder_target_mask is None:
                raise ValueError("encoder_target_mask is required with encoder_target_ids")
            prompt_mask = encoder_attention_mask
            if prompt_mask is None:
                prompt_mask = torch.ones_like(input_ids)
            clean_ids = torch.cat((input_ids, encoder_target_ids), dim=1)
            clean_mask = torch.cat((prompt_mask, encoder_target_mask.to(prompt_mask.dtype)), dim=1)
            clean_mm = None
            if mm_token_type_ids is not None:
                clean_mm = torch.cat((mm_token_type_ids, torch.zeros_like(encoder_target_ids)), dim=1)
            clean_hidden = (
                self.encode(
                    input_ids=clean_ids,
                    position_ids=torch.arange(clean_ids.shape[1], device=clean_ids.device)[None].expand_as(clean_ids),
                    attention_mask=clean_mask,
                    pixel_values=pixel_values,
                    image_position_ids=image_position_ids,
                    mm_token_type_ids=clean_mm,
                )
                .last_hidden_state.transpose(0, 1)
                .contiguous()
            )
            if self.config.sequence_parallel:
                clean_hidden = scatter_to_sequence_parallel_region(clean_hidden, group=self._tp_group())
            weight = self.language_model.shared_embedding_or_output_weight()
            clean_logits, _ = self.language_model.output_layer(clean_hidden, weight=weight, runtime_gather_output=True)
            vocab_size = self.config.text_config.vocab_size
            clean_logits = clean_logits.transpose(0, 1)[..., :vocab_size].float()
            score_mask = torch.cat((torch.zeros_like(prompt_mask), encoder_target_mask), dim=1)[:, 1:].bool()
            targets = clean_ids[:, 1:].masked_fill(~score_mask, -100)
            encoder_loss = F.cross_entropy(
                clean_logits[:, :-1].reshape(-1, vocab_size), targets.reshape(-1), ignore_index=-100
            )
        return DiffusionGemmaBlockDiffusionOutput(
            logits=logits, encoder_last_hidden_state=encoded.last_hidden_state, encoder_loss=encoder_loss
        )

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
        freeze_audio_model: bool = False,
        freeze_audio_projection: bool = False,
        freeze_self_conditioning: bool = False,
    ):
        super().freeze(
            freeze_language_model=freeze_language_model,
            freeze_vision_model=freeze_vision_model,
            freeze_vision_projection=freeze_vision_projection,
            freeze_audio_model=freeze_audio_model,
            freeze_audio_projection=freeze_audio_projection,
        )
        if freeze_self_conditioning and hasattr(self, "self_conditioning"):
            for param in self.self_conditioning.parameters():
                param.requires_grad = False


def _attention_qkv(args, kwargs):
    if len(args) >= 3:
        return args[:3]
    return kwargs["query"], kwargs["key"], kwargs["value"]


def _replace_attention_args(args, kwargs, query, key, value, attention_mask):
    args = list(args)
    if len(args) >= 3:
        args[:3] = [query, key, value]
    else:
        kwargs.update(query=query, key=key, value=value)
    if len(args) >= 4:
        args[3] = attention_mask
    else:
        kwargs["attention_mask"] = attention_mask
    if len(args) >= 5:
        args[4] = AttnMaskType.arbitrary
    else:
        kwargs["attn_mask_type"] = AttnMaskType.arbitrary
    return tuple(args), kwargs
