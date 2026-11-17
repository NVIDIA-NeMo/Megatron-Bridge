# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Understanding-stage composition of HF perception and a Megatron decoder."""

import importlib
from types import SimpleNamespace
from typing import Any

import torch
from megatron.core.transformer.module import MegatronModule
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from megatron.bridge.models.minicpmo.audio import MiniCPMOAudioAttention
from megatron.bridge.models.minicpmo.media import insert_media_embeddings
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


class MiniCPMOPerception(torch.nn.Module):
    """Build only the checkpoint's perception modules, without a second LLM."""

    def __init__(self, model_config: Any, *, source: str, revision: str | None) -> None:
        super().__init__()
        if not model_config.trust_remote_code:
            raise ValueError("MiniCPM-o perception requires explicitly setting trust_remote_code=True.")
        hf_config = AutoConfig.from_pretrained(source, revision=revision, trust_remote_code=True)
        # Reconstruct nested native configs too; silently retaining source
        # encoder settings would make serialized overrides misleading.
        hf_config = type(hf_config).from_dict(model_config.minicpmo_config)
        model_class = get_class_from_dynamic_module(hf_config.auto_map["AutoModel"], source, revision=revision)
        module = importlib.import_module(model_class.__module__)
        self.config = hf_config
        self._hf_model_class = model_class
        facade = SimpleNamespace(config=hf_config)
        self.vpm = model_class.init_vision_module(facade) if hf_config.init_vision else None
        self.resampler = (
            model_class.init_resampler(facade, hf_config.hidden_size, self.vpm.embed_dim)
            if self.vpm is not None
            else None
        )
        self.apm = model_class.init_audio_module(facade) if hf_config.init_audio else None
        if self.apm is not None:
            # Adapt the old three-result call contract without changing HF
            # attention math or checkpoint names. Weights are loaded by Bridge.
            for index, layer in enumerate(self.apm.layers):
                layer.self_attn = MiniCPMOAudioAttention(
                    embed_dim=hf_config.audio_config.d_model,
                    num_heads=hf_config.audio_config.encoder_attention_heads,
                    dropout=hf_config.audio_config.attention_dropout,
                    config=hf_config.audio_config,
                    layer_idx=index,
                )
        self.audio_projection_layer = (
            module.MultiModalProjector(in_dim=int(self.apm.config.encoder_ffn_dim // 4), out_dim=hf_config.hidden_size)
            if self.apm is not None
            else None
        )
        self.audio_avg_pooler = torch.nn.AvgPool1d(hf_config.audio_pool_step, stride=hf_config.audio_pool_step)
        self.audio_encoder_layer = -1
        for name, frozen in (
            ("vpm", model_config.freeze_vision_model),
            ("resampler", model_config.freeze_vision_projection),
            ("apm", model_config.freeze_audio_model),
            ("audio_projection_layer", model_config.freeze_audio_projection),
        ):
            child = getattr(self, name)
            if child is not None:
                child.requires_grad_(not frozen)
                hook_hf_module_setattr_for_tp_grad_sync(child)

    def forward(self, data: dict[str, Any], embeddings: torch.Tensor) -> torch.Tensor:
        """Run native encoders and insert features at each sample's own bounds."""
        batch = embeddings.shape[0]
        empty_bounds = [torch.empty((0, 2), dtype=torch.long) for _ in range(batch)]
        facade = SimpleNamespace(
            config=self.config,
            training=self.training,
            vpm=self.vpm,
            resampler=self.resampler,
            apm=self.apm,
            audio_projection_layer=self.audio_projection_layer,
            audio_avg_pooler=self.audio_avg_pooler,
            audio_encoder_layer=self.audio_encoder_layer,
            llm=SimpleNamespace(model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=embeddings))),
        )
        facade.subsequent_chunk_mask = self._hf_model_class.subsequent_chunk_mask
        facade._get_feat_extract_output_lengths = lambda lengths: (
            self._hf_model_class._get_feat_extract_output_lengths(facade, lengths)
        )
        occupied = None
        pixels = data.get("pixel_values", [[] for _ in range(batch)])
        if len(pixels) != batch:
            raise ValueError("pixel_values must preserve the original sample boundaries.")
        has_image = any(len(row) for row in pixels)
        if has_image:
            if self.vpm is None:
                raise ValueError("Image input requires an enabled vision encoder.")
            sizes = data.get("tgt_sizes", [])
            if len(sizes) != batch:
                raise ValueError("tgt_sizes must preserve the original sample boundaries.")
            vision = [
                self._hf_model_class.get_vision_embedding(
                    facade, {**data, "pixel_values": [row], "tgt_sizes": [size]}
                )[0]
                if len(row)
                else []
                for row, size in zip(pixels, sizes, strict=True)
            ]
        else:
            vision = [[] for _ in range(batch)]
        embeddings, occupied = insert_media_embeddings(
            embeddings, vision, data.get("image_bound", empty_bounds), occupied=occupied
        )
        if len(data.get("audio_features", [])):
            if self.apm is None:
                raise ValueError("Audio input requires an enabled audio encoder.")
            audio = self.encode_audio_samples(facade, data, batch_size=batch)
        else:
            audio = [[] for _ in range(batch)]
        embeddings, _ = insert_media_embeddings(
            embeddings, audio, data.get("audio_bounds", empty_bounds), occupied=occupied
        )
        return embeddings

    def encode_audio_samples(self, facade: Any, data: dict[str, Any], *, batch_size: int) -> list[list[torch.Tensor]]:
        """Encode exact-length clips, retaining sample boundaries and gradients.

        Native Whisper attention masks index post-convolution positions using
        pre-convolution lengths. Cropping before the native call avoids attending
        to padding introduced by other clips in the batch.
        """
        features = data["audio_features"]
        lengths = data.get("audio_feature_lens", [])
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError("audio_features must be a [clips, mel_bins, frames] tensor.")
        if len(lengths) != batch_size:
            raise ValueError("audio_feature_lens must preserve the original sample boundaries.")
        if any(row.ndim != 1 or row.dtype not in (torch.int32, torch.int64) for row in lengths):
            raise ValueError("Each audio_feature_lens row must be a one-dimensional integer tensor.")
        if sum(row.numel() for row in lengths) != features.shape[0]:
            raise ValueError("Audio clip and length counts differ.")
        grouped = []
        offset = 0
        for row in lengths:
            sample = []
            for length in row.tolist():
                if length <= 0 or length > features.shape[-1]:
                    raise ValueError("Audio lengths must be positive and within the mel frame dimension.")
                clip_data = {
                    **data,
                    "audio_features": features[offset : offset + 1, :, :length].contiguous(),
                    "audio_feature_lens": [row.new_tensor([length])],
                }
                encoded = self._hf_model_class.get_audio_embedding(
                    facade, clip_data, chunk_length=self.config.audio_chunk_length, dummy=False
                )
                sample.append(encoded[0][0])
                offset += 1
            grouped.append(sample)
        return grouped


class MiniCPMOModel(MegatronModule):
    """Expose the standard Megatron model API for MiniCPM-o understanding.

    Perception is composed with the existing Qwen3-compatible GPT decoder.
    The initial dense path requires PP=CP=1 and sequence parallelism disabled.
    """

    def __init__(self, language_model: torch.nn.Module, perception: MiniCPMOPerception) -> None:
        super().__init__(config=language_model.config)
        self.language_model = language_model
        self.perception = perception
        self.pre_process = language_model.pre_process
        self.post_process = language_model.post_process
        self.share_embeddings_and_output_weights = language_model.share_embeddings_and_output_weights

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        """Expose the decoder's tied-embedding synchronization hook."""
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor: Any) -> None:
        """Delegate the standard Megatron pipeline hook to the decoder."""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        media: dict[str, Any] | None = None,
        labels: torch.Tensor | None = None,
        packed_seq_params: Any = None,
        inference_params: Any = None,
        output_processor: Any = None,
        output_processor_context: Any = None,
    ) -> Any:
        """Compose dense per-sample inputs and preserve the decoder's output API."""
        if packed_seq_params is not None or inference_params is not None:
            raise NotImplementedError("MiniCPM-o currently exposes dense training forwards only.")
        if attention_mask is not None and (attention_mask.ndim != 4 or attention_mask.dtype != torch.bool):
            raise ValueError("Use a Megatron boolean [batch, 1, query, key] attention mask, not an HF padding mask.")
        if input_ids.ndim != 2 or position_ids.shape != input_ids.shape:
            raise ValueError("MiniCPM-o requires dense [batch, sequence] IDs and matching positions.")
        embeddings = self.language_model.embedding.word_embeddings(input_ids)
        embeddings = self.perception(media or {}, embeddings)
        embeddings = self.language_model.embedding.embedding_dropout(embeddings)
        kwargs = {}
        if output_processor is not None:
            kwargs.update(output_processor=output_processor, output_processor_context=output_processor_context)
        elif output_processor_context is not None:
            raise ValueError("output_processor_context requires output_processor.")
        return self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=embeddings.transpose(0, 1).contiguous(),
            labels=labels,
            **kwargs,
        )
