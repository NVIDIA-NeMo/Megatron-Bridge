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

"""Hybrid language decoder with directly invoked HF perception encoders."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from torch import Tensor, nn

from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerOutputLayerMixin
from megatron.bridge.models.muse_spark_open.modeling_muse_spark_open.blocks import get_muse_spark_hybrid_stack_spec
from megatron.bridge.models.muse_spark_open.muse_spark_open_config import MuseSparkModelConfig
from megatron.bridge.utils.common_utils import (
    hook_hf_module_setattr_for_tp_grad_sync,
    slice_batch_for_context_parallel,
)


class MuseSparkModel(HybridModel):
    """Place HF perception towers on the embedding stage of an MCore HybridModel."""

    def __init__(
        self,
        config: MuseSparkModelConfig,
        *,
        vocab_size: int,
        pg_collection: ProcessGroupCollection,
        pre_process: bool,
        post_process: bool,
    ) -> None:
        super().__init__(
            config=config.transformer,
            hybrid_stack_spec=get_muse_spark_hybrid_stack_spec(config.transformer),
            vocab_size=vocab_size,
            max_sequence_length=config.seq_length,
            hybrid_layer_pattern=config.hybrid_layer_pattern,
            fp16_lm_cross_entropy=config.fp16_lm_cross_entropy,
            parallel_output=config.parallel_output,
            share_embeddings_and_output_weights=config.share_embeddings_and_output_weights,
            position_embedding_type=config.position_embedding_type,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            scatter_embedding_sequence_parallel=False,
            seq_len_interpolation_factor=config.seq_len_interpolation_factor,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            **logit_dtype_kwarg(HybridModel, config.logit_dtype),
        )
        self.model_config = config
        if post_process:
            extend_instance(self.output_layer, MuseGlimmerOutputLayerMixin)
        if pre_process and not config.hf_model_text_only:
            self._build_perception()
        self._freeze_requested_modules()

    def _build_perception(self) -> None:
        # Keep imports lazy: config conversion and ordinary Bridge imports do
        # not require a Transformers build that includes Muse Spark Open.
        try:
            from transformers.models.muse_spark.configuration_muse_spark import MuseSparkConfig
            from transformers.models.muse_spark.modeling_muse_spark import (
                MuseSparkAudioModel,
                MuseSparkVisionAdapter,
                MuseSparkVisionModel,
            )
        except ImportError as error:
            raise ImportError(
                "Muse Spark Open model construction requires a Transformers build that provides transformers.models.muse_spark."
            ) from error

        hf_config = MuseSparkConfig.from_dict(self.model_config.hf_config)
        self.vision_tower = MuseSparkVisionModel(hf_config.vision_config)
        self.audio_tower = MuseSparkAudioModel(hf_config.audio_config)
        self.vision_adapter = MuseSparkVisionAdapter(hf_config)
        self.vision_projection = nn.Linear(hf_config.projector_hidden_size, self.config.hidden_size, bias=False)
        self.audio_samples_per_token = hf_config.audio_config.samples_per_token
        for module in (self.vision_tower, self.audio_tower, self.vision_adapter, self.vision_projection):
            module.to(dtype=self.config.params_dtype)
            hook_hf_module_setattr_for_tp_grad_sync(module)

    def _freeze_requested_modules(self) -> None:
        config = self.model_config
        groups = (
            (config.freeze_language_model, ("embedding", "decoder", "output_layer")),
            (config.freeze_vision_model, ("vision_tower",)),
            (config.freeze_audio_model, ("audio_tower",)),
            (config.freeze_vision_projection, ("vision_adapter", "vision_projection")),
        )
        for freeze, names in groups:
            if freeze:
                for name in names:
                    module = getattr(self, name, None)
                    if module is not None:
                        module.requires_grad_(False)

    def _normalize_embeddings(self, embeddings: Tensor) -> Tensor:
        return F.rms_norm(embeddings, (self.config.hidden_size,), eps=self.config.layernorm_epsilon)

    def _vision_features(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        self._restore_hf_vision_rope_buffers()
        features = self.vision_tower(pixel_values=pixel_values, grid_thw=grid_thw).last_hidden_state
        return self._normalize_embeddings(self.vision_projection(self.vision_adapter(features)))

    def _restore_hf_vision_rope_buffers(self) -> None:
        """Restore HF's nonpersistent FP32 frequencies after model dtype conversions."""
        rotary = self.vision_tower.rotary_emb
        frequencies, scaling = rotary.compute_axial_rope_parameters(rotary.config, device=rotary.inv_freq.device)
        rotary.inv_freq = frequencies.float()
        rotary.original_inv_freq = frequencies.clone().float()
        rotary.attention_scaling = scaling

    def _audio_features(self, input_values: Tensor, input_values_mask: Tensor | None) -> Tensor:
        features = self.audio_tower(input_values=input_values).last_hidden_state
        if input_values_mask is None:
            features = features.reshape(-1, self.config.hidden_size)
        else:
            lengths = input_values_mask.sum(-1)
            lengths = (lengths + self.audio_samples_per_token - 1) // self.audio_samples_per_token
            token_mask = torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
            features = features[token_mask]
        return self._normalize_embeddings(features)

    @staticmethod
    def _insert_features(embeddings: Tensor, input_ids: Tensor, features: Tensor, token_id: int) -> Tensor:
        mask = (input_ids == token_id).unsqueeze(-1).expand_as(embeddings)
        if embeddings[mask].numel() != features.numel():
            raise ValueError(f"Media placeholders do not match encoded features for token {token_id}.")
        return embeddings.masked_scatter(mask, features.to(device=embeddings.device, dtype=embeddings.dtype))

    def forward(
        self,
        input_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        *,
        inputs_embeds: Tensor | None = None,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        input_values: Tensor | None = None,
        input_values_mask: Tensor | None = None,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
        packed_seq_params: Any = None,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Encode media with HF, insert features, then execute the Hybrid decoder.

        ``inputs_embeds`` is batch-first and already normalized, as in HF.
        The wrapper receives full sequences and slices for CP after media insertion.
        """
        if "decoder_input" in kwargs:
            raise ValueError("Pass batch-first inputs_embeds to MuseSparkModel, not decoder_input.")
        if self.model_config.hf_model_text_only and any(
            value is not None
            for value in (
                pixel_values,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
                input_values,
                input_values_mask,
            )
        ):
            raise ValueError("The text-only Muse Spark Open model does not accept perception inputs.")
        inference_context = kwargs.get("inference_context") or kwargs.get("inference_params")
        if inference_context is not None and inference_context.is_dynamic_batching():
            raise ValueError(
                "Muse Spark Open dynamic batching requires validation of the attention scale and output norm."
            )
        decoder_input = None
        if self.pre_process:
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required on the embedding stage.")
                # HF substitutes a valid text ID before embedding media placeholders,
                # including placeholders whose corresponding media is absent.
                embedding_ids = input_ids.clone()
                for token_id in self.model_config.special_token_ids.values():
                    embedding_ids.masked_fill_(input_ids == token_id, 0)
                inputs_embeds = (
                    self._normalize_embeddings(self.embedding(input_ids=embedding_ids, position_ids=None))
                    .transpose(0, 1)
                    .contiguous()
                )
            for pixels, grid, key in (
                (pixel_values, image_grid_thw, "image_token_id"),
                (pixel_values_videos, video_grid_thw, "video_token_id"),
            ):
                if pixels is not None:
                    if grid is None or input_ids is None:
                        raise ValueError("Media inputs require input_ids and the corresponding patch grid.")
                    inputs_embeds = self._insert_features(
                        inputs_embeds, input_ids, self._vision_features(pixels, grid), self.model_config.hf_config[key]
                    )
            if input_values is not None:
                if input_ids is None:
                    raise ValueError("Audio inputs require input_ids to locate placeholders.")
                inputs_embeds = self._insert_features(
                    inputs_embeds,
                    input_ids,
                    self._audio_features(input_values, input_values_mask),
                    self.model_config.hf_config["audio_token_id"],
                )
            decoder_input = inputs_embeds.transpose(0, 1).contiguous()

        if padding_mask is not None:
            # Reuse the same CP partitioning for the router's batch-first mask.
            _, padding_mask, _, _, _ = slice_batch_for_context_parallel(
                inputs_embeds=None,
                labels=padding_mask,
                loss_mask=None,
                position_ids=None,
                attention_mask=None,
                packed_seq_params=packed_seq_params,
                pg_collection=self.pg_collection,
            )
        decoder_input, labels, loss_mask, position_ids, attention_mask = slice_batch_for_context_parallel(
            inputs_embeds=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
            pg_collection=self.pg_collection,
        )
        if self.config.sequence_parallel and decoder_input is not None:
            decoder_input = scatter_to_sequence_parallel_region(decoder_input, group=self.pg_collection.tp)
        output = super().forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
            **kwargs,
        )
        return (output, loss_mask) if loss_mask is not None else output
