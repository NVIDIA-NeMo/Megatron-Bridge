# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Qwen4-Exp vision encoder and HybridModel composition."""

import contextlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm, TERowParallelLinear
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.qwen.modeling_qwen4_exp.model_builder import Qwen4ExpModelBuilder
from megatron.bridge.models.qwen.modeling_qwen4_exp.model_config import Qwen4ExpModelConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import get_vision_model_config
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import PatchMergerSubmodules, reorganize_inputs
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel


@dataclass(kw_only=True)
class Qwen4ExpVLModelConfig(Qwen4ExpModelConfig):
    """Serializable vision-language construction settings."""

    builder: ClassVar[str] = "megatron.bridge.models.qwen.modeling_qwen4_exp.vl_model.Qwen4ExpVLModelBuilder"
    vision_config: dict[str, Any]
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    scatter_embedding_sequence_parallel: bool = False

    def finalize(self) -> None:
        """Validate the supported Qwen4 vision architecture before allocating it."""
        self.qwen4_mrope = True
        self.apply_rope_fusion = False
        super().finalize()
        if self.vision_config.get("deepstack_visual_indexes"):
            raise ValueError("Qwen4-Exp does not use deepstack vision injections.")
        if self.vision_config["out_hidden_size"] != self.hidden_size:
            raise ValueError("Vision merger output width must match the language hidden size.")
        if self.vision_config["hidden_act"] != "gelu_pytorch_tanh":
            raise ValueError("Qwen4-Exp vision requires gelu_pytorch_tanh activation.")
        if self.mrope_section is None:
            raise ValueError("Qwen4-Exp vision-language models require mrope_section.")
        if self.scatter_embedding_sequence_parallel:
            raise ValueError("VLM embeddings must be combined before sequence-parallel scattering.")


class Qwen4ExpVLModelBuilder(Qwen4ExpModelBuilder):
    """Build the vision encoder alongside the local hybrid language decoder."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> "Qwen4ExpVLModel":
        """Build a complete PP=CP=1 vision-language model."""
        language_model = super().build_model(pg_collection, pre_process, post_process, vp_stage)
        context = torch.device("meta") if self._model_config.init_model_with_meta_device else contextlib.nullcontext()
        with context:
            return Qwen4ExpVLModel(self._model_config, language_model, pg_collection)


class Qwen4ExpVLModel(MegatronModule):
    """Compose Qwen vision features with a Qwen4-Exp HybridModel decoder."""

    def __init__(
        self,
        config: Qwen4ExpVLModelConfig,
        language_model: MegatronModule,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__(config=config.transformer)
        self.model_config = config
        self.language_model = language_model
        self.pg_collection = pg_collection
        self.pre_process = self.post_process = True
        self.share_embeddings_and_output_weights = language_model.share_embeddings_and_output_weights
        vision_config = get_vision_model_config(SimpleNamespace(**config.vision_config), config.transformer)
        vision_config.params_dtype = config.params_dtype
        vision_config.bf16 = config.bf16
        vision_config.fp16 = config.fp16
        vision_config.attention_backend = config.attention_backend
        vision_config.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        vision_config.perform_initialization = config.perform_initialization
        spec = get_vit_layer_with_transformer_engine_spec()
        spec.submodules.self_attention.module = Qwen3VLSelfAttention
        self.vision_model = Qwen3VLVisionModel(
            transformer_config=vision_config,
            transformer_layer_spec=spec,
            patch_merger_spec=PatchMergerSubmodules(
                patch_norm=TENorm, linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
            ),
            pre_process=True,
            post_process=True,
            pg_collection=pg_collection,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor | list[torch.Tensor] | None) -> None:
        """Delegate pipeline schedule input setup to the language model."""
        self.language_model.set_input_tensor(input_tensor)

    def shared_embedding_or_output_weight(self) -> torch.Tensor:
        """Expose tied language embeddings for gradient finalization."""
        return self.language_model.shared_embedding_or_output_weight()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        image_input_mask: torch.Tensor | None = None,
        video_input_mask: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        runtime_gather_output: bool | None = None,
        inference_context: BaseInferenceContext | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Merge image/video features before running the hybrid language decoder.

        Packed inputs use flattened THD token rows. Padding masks and cached
        inference retain the language model's explicit unsupported guards.
        """
        # CP=1 leaves the caller's loss mask unchanged; the loss function applies it.
        config = self.model_config
        merge_size = config.vision_config["spatial_merge_size"]
        vision_data, grid_thw, vision_mask = reorganize_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_input_mask=image_input_mask,
            video_input_mask=video_input_mask,
            image_token_id=config.image_token_id,
            video_token_id=config.video_token_id,
            square_merge_size=merge_size**2,
        )
        embeddings = self.language_model.embedding(input_ids=input_ids, position_ids=None)
        if vision_data is not None and vision_data.numel():
            vision_embeddings, _ = self.vision_model(hidden_states=vision_data, grid_thw=grid_thw)
            if int(vision_mask.sum().item()) != vision_embeddings.shape[0]:
                raise ValueError("Visual token count does not match the vision encoder output.")
            embeddings = embeddings.transpose(0, 1).clone()
            embeddings[vision_mask] = vision_embeddings.to(embeddings.dtype)
            embeddings = embeddings.transpose(0, 1).contiguous()
        if config.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                embeddings, group=self.pg_collection.tp
            ).contiguous()
        if position_ids is None or position_ids.ndim != 3:
            position_ids, _ = get_rope_index(
                merge_size,
                config.image_token_id,
                config.video_token_id,
                config.vision_start_token_id,
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                packed_seq_params=packed_seq_params,
            )
        return self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=embeddings,
            labels=labels,
            packed_seq_params=packed_seq_params,
            runtime_gather_output=runtime_gather_output,
            inference_context=inference_context,
            padding_mask=padding_mask,
        )
