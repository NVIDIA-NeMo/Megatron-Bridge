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

import types
from typing import Optional

import torch
import transformers
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from packaging.version import Version as PkgVersion
from torch import Tensor

from megatron.bridge.models.gpt_provider import GPTModelProvider


def _is_transformers_min_version(version: str) -> bool:
    try:
        transformers_version = PkgVersion(transformers.__version__)
        return transformers_version >= PkgVersion(version)
    except Exception:
        return False


class Qwen3VLModel(MegatronModule):
    """
    Qwen3 VL Model wrapper for Megatron-Core.

    This class integrates the HuggingFace Qwen3-VL components (vision + text) with
    Megatron-Core's GPT language model via the provider interface, mirroring the
    integration pattern used for Qwen2.5-VL.
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

        # Import here to avoid hard dependency errors if transformers lacks Qwen3
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLModel as HFQwen3VLModel,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionModel,
        )

        if pre_process:
            # Build HF visual encoder from HF vision config carried by provider
            self.visual = Qwen3VLVisionModel._from_config(config.vision_config)

        # Build Megatron language model via provider
        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Bind HF helper methods onto this wrapper for reuse in forward
        # - get_placeholder_mask
        self.get_placeholder_mask = types.MethodType(HFQwen3VLModel.get_placeholder_mask, self)
        # - image/video features
        self.get_image_features = types.MethodType(HFQwen3VLModel.get_image_features, self)
        self.get_video_features = types.MethodType(HFQwen3VLModel.get_video_features, self)
        # - RoPE index (Qwen3 uses timestamps and returns (position_ids, rope_deltas))
        self.get_rope_index = types.MethodType(HFQwen3VLModel.get_rope_index, self)

        # Cache for rope deltas (follow HF behavior)
        self.rope_deltas = None

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Megatron language model expects [seq, batch, hidden]
        if self.pre_process:
            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # [decoder_seq_len, b, h]
                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [b, s, h]

            image_mask = None
            video_mask = None

            if pixel_values is not None:
                image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Back to [seq, batch, hidden]
            inputs_embeds = inputs_embeds.transpose(1, 0)

            if self.config.sequence_parallel:
                inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds)

        # Qwen3 returns (position_ids (3, B, S), rope_deltas (B, 1))
        # We do not use second_per_grid_ts here; Qwen3 encodes timestamps in get_rope_index
        # Pass along attention_mask directly; Megatron core handles masks internally
        # If caller provided position_ids, honor them; otherwise compute via HF helper
        if position_ids is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas

        outputs = self.language_model.forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
        )
        return outputs

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool):
        modules = []
        if freeze_language_model and hasattr(self, "language_model") and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and hasattr(self, "visual") and self.visual is not None:
            if hasattr(self.visual, "patch_embed"):
                modules.append(self.visual.patch_embed)
            if hasattr(self.visual, "blocks"):
                modules.append(self.visual.blocks)
        if freeze_vision_projection and hasattr(self, "visual") and self.visual is not None:
            if hasattr(self.visual, "merger"):
                modules.append(self.visual.merger)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
