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
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


def is_transformers_min_version(version):
    """Check if minimum version of transformers is installed."""
    try:
        transformers_version = PkgVersion(transformers.__version__)
        return transformers_version >= PkgVersion(version)
    except Exception:
        # If version parsing fails, assume false for safety
        return False


class Qwen25VLModel(MegatronModule):
    """
    Qwen2.5 VL Model. (Based on GPT Transformer language model.)

    Args:
        config (GPTModelProvider):
            language model provider.
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers
        vocab_size (int):
            Vocabulary size
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional):
            Defaults to False.
        parallel_output (bool, optional):
            Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        rope_scaling (bool, optional): Toggle RoPE scaling.
        rope_scaling_factor (float): RoPE scaling factor. Default 8.
        scatter_embedding_sequence_parallel (bool, optional):
            Whether embeddings should be scattered across sequence parallel
            region or not. Defaults to True.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
        pg_collection (ProcessGroupCollection): Model communication process groups
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
            self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
            # Ensure HF visual tower params are marked for TP grad sync and future assignments are hooked.
            hook_hf_module_setattr_for_tp_grad_sync(self.visual)
        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Finalize grad will need these to be bind with module
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        # Bind methods from HF's Qwen2_5_VLModel to this instance
        # get_placeholder_mask is only available in transformers 4.55+
        if is_transformers_min_version("4.55.0"):
            self.get_placeholder_mask = types.MethodType(Qwen2_5_VLModel.get_placeholder_mask, self)
        else:
            raise RuntimeError(
                f"transformers version {transformers.__version__} is not supported. "
                f"get_placeholder_mask requires transformers >= 4.55.0. "
                f"Please upgrade transformers: pip install 'transformers>=4.55.0'"
            )

        self.get_image_features = types.MethodType(Qwen2_5_VLModel.get_image_features, self)
        self.get_video_features = types.MethodType(Qwen2_5_VLModel.get_video_features, self)
        self.get_rope_index = types.MethodType(Qwen2_5_VLModel.get_rope_index, self)

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
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
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        if self.pre_process:
            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # [decoder_seq_len, b, h_language]

                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [b, decoder_seq_len, h_language]

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            inputs_embeds = inputs_embeds.transpose(
                1, 0
            )  # [b, decoder_seq_len, h_language] -> [decoder_seq_len, b, h_language]

            if self.config.sequence_parallel:
                inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds)

        position_ids, rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
        )

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
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module (patch_embed and blocks).
            freeze_vision_projection (bool): Freeze the vision projection module (merger).
        """
        modules = []

        if freeze_language_model and hasattr(self, "language_model") and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model and hasattr(self, "visual") and self.visual is not None:
            # Vision model consists of patch_embed and blocks
            if hasattr(self.visual, "patch_embed"):
                modules.append(self.visual.patch_embed)
            if hasattr(self.visual, "blocks"):
                modules.append(self.visual.blocks)

        if freeze_vision_projection and hasattr(self, "visual") and self.visual is not None:
            # Vision projection is the merger module
            if hasattr(self.visual, "merger"):
                modules.append(self.visual.merger)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
