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

from typing import Any, TypeAlias, cast

import torch
import torch.nn.functional as F
from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel


TokenClassificationOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]


def _token_classification_output_processor(
    *,
    hidden_states: torch.Tensor,
    output_layer: torch.nn.Module,
    output_weight: torch.Tensor | None,
    labels: torch.Tensor | None,
    runtime_gather_output: bool | None,
    **_: Any,
) -> torch.Tensor:
    """Project token logits and compute non-vocabulary-parallel loss when requested."""
    logits, _ = cast(
        tuple[torch.Tensor, None],
        output_layer(
            hidden_states,
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        ),
    )
    if labels is None:
        return logits.transpose(0, 1).contiguous()

    labels = labels.transpose(0, 1).contiguous()
    token_losses = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    return token_losses.view_as(labels).transpose(0, 1).contiguous()


class Qwen3VLForTokenClassification(Qwen3VLModel):
    """Qwen3.5 VL model with replicated per-token classification postprocessing."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        inference_params: InferenceParams | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        extra_block_kwargs: dict[str, object] | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        image_input_mask: torch.Tensor | None = None,
        video_input_mask: torch.Tensor | None = None,
        cp_img_num: list[int] | None = None,
        images_padded: list[bool] | None = None,
        inference_context: object | None = None,
        runtime_gather_output: bool | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> TokenClassificationOutput:
        """Run the VLM and apply token-classification logits or per-token loss.

        Args:
            input_ids: Input token IDs.
            position_ids: Optional Qwen MRoPE position IDs.
            attention_mask: Optional language-model attention mask.
            labels: Optional token-classification labels.
            loss_mask: Optional per-token supervision mask.
            inference_params: Megatron inference state; currently unsupported by Qwen3VL.
            packed_seq_params: Optional packed-sequence metadata.
            extra_block_kwargs: Extra transformer-block keyword arguments.
            pixel_values: Optional image patch values.
            pixel_values_videos: Optional video patch values.
            image_grid_thw: Image temporal/height/width grid metadata.
            video_grid_thw: Video temporal/height/width grid metadata.
            image_input_mask: Positions receiving image embeddings.
            video_input_mask: Positions receiving video embeddings.
            cp_img_num: Per-context-parallel-rank image counts.
            images_padded: Whether individual images were padded.
            inference_context: Compatibility placeholder for inference context.
            runtime_gather_output: Runtime output-gather override.
            mm_token_type_ids: Multimodal token type IDs retained for API compatibility.
            **kwargs: Additional language-model keyword arguments.

        Returns:
            Token logits or per-token losses; context-parallel and non-last-stage
            paths may return a loss-mask tuple or a stage-output dictionary.

        Raises:
            ValueError: If a caller tries to override the reserved output processor.
        """
        if "output_processor" in kwargs:
            raise ValueError("Qwen3VLForTokenClassification reserves output_processor for classification.")
        kwargs["output_processor"] = _token_classification_output_processor
        return cast(
            TokenClassificationOutput,
            super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_input_mask=image_input_mask,
                video_input_mask=video_input_mask,
                cp_img_num=cp_img_num,
                images_padded=images_padded,
                inference_context=inference_context,
                runtime_gather_output=runtime_gather_output,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            ),
        )
