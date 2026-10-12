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
from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_q_cu_seqlens


SequenceClassificationOutput: TypeAlias = torch.Tensor | dict[str, torch.Tensor]


def _pool_unpacked_logits(
    logits: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    pad_token_id: int | None,
) -> torch.Tensor:
    """Pool unpacked [S, B, C] logits at each row's last non-padding token; all-padding rows use position zero."""
    batch_size = logits.size(1)
    if pad_token_id is None:
        if batch_size != 1:
            raise ValueError("Unpacked pooling requires batch_size=1 when pad_token_id is undefined.")
        return logits[-1]

    non_padding_mask = (input_ids != pad_token_id).to(device=logits.device)
    has_non_padding_token, offsets_from_end = non_padding_mask.flip(-1).max(dim=-1)
    last_token_indices = torch.where(has_non_padding_token, logits.size(0) - offsets_from_end - 1, 0)
    return logits[last_token_indices, torch.arange(batch_size, device=logits.device)]


def _pool_packed_logits(
    logits: torch.Tensor,
    *,
    packed_seq_params: PackedSeqParams,
) -> torch.Tensor:
    """Pool packed [T, 1, C] THD logits at each logical sequence's last token."""
    if packed_seq_params.qkv_format != "thd":
        raise ValueError(f"Packed pooling requires qkv_format='thd', got {packed_seq_params.qkv_format!r}.")

    cu_seqlens, physical_cu_seqlens = cast(
        tuple[torch.Tensor, torch.Tensor],
        get_packed_seq_q_cu_seqlens(packed_seq_params),
    )
    if cu_seqlens.numel() < 2 or physical_cu_seqlens.shape != cu_seqlens.shape:
        raise ValueError("Packed pooling requires matching query boundaries for at least one sequence.")

    logical_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    last_token_indices = (physical_cu_seqlens[:-1] + logical_lengths - 1).to(
        device=logits.device,
        dtype=torch.long,
    )
    return logits.index_select(0, last_token_indices)[:, 0]


def _sequence_classification_output_processor(
    *,
    hidden_states: torch.Tensor,
    output_layer: torch.nn.Module,
    output_weight: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    packed_seq_params: PackedSeqParams | None,
    runtime_gather_output: bool | None,
    config: Any,
    **_: Any,
) -> torch.Tensor:
    """Project token states and pool one score vector per logical sequence."""
    logits, _ = cast(
        tuple[torch.Tensor, None],
        output_layer(
            hidden_states,
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        ),
    )
    if packed_seq_params is not None:
        return _pool_packed_logits(logits, packed_seq_params=packed_seq_params)
    return _pool_unpacked_logits(
        logits,
        input_ids=cast(torch.Tensor, input_ids),
        pad_token_id=config.pad_token_id,
    )


class Qwen3VLForSequenceClassification(Qwen3VLModel):
    """Qwen3.5 VL model returning one endpoint score vector per logical sequence."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
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
        padding_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> SequenceClassificationOutput:
        """Run the VLM and return one score vector per logical sequence.

        Args:
            input_ids: Input token IDs.
            position_ids: Optional Qwen MRoPE position IDs.
            attention_mask: Optional language-model attention mask.
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
            padding_mask: Optional MoE padding mask.
            **kwargs: Additional language-model keyword arguments.

        Returns:
            One score vector per logical sequence; non-last-stage paths may return
            a stage-output dictionary.

        Raises:
            ValueError: If a caller tries to override the reserved output processor.
        """
        if "output_processor" in kwargs:
            raise ValueError("Qwen3VLForSequenceClassification reserves output_processor for sequence classification.")
        kwargs["output_processor"] = _sequence_classification_output_processor
        return cast(
            SequenceClassificationOutput,
            super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
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
                padding_mask=padding_mask,
                **kwargs,
            ),
        )
