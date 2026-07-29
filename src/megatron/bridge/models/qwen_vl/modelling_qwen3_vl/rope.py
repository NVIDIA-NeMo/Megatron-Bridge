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


import warnings
from typing import List, Optional

import torch
import torch.nn as nn
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    get_pos_emb_on_this_cp_rank,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params
from torch import Tensor

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.fused_mrope import (
    fused_apply_mrope,
    fused_apply_mrope_thd,
    get_fused_mrope_thd_unavailable_reason,
    get_fused_mrope_unavailable_reason,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_q_cu_seqlens


_ROPE_FUSION_FALLBACK_WARNINGS: set[tuple[str, str]] = set()


def is_raw_mrope_freqs(
    freqs: torch.Tensor | None,
    *,
    sequence_length: int | None = None,
    mrope_section: list[int] | None = None,
) -> bool:
    """Return whether ``freqs`` uses the raw T/H/W mRoPE layout.

    ``sequence_length`` disambiguates raw ``[3, B, S, D/2]`` from a legacy
    materialized embedding whose sequence length happens to be three.
    """
    is_raw_shape = isinstance(freqs, torch.Tensor) and freqs.dim() == 4 and freqs.size(0) == 3
    if not is_raw_shape:
        return False
    if sequence_length is not None and freqs.size(2) != sequence_length:
        return False
    if mrope_section is not None and len(mrope_section) == 3:
        # Materialized split-half RoPE doubles the raw frequency width.
        return freqs.size(-1) != 2 * sum(int(value) for value in mrope_section)
    return True


def materialize_mrope_freqs(
    freqs: torch.Tensor,
    mrope_section: list[int],
    *,
    interleaved_mrope: bool,
    rotary_interleaved: bool = False,
) -> torch.Tensor:
    """Convert raw T/H/W frequencies to Bridge's legacy unfused layout.

    Unlike the fused Qwen3.5-VL kernel, this compatibility converter accepts
    any valid stride-three section split used by existing Qwen3-VL providers.
    """
    if not is_raw_mrope_freqs(freqs):
        raise ValueError(
            f"Raw mRoPE frequencies must have shape [3, batch, seq, rotary_dim / 2], got {tuple(freqs.shape)}"
        )
    if len(mrope_section) != 3:
        raise ValueError(f"mrope_section must contain T/H/W lengths, got {mrope_section}")

    sec_t, sec_h, sec_w = (int(section) for section in mrope_section)
    half_rotary_dim = freqs.size(-1)
    if min(sec_t, sec_h, sec_w) < 0 or sec_t + sec_h + sec_w != half_rotary_dim:
        raise ValueError(
            f"mrope_section {mrope_section} must be non-negative and sum to rotary_dim / 2 = {half_rotary_dim}"
        )

    if interleaved_mrope:
        freqs_out = freqs[0].clone()
        for axis, offset in enumerate((1, 2), start=1):
            idx = slice(offset, mrope_section[axis] * 3, 3)
            freqs_out[..., idx] = freqs[axis, ..., idx]
        if rotary_interleaved:
            batch = freqs_out.size(0)
            emb = torch.stack(
                (freqs_out.reshape(batch, -1, 1), freqs_out.reshape(batch, -1, 1)),
                dim=-1,
            ).view(batch, freqs_out.size(1), -1)
        else:
            emb = torch.cat((freqs_out, freqs_out), dim=-1)
    elif rotary_interleaved:
        batch = freqs.size(1)
        emb = torch.stack(
            (freqs.reshape(3, batch, -1, 1), freqs.reshape(3, batch, -1, 1)),
            dim=-1,
        ).view(3, batch, freqs.size(2), -1)
        doubled_sections = list(mrope_section) * 2
        emb = torch.cat(
            [chunk[index % 3] for index, chunk in enumerate(emb.split(doubled_sections, dim=-1))],
            dim=-1,
        )
    else:
        freqs_out = torch.empty_like(freqs[0])
        freqs_out[..., :sec_t] = freqs[0, ..., :sec_t]
        freqs_out[..., sec_t : sec_t + sec_h] = freqs[1, ..., sec_t : sec_t + sec_h]
        freqs_out[..., sec_t + sec_h :] = freqs[2, ..., sec_t + sec_h :]
        emb = torch.cat((freqs_out, freqs_out), dim=-1)

    return emb[..., None, :].transpose(0, 1).contiguous()


def _fused_section_unavailable_reason(
    t: torch.Tensor,
    freqs: torch.Tensor,
    mrope_section: list[int],
    *,
    interleaved_mrope: bool,
) -> str | None:
    if len(mrope_section) != 3:
        return f"mrope_section must contain three values, got {mrope_section}"
    half_rotary_dim = freqs.size(-1)
    section = tuple(int(value) for value in mrope_section)
    if min(section) < 0 or sum(section) != half_rotary_dim:
        return f"mrope_section {mrope_section} must sum to rotary_dim / 2 = {half_rotary_dim}"
    if interleaved_mrope:
        expected = (
            (half_rotary_dim + 2) // 3,
            (half_rotary_dim + 1) // 3,
            half_rotary_dim // 3,
        )
        if section != expected:
            return (
                f"stride-three interleaved mRoPE requires section {list(expected)} "
                f"for rotary_dim / 2 = {half_rotary_dim}, got {mrope_section}"
            )
    if 2 * half_rotary_dim > t.size(-1):
        return f"raw mRoPE rotary dim {2 * half_rotary_dim} exceeds input head dim {t.size(-1)}"
    return None


def _warn_rope_fusion_fallback(layout: str, reason: str) -> None:
    warning_key = (layout, reason)
    if warning_key in _ROPE_FUSION_FALLBACK_WARNINGS:
        return
    warnings.warn(
        f"Qwen-VL fused mRoPE is unavailable for {layout}: {reason}. Using the unfused implementation.",
        UserWarning,
        stacklevel=3,
    )
    _ROPE_FUSION_FALLBACK_WARNINGS.add(warning_key)


def _get_cp_rank(
    config: Qwen3VLTransformerConfig,
    cp_group: torch.distributed.ProcessGroup | None,
) -> tuple[int, int]:
    cp_size = max(int(getattr(config, "context_parallel_size", 1)), 1)
    if cp_size == 1:
        return 1, 0
    if cp_group is None:
        raise ValueError("Qwen-VL packed mRoPE with context parallelism requires a CP group.")
    if cp_group.size() != cp_size:
        raise ValueError(f"CP group size {cp_group.size()} does not match config size {cp_size}.")
    return cp_size, cp_group.rank()


def _get_thd_cp_freq_indices(
    cu_seqlens: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> torch.Tensor:
    """Build exact packed CP indices, including odd local sequence lengths."""
    cu_seqlens_cpu = cu_seqlens.detach().cpu().tolist()
    indices: list[int] = []
    for global_start, global_end in zip(cu_seqlens_cpu[:-1], cu_seqlens_cpu[1:]):
        global_length = global_end - global_start
        if global_length % cp_size != 0:
            raise ValueError(
                f"Packed sequence length {global_length} must be divisible by context parallel size {cp_size}."
            )
        local_length = global_length // cp_size
        first_length = (local_length + 1) // 2
        second_length = local_length // 2
        indices.extend(range(global_start + cp_rank * first_length, global_start + (cp_rank + 1) * first_length))
        indices.extend(range(global_end - (cp_rank + 1) * second_length, global_end - cp_rank * second_length))
    return torch.tensor(indices, dtype=torch.long, device=device)


def _apply_unfused_raw_mrope(
    t: torch.Tensor,
    freqs: torch.Tensor,
    config: Qwen3VLTransformerConfig,
    *,
    cu_seqlens: torch.Tensor | None,
    cp_size: int,
    cp_rank: int,
    freqs_are_local: bool,
) -> torch.Tensor:
    materialized = materialize_mrope_freqs(
        freqs,
        list(config.mrope_section),
        interleaved_mrope=bool(getattr(config, "mrope_interleaved", True)),
        rotary_interleaved=config.rotary_interleaved,
    )
    if cu_seqlens is not None and not freqs_are_local:
        indices = _get_thd_cp_freq_indices(
            cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
            device=materialized.device,
        )
        materialized = materialized.index_select(0, indices)

    orig_dtype = t.dtype
    compute_input = t.float() if config.apply_rotary_pos_emb_in_fp32 else t
    if cu_seqlens is None:
        result = _apply_rotary_pos_emb_bshd(
            compute_input,
            materialized,
            rotary_interleaved=config.rotary_interleaved,
        )
    else:
        result = _apply_rotary_pos_emb_bshd(
            compute_input[:, None],
            materialized,
            rotary_interleaved=config.rotary_interleaved,
        ).squeeze(1)
    return result.to(orig_dtype) if config.apply_rotary_pos_emb_in_fp32 else result


def _get_flat_packed_ranges(
    input_ids: torch.Tensor,
    packed_seq_params: PackedSeqParams | None,
) -> list[tuple[int, int, int]] | None:
    """Return ``(padded_start, valid_end, padded_end)`` ranges for flat packed input."""
    if packed_seq_params is None or input_ids is None or input_ids.dim() != 2 or input_ids.size(0) != 1:
        return None

    cu_seqlens_unpadded, cu_seqlens_padded = get_packed_seq_q_cu_seqlens(packed_seq_params)
    if (
        cu_seqlens_padded is None
        or cu_seqlens_unpadded is None
        or cu_seqlens_padded.numel() < 3
        or cu_seqlens_unpadded.numel() < cu_seqlens_padded.numel()
    ):
        return None

    max_len = input_ids.size(1)
    if int(cu_seqlens_padded[-1].item()) != max_len:
        return None

    ranges = []
    for idx in range(cu_seqlens_padded.numel() - 1):
        padded_start = int(cu_seqlens_padded[idx].item())
        padded_end = int(cu_seqlens_padded[idx + 1].item())
        unpadded_len = int((cu_seqlens_unpadded[idx + 1] - cu_seqlens_unpadded[idx]).item())
        valid_end = min(padded_start + unpadded_len, padded_end)
        ranges.append((padded_start, valid_end, padded_end))
    return ranges


def get_packed_seq_attention_mask(input_ids: torch.Tensor, packed_seq_params: PackedSeqParams) -> torch.Tensor:
    """Build a dense keep mask matching packed sequence metadata.

    Collate-time in-batch packing emits a flattened ``[1, total_padded]``
    token tensor. ``cu_seqlens_q_padded`` identifies segment boundaries in
    that flattened tensor, while ``cu_seqlens_q`` may identify the unpadded
    token counts. Qwen3-VL still needs a dense mask for its local THD
    conversion, so derive it from the same metadata used by attention.
    """
    cu_seqlens_unpadded, cu_seqlens_padded = get_packed_seq_q_cu_seqlens(packed_seq_params)

    if cu_seqlens_padded is None or cu_seqlens_unpadded is None or cu_seqlens_padded.numel() < 2:
        return torch.ones_like(input_ids, dtype=torch.bool)

    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    seq_count = cu_seqlens_padded.numel() - 1

    flat_packed_ranges = _get_flat_packed_ranges(input_ids, packed_seq_params)
    if flat_packed_ranges is not None:
        for padded_start, valid_end, _ in flat_packed_ranges:
            attention_mask[0, padded_start:valid_end] = True
        return attention_mask

    if input_ids.dim() == 2 and input_ids.size(0) == 1 and seq_count > 1:
        raise ValueError("Flat packed input length does not match its padded cu-seqlens metadata.")

    for idx in range(min(input_ids.size(0), seq_count)):
        seq_len = int((cu_seqlens_unpadded[idx + 1] - cu_seqlens_unpadded[idx]).item())
        attention_mask[idx, : min(seq_len, input_ids.size(1))] = True
    return attention_mask


class Qwen3VLMultimodalRotaryEmbedding(nn.Module):
    """Multimodal Rotary Embedding for language model.
    only support for qwen3vl

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position
            embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE
            for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float = 1.0,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_base: int = 10000,
        cp_group: torch.distributed.ProcessGroup = None,
        return_raw_freqs: bool = False,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved
        assert not self.rotary_interleaved, "only support qwen3vl"

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) / dim)
        )
        self.is_thd_format = False  # if is thd format, we do not need to split the rotary_pos_emb along CP
        self.return_raw_freqs = return_raw_freqs

        # default mrope section is [24, 20, 20], if no mrope section is provided, use default mrope section
        self.mrope_section = [24, 20, 20]
        assert cp_group is not None, "cp_group is required"
        self.cp_group = cp_group

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0].clone()  # overwrite a copy of the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(
        self,
        position_ids: torch.Tensor,
        mrope_section: List[int] | None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (torch.Tensor): A postion_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal,
                height and width in rope calculation.
            packed_seq_params (PackedSeqParams, optional): Packed sequence params. Defaults to None.
        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        # Use fp32 for position indices to avoid precision loss when inv_freq is bf16.
        seq = position_ids.to(device=self.inv_freq.device, dtype=torch.float32)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
        mrope_section = self.mrope_section if mrope_section is None else mrope_section
        if self.return_raw_freqs:
            if self.cp_group.size() > 1 and not self.is_thd_format:
                freqs = get_pos_emb_on_this_cp_rank(freqs, 2, self.cp_group)
            return freqs.contiguous()

        freqs = self.apply_interleaved_mrope(freqs, mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        if self.cp_group.size() > 1 and not self.is_thd_format:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current
            # CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0, self.cp_group)
        return emb

    def get_rotary_seq_len(
        self,
        inference_context: BaseInferenceContext,
        transformer: TransformerBlock,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
        packed_seq_params: Optional[PackedSeqParams] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> int:
        """Compatibility shim for newer MCore GPT preprocessing.

        Qwen3-VL/Qwen3-Omni mRoPE uses explicit multimodal `position_ids`, but the upstream
        GPT preprocess path still queries a rotary sequence length helper when preparing inputs.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        if packed_seq_params is not None:
            return max(packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv)
        if inference_context is not None:
            context_max_seq_len = inference_context.max_sequence_length
            input_seq_len = 0
            if transformer_input is not None:
                input_seq_len = transformer_input.size(0)
            elif transformer is not None and transformer.input_tensor is not None:
                input_seq_len = transformer.input_tensor.size(0)
            return max(context_max_seq_len, input_seq_len)

        if transformer is not None and transformer.input_tensor is not None:
            rotary_seq_len = transformer.input_tensor.size(0)
        else:
            rotary_seq_len = transformer_input.size(0)

        if transformer_config.sequence_parallel:
            rotary_seq_len *= transformer_config.tensor_model_parallel_size

        return rotary_seq_len


def _build_llm_rope_positions(
    sample_input_ids: torch.Tensor,
    *,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    image_index: int,
    video_index: int,
) -> tuple[torch.Tensor, int, int]:
    """Build Qwen3-VL MRoPE positions for one logical sample."""
    vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
    vision_tokens = sample_input_ids[vision_start_indices + 1]
    image_nums = int((vision_tokens == image_token_id).sum().item())
    video_nums = int((vision_tokens == video_token_id).sum().item())
    input_tokens = sample_input_ids.tolist()
    llm_pos_ids_list: list[torch.Tensor] = []
    st = 0
    remain_images, remain_videos = image_nums, video_nums
    for _ in range(image_nums + video_nums):
        if image_token_id in input_tokens and remain_images > 0:
            ed_image = input_tokens.index(image_token_id, st)
        else:
            ed_image = len(input_tokens) + 1
        if video_token_id in input_tokens and remain_videos > 0:
            ed_video = input_tokens.index(video_token_id, st)
        else:
            ed_video = len(input_tokens) + 1
        if ed_image < ed_video:
            t, h, w = (
                image_grid_thw[image_index][0],
                image_grid_thw[image_index][1],
                image_grid_thw[image_index][2],
            )
            image_index += 1
            remain_images -= 1
            ed = ed_image

        else:
            t, h, w = (
                video_grid_thw[video_index][0],
                video_grid_thw[video_index][1],
                video_grid_thw[video_index][2],
            )
            video_index += 1
            remain_videos -= 1
            ed = ed_video
        llm_grid_t, llm_grid_h, llm_grid_w = (
            t.item(),
            h.item() // spatial_merge_size,
            w.item() // spatial_merge_size,
        )
        text_len = ed - st

        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        # t_index is always 0 because timestamps encode temporal information for videos.
        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    if st < len(input_tokens):
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    return llm_positions, image_index, video_index


# Slightly modified from Qwen3VLModel.get_rope_index
def get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

    # Since we use timestamps to separate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    flat_packed_ranges = _get_flat_packed_ranges(input_ids, packed_seq_params)
    if flat_packed_ranges is not None:
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        mrope_position_deltas = []
        for padded_start, valid_end, padded_end in flat_packed_ranges:
            sample_input_ids = input_ids[0, padded_start:valid_end]
            llm_positions, image_index, video_index = _build_llm_rope_positions(
                sample_input_ids,
                spatial_merge_size=spatial_merge_size,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                vision_start_token_id=vision_start_token_id,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_index=image_index,
                video_index=video_index,
            )
            position_ids[..., 0, padded_start:valid_end] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - (padded_end - padded_start))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if packed_seq_params is not None and attention_mask is None and input_ids is not None:
        attention_mask = get_packed_seq_attention_mask(input_ids, packed_seq_params).to(dtype=input_ids.dtype)

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        # Handle multi-dimensional attention masks
        elif attention_mask.dim() > 2:
            # Collapse to [batch, seq] while preserving padding information
            attention_mask = attention_mask.any(dim=-1)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)
            attention_mask = attention_mask.to(dtype=total_input_ids.dtype)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, sample_input_ids in enumerate(total_input_ids):
            sample_input_ids = sample_input_ids[attention_mask[i] == 1]
            llm_positions, image_index, video_index = _build_llm_rope_positions(
                sample_input_ids,
                spatial_merge_size=spatial_merge_size,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                vision_start_token_id=vision_start_token_id,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_index=image_index,
                video_index=video_index,
            )
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=total_input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            # Handle multi-dimensional attention mask
            if attention_mask.dim() > 2:
                # Collapse to [batch, seq] while preserving padding information
                attention_mask = attention_mask.any(dim=-1)
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.squeeze(1)
                attention_mask = attention_mask.to(dtype=torch.long)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def apply_rotary_pos_emb_thd_absolute(
    t: Tensor, cu_seqlens: Tensor, freqs: Tensor, rotary_interleaved: bool = False
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32. Currently unused but kept for API consistency.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """
    return _apply_rotary_pos_emb_bshd(t[:, None], freqs, rotary_interleaved=rotary_interleaved).squeeze(1)


def apply_rotary_pos_emb_absolute(
    t: Tensor,
    freqs: Tensor,
    config: Qwen3VLTransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    *,
    cp_group: torch.distributed.ProcessGroup | None = None,
    max_seqlen: int | None = None,
) -> Tensor:
    """Apply Qwen-VL absolute mRoPE in BSHD or packed THD layout.

    Raw ``[3, batch, seq, rotary_dim / 2]`` frequencies use the local Triton
    kernel when requested and supported. Legacy materialized frequencies keep
    the existing unfused behavior.
    """
    del max_seqlen  # Propagated by attention for a stable packed call contract.

    section = list(config.mrope_section)
    cp_size_hint = max(int(getattr(config, "context_parallel_size", 1)), 1)
    raw_sequence_lengths = {t.size(0), t.size(0) * cp_size_hint} if cu_seqlens is not None else {t.size(0)}
    if is_raw_mrope_freqs(freqs, mrope_section=section) and freqs.size(2) in raw_sequence_lengths:
        interleaved_mrope = bool(getattr(config, "mrope_interleaved", True))
        section_reason = _fused_section_unavailable_reason(
            t,
            freqs,
            section,
            interleaved_mrope=interleaved_mrope,
        )
        cp_size, cp_rank = _get_cp_rank(config, cp_group)

        if cu_seqlens is None:
            if freqs.size(1) != t.size(1) or freqs.size(2) != t.size(0):
                raise ValueError(
                    "BSHD raw mRoPE frequencies must match input batch and sequence dimensions: "
                    f"input={tuple(t.shape)}, freqs={tuple(freqs.shape)}"
                )
            reason = section_reason
            if config.apply_rope_fusion and reason is None:
                reason = get_fused_mrope_unavailable_reason(t, freqs, config.rotary_interleaved)
            if config.apply_rope_fusion and reason is None:
                return fused_apply_mrope(
                    t,
                    freqs,
                    section,
                    interleaved_mrope=interleaved_mrope,
                    fp32_compute=config.apply_rotary_pos_emb_in_fp32,
                )
            if config.apply_rope_fusion:
                _warn_rope_fusion_fallback("BSHD", reason)
            return _apply_unfused_raw_mrope(
                t,
                freqs,
                config,
                cu_seqlens=None,
                cp_size=1,
                cp_rank=0,
                freqs_are_local=True,
            )

        if t.dim() != 3:
            raise ValueError(f"Packed THD input must have shape [tokens, heads, head_dim], got {tuple(t.shape)}")
        if freqs.size(1) != 1:
            raise ValueError(f"Packed THD raw mRoPE requires frequency batch size 1, got {freqs.size(1)}")

        local_tokens = t.size(0)
        global_tokens = local_tokens * cp_size
        if freqs.size(2) == local_tokens:
            freqs_are_local = True
            launch_cp_size, launch_cp_rank = 1, 0
            if cp_size > 1:
                if bool(torch.any((cu_seqlens[1:] - cu_seqlens[:-1]) % cp_size).item()):
                    raise ValueError("Each packed sequence length must be divisible by context parallel size.")
                launch_cu_seqlens = torch.div(cu_seqlens, cp_size, rounding_mode="floor")
            else:
                launch_cu_seqlens = cu_seqlens
        elif freqs.size(2) == global_tokens:
            freqs_are_local = False
            launch_cp_size, launch_cp_rank = cp_size, cp_rank
            launch_cu_seqlens = cu_seqlens
        else:
            raise ValueError(
                "Packed THD raw mRoPE frequency sequence length must be local or global: "
                f"input tokens={local_tokens}, CP={cp_size}, freqs={freqs.size(2)}"
            )

        reason = section_reason
        if config.apply_rope_fusion and reason is None:
            reason = get_fused_mrope_thd_unavailable_reason(
                t,
                launch_cu_seqlens,
                freqs,
                config.rotary_interleaved,
                cp_size=launch_cp_size,
                cp_rank=launch_cp_rank,
            )
        if config.apply_rope_fusion and reason is None:
            return fused_apply_mrope_thd(
                t,
                launch_cu_seqlens,
                freqs,
                section,
                interleaved_mrope=interleaved_mrope,
                cp_size=launch_cp_size,
                cp_rank=launch_cp_rank,
                fp32_compute=config.apply_rotary_pos_emb_in_fp32,
            )
        if config.apply_rope_fusion:
            _warn_rope_fusion_fallback("THD", reason)
        return _apply_unfused_raw_mrope(
            t,
            freqs,
            config,
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
            freqs_are_local=freqs_are_local,
        )

    orig_t_dtype = t.dtype
    if config.apply_rotary_pos_emb_in_fp32:
        t = t.float()

    if cu_seqlens is None:
        result = _apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved=config.rotary_interleaved)
    else:
        result = apply_rotary_pos_emb_thd_absolute(t, cu_seqlens, freqs, rotary_interleaved=config.rotary_interleaved)

    if config.apply_rotary_pos_emb_in_fp32:
        result = result.to(orig_t_dtype)

    return result
