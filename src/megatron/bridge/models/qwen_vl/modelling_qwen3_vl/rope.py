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


from typing import List, Optional

import torch
import torch.nn as nn
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    fused_apply_rotary_pos_emb,
    get_pos_emb_on_this_cp_rank,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params
from torch import Tensor

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_q_cu_seqlens
from megatron.bridge.utils.common_utils import warn_rank_0


# Reasons already reported by _warn_fused_rope_fallback: each one is reported once per process.
_FUSED_ROPE_FALLBACK_REASONS: set[str] = set()


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
    """Multimodal Rotary Embedding for Qwen3-VL/Qwen3-Omni language model.

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
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) / dim)
        )
        self.is_thd_format = False  # if is thd format, we do not need to split the rotary_pos_emb along CP

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
        freqs_t = freqs[0]  # just overwrite the first dimension T
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
        *,
        cp_group: torch.distributed.ProcessGroup | None = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (torch.Tensor): A postion_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal,
                height and width in rope calculation.
            packed_seq_params (PackedSeqParams, optional): Packed sequence params. Defaults to None.
            cp_group (ProcessGroup, optional): Runtime CP group; defaults to the construction group.
        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        # Use fp32 for position indices to avoid precision loss when inv_freq is bf16.
        seq = position_ids.to(device=self.inv_freq.device, dtype=torch.float32)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, seq_length, dim)
        freqs = seq[..., None] * self.inv_freq
        if mrope_section is not None:
            freqs = self.apply_interleaved_mrope(freqs, mrope_section)
        else:
            # if mrope_section is not provided, use default mrope section
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        cp_group = self.cp_group if cp_group is None else cp_group
        if cp_group.size() > 1 and not self.is_thd_format:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current
            # CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0, cp_group)
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


def _fused_rope_unsupported_reason(t: Tensor, freqs: Tensor, cu_seqlens: Tensor | None) -> str | None:
    """Return why Transformer Engine's fused RoPE cannot replace the unfused code here, or None if it can.

    The unfused code pairs row ``i`` of ``freqs`` with position ``i`` of ``t`` (token ``i`` for packed input),
    broadcasting it over the heads, and over the batch when ``freqs`` has a single batch column. Transformer
    Engine's SBHD kernel does the same for a dense ``[rows, 1, 1, rotary_dim]`` table, so any input that can be
    presented in that form takes the fused path. Only tensor metadata is read: the check does not synchronize
    with the device and is safe during CUDA graph capture.

    Args:
        t: Query or key, ``[s, b, h, d]``, or ``[t, h, d]`` when ``cu_seqlens`` is given.
        freqs: Rotary angles, one row per position (and sample) or per packed token.
        cu_seqlens: Packed-sequence boundaries, or None for ``sbhd`` input.

    Returns:
        A short reason when the inputs need the unfused implementation, else None.
    """
    if fused_apply_rotary_pos_emb is None:
        return "Transformer Engine's fused RoPE is not available"
    if not t.is_cuda:
        return "the input is not on a CUDA device"
    if t.numel() == 0:
        return "the input is empty"
    # The kernel computes element offsets in 32-bit integers.
    span = t.storage_offset() + sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
    if max(span, t.numel()) >= 2**31:
        return "the input is too large for the kernel's 32-bit element offsets"
    if freqs.requires_grad:
        return "the rotary frequencies require gradients, which the fused kernel does not compute"
    if freqs.dim() != 4 or freqs.size(2) != 1:
        return "the rotary frequencies are not shaped [rows, batch, 1, rotary_dim]"
    rotary_dim = freqs.size(3)
    if rotary_dim % 2 or rotary_dim > t.size(-1):
        return "the rotary dimension is odd or larger than the head dimension"
    if cu_seqlens is None:
        if t.dim() != 4 or freqs.size(0) != t.size(0) or freqs.size(1) not in (1, t.size(1)):
            return "the rotary frequencies do not hold one row per sequence position"
    elif t.dim() != 3 or freqs.size(0) != t.size(0) or freqs.size(1) != 1:
        return "the packed rotary frequencies do not hold one row per token"
    return None


def _warn_fused_rope_fallback(reason: str) -> None:
    """Report, once per reason and process, that apply_rope_fusion fell back to the unfused implementation."""
    if reason not in _FUSED_ROPE_FALLBACK_REASONS:
        _FUSED_ROPE_FALLBACK_REASONS.add(reason)
        warn_rank_0(f"apply_rope_fusion is enabled, but {reason}; using the unfused RoPE implementation.")


def _apply_fused_rotary_pos_emb_absolute(t: Tensor, freqs: Tensor, *, rotary_interleaved: bool) -> Tensor:
    """Apply per-token RoPE with Transformer Engine's fused SBHD kernel.

    The kernel computes in FP32, rounds once to ``t``'s dtype and returns a new contiguous tensor. Callers check the
    inputs with :func:`_fused_rope_unsupported_reason` first.
    """
    # The kernel reads freqs as a dense [rows, rotary_dim] table without checking its strides.
    freqs = freqs.contiguous()
    # Unlike the unfused code, which saves cos and sin, the kernel saves freqs itself and reads it again in the
    # backward. A CUDA graph static input that holds freqs must therefore keep a sample's values until that sample's
    # backward has run. Transformer Engine's per-layer graphs and Megatron-Core's local graphs reuse a static input
    # only after that backward; any new grouping of graphed callables must keep this guarantee.
    if t.dim() == 3:
        # Packed THD: one sequence of batch size 1 whose token i takes freqs row i, as in the unfused code. Transformer
        # Engine's THD mode is not used: it restarts the row index at every cu_seqlens boundary.
        return fused_apply_rotary_pos_emb(t.unsqueeze(1), freqs, interleaved=rotary_interleaved).squeeze(1)
    if freqs.size(1) == 1:
        # One row per position, shared by every sample of the micro batch.
        return fused_apply_rotary_pos_emb(t, freqs, interleaved=rotary_interleaved)
    # One row per (position, sample), as MRoPE has for micro batch > 1: the kernel takes one row per position, so the
    # micro batch is presented as one sequence of s * b rows. reshape is a view when t's sequence and batch dims merge,
    # which holds for q and k split from the QKV projection, and a copy otherwise.
    seq_len, batch = t.shape[:2]
    out = fused_apply_rotary_pos_emb(
        t.reshape(seq_len * batch, 1, *t.shape[2:]),
        freqs.view(seq_len * batch, 1, 1, freqs.size(3)),
        interleaved=rotary_interleaved,
    )
    return out.view(t.shape)


def apply_rotary_pos_emb_absolute(
    t: Tensor,
    freqs: Tensor,
    config: Qwen3VLTransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
) -> Tensor:
    """Apply Qwen's per-token rotary embedding to a query or key tensor.

    Unlike Megatron-Core's ``apply_rotary_pos_emb``, ``freqs`` is not a table indexed by the position within a
    sequence: row ``i`` holds the angles of position ``i`` of ``t`` (token ``i`` for packed input), already sliced
    for this context-parallel rank, so ``cu_seqlens`` is not needed to pair them.

    With ``config.apply_rope_fusion``, Transformer Engine's fused kernel runs whenever it computes the same rotation
    (see :func:`_fused_rope_unsupported_reason`); other inputs use the unfused implementation, with one warning per
    reason. The kernel computes in FP32 and rounds once to ``t``'s dtype, which is what
    ``config.apply_rotary_pos_emb_in_fp32`` asks for, so ``t`` is passed to it without casts.

    Args:
        t: Query or key, ``[s, b, h, d]``, or ``[t, h, d]`` for packed (``thd``) input.
        freqs: Rotary angles. For ``[s, b, h, d]`` input, ``[s, b, 1, rotary_dim]`` (one row per position and
            sample, as MRoPE produces) or ``[s, 1, 1, rotary_dim]``; for packed input, ``[t, 1, 1, rotary_dim]``.
        config: The attention config; reads ``apply_rope_fusion``, ``apply_rotary_pos_emb_in_fp32`` and
            ``rotary_interleaved``.
        cu_seqlens: Packed-sequence boundaries for ``thd`` input, or None.

    Returns:
        ``t`` with its first ``rotary_dim`` channels rotated, in ``t``'s dtype.
    """
    if config.apply_rope_fusion:
        reason = _fused_rope_unsupported_reason(t, freqs, cu_seqlens)
        if reason is None:
            return _apply_fused_rotary_pos_emb_absolute(t, freqs, rotary_interleaved=config.rotary_interleaved)
        _warn_fused_rope_fallback(reason)

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
