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


from typing import Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams


def split_part_by_cp_tp(cp_size, cp_rank, tp_size, tp_rank, split_size):
    part_list = list(range(split_size))

    cp_rank2 = 2 * cp_size - cp_rank - 1
    cp_part_list = part_list[cp_rank * tp_size:(cp_rank + 1) *
                             tp_size] + part_list[cp_rank2 * tp_size:(cp_rank2 + 1) * tp_size]

    assert len(cp_part_list) % tp_size == 0
    echo_tp_len = len(cp_part_list) // tp_size
    cp_tp_part_list = cp_part_list[tp_rank * echo_tp_len:(tp_rank + 1) * echo_tp_len]
    return cp_tp_part_list


def split_deepstack_embs(
    visual_pos_masks: torch.Tensor,
    deepstack_visual_embeds: list[torch.Tensor],
    tp_size: int = 1,
    tp_rank: int = 0,
    cp_size: int = 1,
    cp_rank: int = 0,
    sequence_parallel: bool = True,
):
    """Split deepstack visual embeddings for tensor and context parallelism.

    Args:
        visual_pos_masks: Visual position masks tensor
        deepstack_visual_embeds: List of deepstack visual embeddings
        tp_size: Tensor parallel size (default: 1)
        tp_rank: Tensor parallel rank (default: 0)
        cp_size: Context parallel size (default: 1)
        cp_rank: Context parallel rank (default: 0)

    Returns:
        Split visual embeddings based on parallelism configuration
    """
    if not sequence_parallel:
        tp_size = 1
        tp_rank = 0
    split_size = tp_size
    if cp_size > 1:
        split_size *= (cp_size * 2)
    if split_size == 1 or visual_pos_masks is None:
        return visual_pos_masks, deepstack_visual_embeds

    assert visual_pos_masks.dim() == 2
    assert visual_pos_masks.shape[-1] % split_size == 0
    batch_size = visual_pos_masks.size(0)
    
    # first split by cp(zigzag), then split by sp
    # for example cp=2/tp=4
    # visual_pos_masks will split in 16 part:
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # first split by cp(zigzag) is:
    # cp_rank0: [0, 1, 2, 3, 12, 13, 14, 15]
    # cp_rank1: [4, 5, 6, 7, 8, 9, 10, 11]
    # then split by sp:
    # cp_rank0/tp_rank0 = [0, 1]
    # cp_rank0/tp_rank1 = [2, 3]
    # ...
    # cp_rank1/tp_rank2 = [8, 9]
    # cp_rank1/tp_rank3 = [10, 11]
    cp_tp_part_list = split_part_by_cp_tp(cp_size, cp_rank, tp_size, tp_rank, split_size)
    visual_pos_masks_list = visual_pos_masks.chunk(split_size, dim=-1)
    embed_lens = [ele.sum(-1) for ele in visual_pos_masks_list]

    embed_lens = torch.stack(embed_lens, dim=-1)
    embed_cu_lens = embed_lens.view(-1).cumsum(dim=-1).tolist()
    assert len(embed_cu_lens) == split_size * batch_size
    embed_cu_lens = [0] + embed_cu_lens

    cp_tp_slices = []
    for i in range(batch_size):
        for idx in cp_tp_part_list:
            idx += i * split_size
            cp_tp_slices.append(slice(embed_cu_lens[idx], embed_cu_lens[idx + 1]))

    deepstack_visual_embeds_ret = []
    for deepstack_visual_embed in deepstack_visual_embeds:
        tmp_slice_tensor = []
        for tp_slice in cp_tp_slices:
            tmp_slice_tensor.append(deepstack_visual_embed[tp_slice])
        deepstack_visual_embeds_ret.append(torch.cat(tmp_slice_tensor, dim=0))

    visual_pos_masks_ret = torch.cat([visual_pos_masks_list[i] for i in cp_tp_part_list], dim=-1)

    return visual_pos_masks_ret, deepstack_visual_embeds_ret


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

    # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if packed_seq_params is not None and attention_mask is None and input_ids is not None:
        # Build an attention mask from packed sequence metadata when one is not provided.
        # cu_seqlens_q entries are cumulative lengths; their diffs give per-sample lengths.
        cu_seqlens = packed_seq_params.cu_seqlens_q
        if cu_seqlens is not None and cu_seqlens.numel() >= 2:
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            attention_mask = torch.zeros_like(input_ids, dtype=input_ids.dtype)
            max_len = attention_mask.shape[1]
            for i, seq_len in enumerate(seq_lens.tolist()):
                valid = min(int(seq_len), max_len)
                attention_mask[i, :valid] = 1
        else:
            # Fallback to a dense mask if packed metadata is missing.
            attention_mask = torch.ones_like(input_ids)

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
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

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
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
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
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


def find_vision_id_index(
    input_ids: torch.Tensor,
    image_token_id: int,
    video_token_id: int,
):
    assert input_ids.dim() == 1, "input_ids should be flaaten"
    if input_ids.numel() == 0:
        return []

    device = input_ids.device
    dtype = input_ids.dtype
    assert dtype in [torch.int, torch.int64]

    # keep the value of image_token_id/video_token_id value, others are -1
    code = torch.where(
        (input_ids == image_token_id) | (input_ids == video_token_id),
        input_ids,
        torch.tensor(-1, device=device, dtype=dtype),
    )

    # find the change idx
    first = torch.tensor([True], device=device, dtype=torch.bool)
    change = torch.cat([first, code[1:] != code[:-1]])
    change_idx = torch.nonzero(change, as_tuple=False).flatten()

    # only keep the change of image_token_id/video_token_id
    keep = code[change_idx] > 0
    starts = change_idx[keep]

    # last change position is input_ids.numel()
    next_change = torch.cat([
        change_idx[1:],
        torch.tensor([input_ids.numel()], device=device, dtype=change_idx.dtype),
    ])
    ends = next_change[keep]

    vals = code[starts]
    starts_cpu = starts.tolist()
    ends_cpu = ends.tolist()
    vals_cpu = vals.tolist()
    return [(int(s), int(e), int(v)) for s, e, v in zip(starts_cpu, ends_cpu, vals_cpu)]


def reorganize_inputs(
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor = None,
    pixel_values_videos: torch.Tensor = None,
    image_grid_thw: torch.Tensor = None,
    video_grid_thw: torch.Tensor = None,
    image_input_mask: torch.Tensor = None,
    video_input_mask: torch.Tensor = None,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    square_merge_size: int = 4,
):
    if pixel_values is None:
        if video_input_mask is None and pixel_values_videos is not None:
            video_input_mask = (input_ids == video_token_id).contiguous()
        return pixel_values_videos, video_grid_thw, video_input_mask

    if pixel_values_videos is None:
        if image_input_mask is None and pixel_values is not None:
            image_input_mask = (input_ids == image_token_id).contiguous()
        return pixel_values, image_grid_thw, image_input_mask

    image_thw_cpu = image_grid_thw.tolist()
    video_thw_cpu = video_grid_thw.tolist()
    vision_indexs = find_vision_id_index(input_ids.view(-1), image_token_id, video_token_id)
    len_split = sum([thw[0] for thw in image_thw_cpu])
    len_split += sum([thw[0] for thw in video_thw_cpu])
    assert len_split == len(vision_indexs)

    vision_values = []
    vision_grid_thw = []
    idx = 0
    video_idx = 0
    image_idx = 0
    video_seqlen = 0
    image_seqlen = 0
    while idx < len(vision_indexs):
        start, end, token_id = vision_indexs[idx]
        if token_id == image_token_id:
            seqlen = 0
            thw = image_thw_cpu[image_idx]
            for i in range(thw[0]):
                start, end, token_id = vision_indexs[idx + i]
                assert token_id == image_token_id
                seqlen += (end - start) * square_merge_size
            assert seqlen == thw[0] * thw[1] * thw[2]
            vision_values.append(pixel_values[image_seqlen:(image_seqlen + seqlen)])
            vision_grid_thw.append(thw)

            image_idx += 1
            idx += thw[0]
            image_seqlen += seqlen
        elif token_id == video_token_id:
            seqlen = 0
            thw = video_thw_cpu[video_idx]
            for i in range(thw[0]):
                start, end, token_id = vision_indexs[idx + i]
                assert token_id == video_token_id
                seqlen += (end - start) * square_merge_size
            assert seqlen == thw[0] * thw[1] * thw[2]
            vision_values.append(pixel_values_videos[video_seqlen:(video_seqlen + seqlen)])
            vision_grid_thw.append(thw)

            video_idx += 1
            idx += thw[0]
            video_seqlen += seqlen
        else:
            assert False, f"should not have {token_id=}"

    if video_input_mask is None:
        video_input_mask = input_ids == video_token_id

    if image_input_mask is None:
        image_input_mask = input_ids == image_token_id

    vision_values = torch.cat(vision_values)
    vision_grid_thw = torch.tensor(vision_grid_thw,
                                   device=image_grid_thw.device,
                                   dtype=image_grid_thw.dtype)
    vision_input_mask = (video_input_mask | image_input_mask)

    return vision_values, vision_grid_thw, vision_input_mask
