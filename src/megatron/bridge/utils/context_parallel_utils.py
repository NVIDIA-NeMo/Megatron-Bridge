import torch

from megatron.core import parallel_state as mpu


def expand_thw(thw: torch.Tensor) -> torch.Tensor:
    assert thw.dim() == 2
    repeats = thw[:, 0].to(torch.long)
    assert torch.all(repeats > 0), "thw[:,0] must be > 0"

    idx = torch.arange(thw.size(0), device=thw.device).repeat_interleave(repeats)
    out = thw[idx].clone()
    out[:, 0] = 1
    return out


def collapse_thw(expanded: torch.Tensor) -> torch.Tensor:
    assert expanded.dim() == 2
    assert expanded.size(1) >= 2
    if expanded.shape[0] < 2:
        return expanded

    # find the diff
    other = expanded[:, 1:]
    prev = torch.cat([other[:1], other[:-1]], dim=0)
    change = (other != prev).any(dim=1)
    # the index0 must be now row
    change[0] = True  

    # find the diff
    starts = torch.nonzero(change, as_tuple=False).squeeze(1)
    ends = torch.cat([starts[1:], torch.tensor([other.size(0)], device=other.device)]) - 1
    counts = ends - starts + 1

    rows_other = other[starts]
    result_first_col = counts.to(expanded.dtype).unsqueeze(1)
    result = torch.cat([result_first_col, rows_other], dim=1)
    return result


# also can use in qwen2vl/qwen2.5vl
def qwen2vl_parallel_split(
    parallel_size: int,
    pixel_values: list[torch.Tensor],
    image_grid_thws: list[torch.Tensor],
):
    assert len(pixel_values) == len(image_grid_thws)
    # split the pixel_values
    split_pixel_values = []
    split_image_grid_thws = []
    for pixel_value, image_grid_thw in zip(pixel_values, image_grid_thws):
        split_image_grid_thw = list(torch.split(image_grid_thw, 1, dim=0))
        split_image_grid_thws.extend(split_image_grid_thw)
        slice_begin = 0
        for ele in split_image_grid_thw:
            slice_end = slice_begin + ele.prod().item()
            split_pixel_values.append(pixel_value[slice_begin:slice_end].clone())
            slice_begin = slice_end

    pixel_values = split_pixel_values
    image_grid_thws = split_image_grid_thws
    img_num = len(image_grid_thws)

    img_num_per_rank = img_num // parallel_size
    img_num_remain = img_num % parallel_size
    cp_img_num = []
    for i in range(parallel_size):
        cp_img_num.append(img_num_per_rank)
        if i < img_num_remain:
            cp_img_num[i] += 1

    img_idx = 0
    new_pixel_values = []
    new_image_grid_thws = []
    for i in range(parallel_size):
        seq_len = 0
        img_begin_idx = img_idx
        img_end_idx = img_begin_idx + cp_img_num[i]
        img_idx += cp_img_num[i]

        for j in range(img_begin_idx, img_end_idx):
            seq_len += pixel_values[j].size(0)
            new_pixel_values.append(pixel_values[j])
            new_image_grid_thws.append(image_grid_thws[j])

    return new_pixel_values, new_image_grid_thws, cp_img_num


@torch.no_grad
def qwen3vl_parallel_split(
    parallel_size: int,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
):
    assert parallel_size > 1
    if pixel_values is None:
        assert image_grid_thw is None
        return None, None, None

    assert not pixel_values.requires_grad
    assert not image_grid_thw.requires_grad
    # expand video thw
    image_grid_thw = expand_thw(image_grid_thw)

    new_pixel_values, new_image_grid_thws, parallel_img_num = (
        qwen2vl_parallel_split(
            parallel_size,
            [pixel_values],
            [image_grid_thw],
        )
    )
    pixel_values = torch.cat(new_pixel_values, dim=0)
    image_grid_thw = torch.cat(new_image_grid_thws, dim=0)
    return pixel_values, image_grid_thw, parallel_img_num


def get_vision_parallel_data(
    vision_data: torch.Tensor,
    vision_grid_thw: torch.Tensor,
    square_merge_size: int,
    parallel_img_num: list[int],
):
    """Get vision data and grid_thw for context parallelism.
    Returns:
        vision_data (torch.Tensor): Vision data of shape [total_thw_size, n_features].
        vision_grid_thw (torch.Tensor): Vision grid_thw of shape [total_thw_size, 3].
        seqlens_list (list of torch.Tensor): List of seqlens of the vision data in each context parallel rank,
                                             for the all gather after vision encoder.
    """
    # we use the context parallelism size and context parallel group of LLM for vision model.
    # we only divide the number of images in each context parallel rank.
    parallel_size = mpu.get_tensor_and_context_parallel_world_size()
    parallel_rank = mpu.get_tensor_and_context_parallel_rank()
    assert parallel_size == len(parallel_img_num), f'{parallel_size=} {len(parallel_img_num)=}'

    seqlens = torch.repeat_interleave(
        vision_grid_thw[:, 1] * vision_grid_thw[:, 2], vision_grid_thw[:, 0]
    )
    vision_grid_thw_list = []
    vision_data_list = []
    seqlens_list = []
    img_idx = 0
    for i in range(parallel_size):
        start_idx = img_idx
        end_idx = start_idx + parallel_img_num[i]
        img_idx += parallel_img_num[i]

        vision_grid_thw_list.append(vision_grid_thw[start_idx:end_idx])
        seqlens_list.append(seqlens[start_idx:end_idx])
        data_start_idx = seqlens[:start_idx].sum()
        data_end_idx = seqlens[:end_idx].sum()
        vision_data_list.append(vision_data[data_start_idx:data_end_idx])
    new_vision_grid_thw = vision_grid_thw_list[parallel_rank]
    new_vision_data = vision_data_list[parallel_rank]
    new_seqlens_list = [t // square_merge_size for t in seqlens_list]
    return new_vision_data, new_vision_grid_thw, new_seqlens_list


class AllGatherVisionEmbeddings(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, seqlens_on_parallel_ranks, parallel_group):
        outputs = []
        for i in range(len(seqlens_on_parallel_ranks)):
            o = torch.zeros(
                (seqlens_on_parallel_ranks[i].sum(), *input.shape[1:]),
                device=input.device,
                dtype=input.dtype,
                layout=input.layout,
            )
            outputs.append(o)
        torch.distributed.all_gather(
            outputs, input, group=parallel_group
        )
        parallel_rank = parallel_group.rank()
        ctx.parallel_rank = parallel_rank
        ctx.save_for_backward(*seqlens_on_parallel_ranks)

        output = torch.cat(outputs, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        parallel_rank = ctx.parallel_rank
        seqlens_on_parallel_ranks = ctx.saved_tensors
        start_idx = (
            torch.cat(seqlens_on_parallel_ranks[:parallel_rank]).sum() if parallel_rank != 0 else 0
        )
        end_idx = start_idx + seqlens_on_parallel_ranks[parallel_rank].sum()
        grad_output = grad_output[start_idx:end_idx]
        return grad_output, None, None


def split_data_cp_rank(
    val: torch.Tensor, cp_size: int, seq_dim: int, cp_rank: int = None
):
    assert cp_size > 1
    assert 0 == val.shape[seq_dim] % (2 * cp_size), f"{val.shape=} {cp_size=}"
    if cp_rank is None:
        cp_rank = mpu.get_context_parallel_rank()
    if val is None:
        return val
    val = val.view(
        *val.shape[0:seq_dim],
        2 * cp_size,
        val.shape[seq_dim] // (2 * cp_size),
        *val.shape[(seq_dim + 1) :],
    )
    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
    val = val.index_select(seq_dim, index)
    val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])

    return val
