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

"""Exchange compact ERNIE expert counts between context ranks before BF16 accumulation."""

import torch
import torch.distributed as dist

from .context_parallel import zigzag_cp_geometry


@torch.no_grad()
def contiguous_owner_histogram(indices, padding_mask, packed, cp_group, num_experts):
    """Count the locally selected experts by contiguous global CP owner. Ignore router padding."""
    if packed is None or getattr(packed, "qkv_format", None) != "thd":
        raise ValueError("Contiguous CP router counts require packed THD metadata on every routed layer.")
    cp_size, cp_rank = cp_group.size(), cp_group.rank()
    if getattr(packed, "local_cp_size", None) is not None:
        raise ValueError("Contiguous CP router counts require a static context-parallel group.")
    if indices.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Contiguous CP router counts require the router to execute outside CUDA graph capture.")
    nrows = indices.size(0)
    if not nrows:
        raise ValueError("Contiguous CP router counts require at least one local packed row.")
    layout = getattr(packed, "cp_partition_mode", None)
    if layout == "zigzag":
        geometry = zigzag_cp_geometry(
            packed, cp_group=cp_group, local_rows=nrows, device=indices.device, what="ERNIE expert-bias counts"
        )
        owner = torch.div(geometry.query_positions, nrows, rounding_mode="floor")
        bins = indices + (owner * num_experts).unsqueeze(1)
    elif layout == "contiguous":
        bins = indices + cp_rank * num_experts
    else:
        raise ValueError(f"Unsupported packed CP layout for contiguous router counts: {layout!r}.")
    if padding_mask is None:
        weights = torch.ones_like(indices, dtype=torch.int32)
    else:
        if padding_mask.numel() != nrows:
            raise ValueError("The router padding mask and the physical CP rows must have the same length.")
        weights = (~padding_mask.reshape(-1, 1)).to(torch.int32).expand_as(indices)
    counts = torch.zeros(cp_size * num_experts, device=indices.device, dtype=torch.int32)
    counts.scatter_add_(0, bins.flatten(), weights.reshape(-1))
    return counts.view(cp_size, num_experts)


@torch.no_grad()
def _prepare_contiguous_cp_counts(routers):
    """Redistribute all local layers and microbatches at once, then accumulate in local call order."""
    group = routers[0].cp_group
    microbatches = len(routers[0]._ernie_cp_bias_histograms)
    if any(r.cp_group is not group or len(r._ernie_cp_bias_histograms) != microbatches for r in routers):
        raise RuntimeError(
            "Contiguous CP router counts require equal microbatch counts and one static CP group per stage."
        )
    # Send [destination CP owner, layer, microbatch, expert] in native logical layer order.
    send = torch.stack([torch.stack(r._ernie_cp_bias_histograms, dim=1) for r in routers], dim=1)
    received = torch.empty_like(send[0])
    dist.reduce_scatter_tensor(received, send.flatten(0, 1).contiguous(), group=group)
    for router, counts in zip(routers, received):
        local = router.local_tokens_per_expert
        if local.dtype != torch.bfloat16:
            raise RuntimeError("Contiguous CP router counts require Megatron-Core's BF16 router counter.")
        for microbatch_counts in counts.unbind(0):
            local.add_(microbatch_counts.long())
        router._ernie_cp_bias_histograms.clear()


def finalize_with_contiguous_cp_counts(finalize, model, *args, **kwargs):
    """Recreate reference CP count ownership, then run the configured native gradient finalizer."""
    routers = [
        router for chunk in model for router in chunk.modules() if getattr(router, "_ernie_cp_bias_histograms", None)
    ]
    if routers:
        _prepare_contiguous_cp_counts(routers)
    return finalize(model, *args, **kwargs)
