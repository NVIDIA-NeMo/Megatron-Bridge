# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from megatron.core import parallel_state as ps
from megatron.core.packed_seq_params import PackedSeqParams


def dit_data_step(qkv_format, dataloader_iter):
    batch = next(dataloader_iter)
    batch["is_preprocessed"] = True  # assume data is preprocessed
    batch = {k: v.to(device="cuda", non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    batch = encode_seq_length(batch, format=qkv_format)
    batch = get_batch_on_this_cp_rank(batch)
    return batch


def encode_seq_length(batch, format):
    if ("seq_len_q" in batch) and ("seq_len_kv" in batch):
        zero = torch.zeros([1], dtype=torch.int32, device="cuda")

        def cumsum(key):
            return torch.cat((zero, batch[key].cumsum(dim=0).to(torch.int32)))

        cu_seqlens_q = cumsum("seq_len_q")
        cu_seqlens_kv = cumsum("seq_len_kv")
        cu_seqlens_q_padded = cumsum("seq_len_q_padded")
        cu_seqlens_kv_padded = cumsum("seq_len_kv_padded")

        batch["packed_seq_params"] = {
            "self_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_q,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_q_padded,
                qkv_format=format,
            ),
            "cross_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                qkv_format=format,
            ),
        }

    return batch


def get_batch_on_this_cp_rank(data):
    """Split the data for context parallelism."""
    cp_size = ps.get_context_parallel_world_size()
    if cp_size > 1:
        import transformer_engine_torch as tex

        cp_rank = ps.get_context_parallel_rank()
        for key in ["video", "loss_mask", "pos_ids"]:
            if data[key] is not None:
                index = tex.thd_get_partitioned_indices(
                    data["packed_seq_params"]["self_attention"].cu_seqlens_q_padded,
                    data[key].size(1),
                    cp_size,
                    cp_rank,
                ).to(device=data[key].device, dtype=torch.long)
                data[key] = data[key].index_select(1, index).contiguous()

        for key in ["context_embeddings", "context_mask"]:
            if data[key] is not None:
                index = tex.thd_get_partitioned_indices(
                    data["packed_seq_params"]["cross_attention"].cu_seqlens_kv, data[key].size(1), cp_size, cp_rank
                ).to(device=data[key].device, dtype=torch.long)
                data[key] = data[key].index_select(1, index).contiguous()

    return data
