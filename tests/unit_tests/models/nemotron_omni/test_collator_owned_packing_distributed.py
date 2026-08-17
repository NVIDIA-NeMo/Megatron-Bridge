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

"""Two-rank CP check for collator-owned Nemotron Omni THD batches.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/models/nemotron_omni/test_collator_owned_packing_distributed.py
"""

import os

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from torch import nn

from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel


_CP_SIZE = 2


@pytest.mark.gpu
def test_collator_owned_thd_tensors_use_one_real_cp_partition_index() -> None:
    """MCore's THD index must shard every token-aligned tensor identically."""
    if int(os.environ.get("WORLD_SIZE", "1")) != _CP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    owns_process_group = not dist.is_initialized()
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if owns_process_group:
        dist.init_process_group(backend="nccl")

    try:
        if owns_model_parallel:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=_CP_SIZE,
            )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        model = NemotronOmniModel.__new__(NemotronOmniModel)
        nn.Module.__init__(model)
        model.context_parallel_lm = _CP_SIZE
        model.pg_collection = pg_collection

        input_ids = torch.tensor([[7, 18, 9, 0, 11, 12, 13, 0]], device="cuda")
        combined_embeddings = input_ids.transpose(0, 1).unsqueeze(-1).to(dtype=torch.float32)
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device="cuda")
        labels = input_ids.clone()
        loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]], device="cuda")
        padding_mask = torch.tensor(
            [[False, False, False, True, False, False, False, True]],
            device="cuda",
        )
        cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
        cu_seqlens_padded = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=4,
            max_seqlen_kv=4,
            total_tokens=8,
        )

        (
            local_input_ids,
            local_embeddings,
            local_position_ids,
            local_attention_mask,
            local_labels,
            local_loss_mask,
            local_padding_mask,
            loss_mask_was_sliced,
        ) = model._apply_context_parallel_sharding(
            input_ids=input_ids,
            combined_embeddings=combined_embeddings,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            packed_seq_params=packed_seq_params,
        )

        expected_index = (
            torch.tensor([0, 3, 4, 7], device="cuda")
            if dist.get_rank() == 0
            else torch.tensor([1, 2, 5, 6], device="cuda")
        )
        assert torch.equal(local_input_ids, input_ids.index_select(1, expected_index))
        assert torch.equal(local_embeddings, combined_embeddings.index_select(0, expected_index))
        assert torch.equal(local_position_ids, position_ids.index_select(1, expected_index))
        assert torch.equal(local_labels, labels.index_select(1, expected_index))
        assert torch.equal(local_loss_mask, loss_mask.index_select(1, expected_index))
        assert torch.equal(local_padding_mask, padding_mask.index_select(1, expected_index))
        assert local_attention_mask is None
        assert loss_mask_was_sliced is True
        assert packed_seq_params.cu_seqlens_q is cu_seqlens
        assert packed_seq_params.cu_seqlens_q_padded is cu_seqlens_padded
    finally:
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
