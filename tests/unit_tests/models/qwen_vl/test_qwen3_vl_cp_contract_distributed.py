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

"""Two-rank context-parallel input contract for Qwen3-VL.

Qwen3VLModel shards the sequence itself, so a forward step must hand it full-length
``input_ids`` and no ``position_ids`` -- which is what ``qwen3_vl_step`` does. A step that
CP-shards the batch first (the GPT step) makes the model shard a second time.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/models/qwen_vl/test_qwen3_vl_cp_contract_distributed.py
"""

import os

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_batch_on_this_cp_rank

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import _split_if_full_sequence
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import Qwen3VLMultimodalRotaryEmbedding


_CP_SIZE = 2
_SEQ = 4096
_KV_CHANNELS = 128


@pytest.fixture(scope="module")
def cp_group():
    """Context-parallel group over both ranks, built once for the module."""
    if int(os.environ.get("WORLD_SIZE", "1")) != _CP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    owns_process_group = not dist.is_initialized()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if owns_process_group:
        dist.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(context_parallel_size=_CP_SIZE)
    try:
        yield ProcessGroupCollection.use_mpu_process_groups().cp
    finally:
        parallel_state.destroy_model_parallel()
        if owns_process_group:
            dist.destroy_process_group()


def _model_sequence_length(cp_group, input_ids: torch.Tensor) -> int:
    """Sequence length the model's language stack sees, per Qwen3VLModel.forward."""
    sequence_length, _ = _split_if_full_sequence(
        input_ids,
        cp_size=_CP_SIZE,
        seq_dim=1,
        cp_rank=cp_group.rank(),
        full_sequence_length=input_ids.size(1),
    )
    return sequence_length.size(1)


@pytest.mark.gpu
def test_full_length_input_ids_leave_the_model_at_one_shard(cp_group) -> None:
    """The contract `qwen3_vl_step` satisfies: hand the model the whole sequence."""
    full_input_ids = torch.arange(_SEQ, device="cuda").view(1, _SEQ)

    assert _model_sequence_length(cp_group, full_input_ids) == _SEQ // _CP_SIZE


@pytest.mark.gpu
def test_pre_sharded_input_ids_shard_twice(cp_group) -> None:
    """Why the GPT step cannot drive this model: its batch is already CP-sharded."""
    batch = {"tokens": torch.arange(_SEQ, device="cuda").view(1, _SEQ), "attention_mask": None}
    pre_sharded = get_batch_on_this_cp_rank(batch, is_hybrid_cp=False, cp_group=cp_group)["tokens"]
    assert pre_sharded.size(1) == _SEQ // _CP_SIZE

    assert _model_sequence_length(cp_group, pre_sharded) == _SEQ // _CP_SIZE**2


@pytest.mark.gpu
def test_rope_matches_hidden_states_for_full_length_position_ids(cp_group) -> None:
    """The rotary embedding shards once too, so it lines up with the hidden states."""
    rotary = Qwen3VLMultimodalRotaryEmbedding(kv_channels=_KV_CHANNELS, cp_group=cp_group)
    position_ids = torch.arange(_SEQ, device="cuda").view(1, -1).expand(3, 1, -1).contiguous()

    assert rotary(position_ids, rotary.mrope_section).shape[0] == _SEQ // _CP_SIZE
