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

"""Two-rank parity test for FC2 LoRA under shared-expert TP overlap.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/peft/test_shared_expert_lora_tp_distributed.py
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.peft.utils import ParallelLinearAdapter


_TP_SIZE = 2


@contextmanager
def _distributed_tp() -> Iterator[ProcessGroupCollection]:
    """Initialize the minimum two-rank TP topology used by this test."""
    owns_process_group = not dist.is_initialized()
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if owns_process_group:
        dist.init_process_group(backend="nccl")
    if owns_model_parallel:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=_TP_SIZE,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
    model_parallel_cuda_manual_seed(2026, force_reset_rng=True)

    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _make_fc2_adapter(
    pg_collection: ProcessGroupCollection,
    *,
    overlap: bool,
    sequence_parallel: bool,
    base_linear_name: str = "decoder.layers.0.mlp.shared_experts.linear_fc2",
) -> ParallelLinearAdapter:
    """Construct the row-parallel adapter used by shared-expert FC2."""
    config = ModelParallelConfig(
        tensor_model_parallel_size=_TP_SIZE,
        sequence_parallel=sequence_parallel,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
    )
    return ParallelLinearAdapter(
        in_features=8,
        out_features=8,
        dim=4,
        base_linear_name=base_linear_name,
        activation="identity",
        input_is_parallel=True,
        model_parallel_config=config,
        alpha=4,
        disable_tensor_parallel_comm=overlap,
        disable_sequence_parallel_comm=overlap,
        pg_collection=pg_collection,
    )


def _set_nonzero_weights(adapter: ParallelLinearAdapter) -> None:
    """Set deterministic nonzero weights on each TP shard."""
    rank = dist.get_rank()
    with torch.no_grad():
        for index, parameter in enumerate(adapter.parameters(), start=1):
            values = torch.arange(
                1,
                parameter.numel() + 1,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.copy_(values.reshape_as(parameter) * (0.01 * index) + rank)


@pytest.mark.gpu
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_shared_expert_fc2_overlap_matches_standard_tp_forward_and_backward(
    sequence_parallel: bool,
) -> None:
    """External overlap reduction must preserve standard FC2 LoRA values and gradients."""
    if int(os.environ.get("WORLD_SIZE", "1")) != _TP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    with _distributed_tp() as pg_collection:
        standard = _make_fc2_adapter(pg_collection, overlap=False, sequence_parallel=sequence_parallel)
        overlap = _make_fc2_adapter(pg_collection, overlap=True, sequence_parallel=sequence_parallel)
        routed_expert = _make_fc2_adapter(
            pg_collection,
            overlap=True,
            sequence_parallel=sequence_parallel,
            base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
        )
        assert routed_expert._external_tp_reduce_scale == 1.0
        _set_nonzero_weights(standard)
        overlap.load_state_dict(standard.state_dict())

        rank = dist.get_rank()
        values = torch.arange(1, 33, device="cuda", dtype=torch.float32).reshape(4, 2, 4)
        standard_input = (values + rank).requires_grad_(True)
        overlap_input = standard_input.detach().clone().requires_grad_(True)

        standard_output = standard(standard_input)
        overlap_output = overlap(overlap_input)
        if sequence_parallel:
            overlap_output = reduce_scatter_to_sequence_parallel_region(overlap_output, group=pg_collection.tp)
        else:
            overlap_output = reduce_from_tensor_model_parallel_region(overlap_output, group=pg_collection.tp)
        torch.testing.assert_close(overlap_output, standard_output, rtol=1e-6, atol=1e-6)

        output_grad = torch.arange(
            1,
            standard_output.numel() + 1,
            device=standard_output.device,
            dtype=standard_output.dtype,
        ).reshape_as(standard_output)
        standard_output.backward(output_grad)
        overlap_output.backward(output_grad)

        torch.testing.assert_close(overlap_input.grad, standard_input.grad, rtol=1e-6, atol=1e-6)
        assert dict(overlap.named_parameters()).keys() == dict(standard.named_parameters()).keys()
        for name, parameter in standard.named_parameters():
            overlap_parameter = dict(overlap.named_parameters())[name]
            assert parameter.grad is not None
            assert overlap_parameter.grad is not None
            torch.testing.assert_close(overlap_parameter.grad, parameter.grad, rtol=1e-6, atol=1e-6)
