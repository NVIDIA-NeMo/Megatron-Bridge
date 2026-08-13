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

"""Two-rank parity tests for shared expert LoRA parameters.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/peft/test_shared_expert_lora_distributed.py
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
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.peft.utils import ParallelLinearAdapter


_EP_SIZE = 2
_ADAPTER_CASES = (
    ("decoder.layers.0.mlp.experts.linear_fc1", False),
    ("decoder.layers.0.mlp.experts.linear_fc2", True),
)


@contextmanager
def _distributed_ep() -> Iterator[ProcessGroupCollection]:
    """Initialize the minimum two-rank EP topology used by these tests."""
    owns_process_group = not dist.is_initialized()
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if owns_process_group:
        dist.init_process_group(backend="nccl")
    if owns_model_parallel:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=_EP_SIZE,
            expert_tensor_parallel_size=1,
        )
    model_parallel_cuda_manual_seed(2026, force_reset_rng=True)

    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _make_adapter(
    pg_collection: ProcessGroupCollection,
    *,
    base_linear_name: str,
    input_is_parallel: bool,
    use_cpu_initialization: bool = False,
) -> ParallelLinearAdapter:
    """Construct a real MCore-backed shared expert adapter."""
    config = ModelParallelConfig(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=_EP_SIZE,
        expert_tensor_parallel_size=1,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=use_cpu_initialization,
    )
    return ParallelLinearAdapter(
        in_features=8,
        out_features=8,
        dim=4,
        base_linear_name=base_linear_name,
        activation="identity",
        input_is_parallel=input_is_parallel,
        is_expert=True,
        model_parallel_config=config,
        pg_collection=pg_collection,
    )


def _assert_identical_across_ep(tensor: torch.Tensor, ep_group: object) -> None:
    """Require bitwise equality for one logical tensor on every EP rank."""
    collective_tensor = tensor if tensor.is_cuda else tensor.cuda()
    gathered = [torch.empty_like(collective_tensor) for _ in range(_EP_SIZE)]
    dist.all_gather(gathered, collective_tensor, group=ep_group)
    for replica in gathered[1:]:
        torch.testing.assert_close(replica, gathered[0], rtol=0, atol=0)


def _set_identical_nonzero_weights(adapter: ParallelLinearAdapter) -> None:
    """Make gradients for both LoRA projections nonzero on the first backward."""
    with torch.no_grad():
        for index, parameter in enumerate(adapter.parameters(), start=1):
            values = torch.arange(1, parameter.numel() + 1, device=parameter.device, dtype=parameter.dtype)
            parameter.copy_(values.reshape_as(parameter) * (0.01 * index))


@pytest.fixture(scope="module")
def ep_pg_collection() -> Iterator[ProcessGroupCollection]:
    """Provide one shared two-rank EP topology for this test module."""
    if int(os.environ.get("WORLD_SIZE", "1")) != _EP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    with _distributed_ep() as pg_collection:
        yield pg_collection


@pytest.mark.gpu
def test_shared_expert_lora_initialization_is_identical_across_ep(
    ep_pg_collection: ProcessGroupCollection,
) -> None:
    """Every shared expert LoRA parameter must start bitwise identical across EP."""
    for base_linear_name, input_is_parallel in _ADAPTER_CASES:
        adapter = _make_adapter(
            ep_pg_collection,
            base_linear_name=base_linear_name,
            input_is_parallel=input_is_parallel,
        )
        for parameter in adapter.parameters():
            _assert_identical_across_ep(parameter.detach(), ep_pg_collection.ep)


@pytest.mark.gpu
def test_shared_expert_lora_gradients_are_identical_across_ep(
    ep_pg_collection: ProcessGroupCollection,
) -> None:
    """EP gradient synchronization must produce identical nonzero gradients."""
    ep_rank = dist.get_rank(group=ep_pg_collection.ep)
    for base_linear_name, input_is_parallel in _ADAPTER_CASES:
        adapter = _make_adapter(
            ep_pg_collection,
            base_linear_name=base_linear_name,
            input_is_parallel=input_is_parallel,
        )
        _set_identical_nonzero_weights(adapter)
        inputs = torch.arange(1, 25, device="cuda", dtype=torch.float32).reshape(3, 8) + ep_rank
        adapter(inputs).square().sum().backward()

        for parameter in adapter.parameters():
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) > 0
            _assert_identical_across_ep(parameter.grad, ep_pg_collection.ep)


@pytest.mark.gpu
def test_shared_expert_lora_cpu_initialization_is_identical_across_ep(
    ep_pg_collection: ProcessGroupCollection,
) -> None:
    """CPU initialization must be synchronized when EP ranks use different seeds."""
    ep_rank = dist.get_rank(group=ep_pg_collection.ep)
    for base_linear_name, input_is_parallel in _ADAPTER_CASES:
        torch.manual_seed(2026 + ep_rank)
        adapter = _make_adapter(
            ep_pg_collection,
            base_linear_name=base_linear_name,
            input_is_parallel=input_is_parallel,
            use_cpu_initialization=True,
        )
        for parameter in adapter.parameters():
            assert parameter.device.type == "cpu"
            _assert_identical_across_ep(parameter.detach(), ep_pg_collection.ep)
