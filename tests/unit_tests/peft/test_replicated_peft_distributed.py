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

"""Two-rank correctness tests for PEFT on duplicated sequence-parallel linears.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/peft/test_replicated_peft_distributed.py
"""

import os
from collections.abc import Iterator

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.peft.dora import DoRA
from megatron.bridge.peft.dora_layers import DoRALinear
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.lora_layers import LoRALinear
from megatron.bridge.peft.multi_lora import MultiLoRA
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, set_tokens_per_adapter_slot


_TP_SIZE = 2
_HIDDEN_SIZE = 8
_OUTPUT_SIZE = 4
_LORA_RANK = 2
_LOCAL_TOKENS = 2


@pytest.fixture(scope="module", autouse=True)
def distributed_tp() -> Iterator[None]:
    if int(os.environ.get("WORLD_SIZE", "1")) != _TP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    owns_process_group = not dist.is_initialized()
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if owns_process_group:
        dist.init_process_group(backend="nccl")
    if owns_model_parallel:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=_TP_SIZE)
    model_parallel_cuda_manual_seed(2026, force_reset_rng=True)

    try:
        yield
    finally:
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=2,
        tensor_model_parallel_size=_TP_SIZE,
        sequence_parallel=True,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
    )


def _base_linear() -> TELinear:
    config = _config()
    return TELinear(
        _HIDDEN_SIZE,
        _OUTPUT_SIZE,
        parallel_mode="duplicated",
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        tp_group=None,
        name="linear_q_down_proj",
    )


def _full_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    total_tokens = _TP_SIZE * _LOCAL_TOKENS
    x = torch.arange(
        1,
        total_tokens * _HIDDEN_SIZE + 1,
        device="cuda",
        dtype=torch.float32,
    ).reshape(total_tokens, 1, _HIDDEN_SIZE)
    output_grad = torch.arange(
        1,
        total_tokens * _OUTPUT_SIZE + 1,
        device="cuda",
        dtype=torch.float32,
    ).reshape(total_tokens, 1, _OUTPUT_SIZE)
    return x / 10, output_grad / 100


def _logical_weight(rows: int, columns: int, offset: int) -> torch.Tensor:
    return (
        torch.arange(
            offset,
            offset + rows * columns,
            device="cuda",
            dtype=torch.float32,
        ).reshape(rows, columns)
        / 100
    )


def _copy_logical_weight(parameter: torch.Tensor, logical_weight: torch.Tensor) -> None:
    """Load either the pre-fix TP shard or the fixed replicated parameter."""
    with torch.no_grad():
        if parameter.shape == logical_weight.shape:
            parameter.copy_(logical_weight)
            return
        shards = logical_weight.chunk(_TP_SIZE, dim=0)
        assert parameter.shape == shards[dist.get_rank()].shape
        parameter.copy_(shards[dist.get_rank()])


def _assert_replicated_parameter(parameter: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    assert parameter.shape == expected_shape
    assert not getattr(parameter, "tensor_model_parallel", False)
    assert getattr(parameter, "sequence_parallel", False)


def _summed_grad(parameter: torch.Tensor) -> torch.Tensor:
    assert parameter.grad is not None
    grad = parameter.grad.detach().clone()
    dist.all_reduce(grad)
    return grad


@pytest.mark.gpu
def test_lora_replicated_base_matches_dense_forward_and_gradients() -> None:
    wrapped = LoRA(
        target_modules=["linear_q_down_proj"],
        dim=_LORA_RANK,
        alpha=_LORA_RANK,
        lora_A_init_method="xavier",
    ).transform(_base_linear(), name="linear_q_down_proj")
    assert isinstance(wrapped, LoRALinear)
    adapter = wrapped.adapter

    full_a = _logical_weight(_LORA_RANK, _HIDDEN_SIZE, 1)
    full_b = _logical_weight(_OUTPUT_SIZE, _LORA_RANK, 101)
    _copy_logical_weight(adapter.linear_in.weight, full_a)
    _copy_logical_weight(adapter.linear_out.weight, full_b)

    x_full, grad_full = _full_inputs()
    rank = dist.get_rank()
    x_local = x_full.chunk(_TP_SIZE, dim=0)[rank]
    grad_local = grad_full.chunk(_TP_SIZE, dim=0)[rank]
    actual = adapter(x_local)
    expected = F.linear(F.linear(x_local, full_a), full_b)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    actual.backward(grad_local)
    actual_a_grad = _summed_grad(adapter.linear_in.weight)
    actual_b_grad = _summed_grad(adapter.linear_out.weight)
    a_reference = full_a.detach().clone().requires_grad_()
    b_reference = full_b.detach().clone().requires_grad_()
    reference = F.linear(F.linear(x_full, a_reference), b_reference)
    reference_a_grad, reference_b_grad = torch.autograd.grad((reference * grad_full).sum(), (a_reference, b_reference))

    _assert_replicated_parameter(adapter.linear_in.weight, full_a.shape)
    _assert_replicated_parameter(adapter.linear_out.weight, full_b.shape)
    torch.testing.assert_close(actual_a_grad, reference_a_grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_b_grad, reference_b_grad, rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
def test_dora_replicated_base_matches_dense_forward_and_gradients() -> None:
    base = _base_linear()
    base_weight = _logical_weight(_OUTPUT_SIZE, _HIDDEN_SIZE, 201)
    with torch.no_grad():
        base.weight.copy_(base_weight)
    wrapped = DoRA(
        target_modules=["linear_q_down_proj"],
        dim=_LORA_RANK,
        alpha=_LORA_RANK,
        lora_A_init_method="xavier",
    ).transform(base, name="linear_q_down_proj")
    assert isinstance(wrapped, DoRALinear)
    adapter = wrapped.adapter

    full_a = _logical_weight(_LORA_RANK, _HIDDEN_SIZE, 301)
    full_b = _logical_weight(_OUTPUT_SIZE, _LORA_RANK, 401)
    _copy_logical_weight(adapter.linear_in.weight, full_a)
    _copy_logical_weight(adapter.linear_out.weight, full_b)
    direction = base_weight + full_b @ full_a
    magnitude = torch.linalg.norm(direction, dim=1) * torch.linspace(0.8, 1.1, _OUTPUT_SIZE, device="cuda")
    with torch.no_grad():
        adapter.weight_magnitude.copy_(magnitude)

    x_full, grad_full = _full_inputs()
    rank = dist.get_rank()
    x_local = x_full.chunk(_TP_SIZE, dim=0)[rank]
    grad_local = grad_full.chunk(_TP_SIZE, dim=0)[rank]
    base_output_local, _ = base(x_local)
    actual, _ = wrapped(x_local)
    magnitude_scale = (magnitude / torch.linalg.norm(direction, dim=1)).view(1, 1, -1)
    expected = (base_output_local.detach() + F.linear(F.linear(x_local, full_a), full_b)) * magnitude_scale
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    actual.backward(grad_local)
    actual_a_grad = _summed_grad(adapter.linear_in.weight)
    actual_b_grad = _summed_grad(adapter.linear_out.weight)
    actual_magnitude_grad = _summed_grad(adapter.weight_magnitude)
    a_reference = full_a.detach().clone().requires_grad_()
    b_reference = full_b.detach().clone().requires_grad_()
    magnitude_reference = magnitude.detach().clone().requires_grad_()
    direction_reference = base_weight + b_reference @ a_reference
    norm_reference = torch.linalg.norm(direction_reference, dim=1).detach()
    base_output_parts = [torch.empty_like(base_output_local) for _ in range(_TP_SIZE)]
    dist.all_gather(base_output_parts, base_output_local.detach())
    base_output_full = torch.cat(base_output_parts, dim=0)
    reference = (base_output_full + F.linear(F.linear(x_full, a_reference), b_reference)) * (
        magnitude_reference / norm_reference
    ).view(1, 1, -1)
    reference_grads = torch.autograd.grad(
        (reference * grad_full).sum(), (a_reference, b_reference, magnitude_reference)
    )

    _assert_replicated_parameter(adapter.linear_in.weight, full_a.shape)
    _assert_replicated_parameter(adapter.linear_out.weight, full_b.shape)
    _assert_replicated_parameter(adapter.weight_magnitude, magnitude.shape)
    torch.testing.assert_close(actual_a_grad, reference_grads[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_b_grad, reference_grads[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_magnitude_grad, reference_grads[2], rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
def test_multi_lora_replicated_base_matches_dense_forward_and_gradients() -> None:
    base = _base_linear()
    base_weight = _logical_weight(_OUTPUT_SIZE, _HIDDEN_SIZE, 501)
    with torch.no_grad():
        base.weight.copy_(base_weight)
    wrapped = MultiLoRA(
        target_modules=["linear_q_down_proj"],
        n_adapters=2,
        dim=_LORA_RANK,
        alpha=_LORA_RANK,
    ).transform(base, name="linear_q_down_proj")
    assert isinstance(wrapped, MultiLoRALinear)

    full_weights: list[tuple[torch.Tensor, torch.Tensor]] = []
    for slot, adapter in enumerate(wrapped.adapters):
        full_a = _logical_weight(_LORA_RANK, _HIDDEN_SIZE, 601 + slot * 200)
        full_b = _logical_weight(_OUTPUT_SIZE, _LORA_RANK, 701 + slot * 200)
        _copy_logical_weight(adapter.linear_in.weight, full_a)
        _copy_logical_weight(adapter.linear_out.weight, full_b)
        wrapped.init_adapter_slot(slot, rank=_LORA_RANK, alpha=_LORA_RANK)
        full_weights.append((full_a, full_b))

    x_full, grad_full = _full_inputs()
    rank = dist.get_rank()
    x_local = x_full.chunk(_TP_SIZE, dim=0)[rank]
    grad_local = grad_full.chunk(_TP_SIZE, dim=0)[rank]
    set_tokens_per_adapter_slot(
        wrapped,
        torch.tensor([_LOCAL_TOKENS, _LOCAL_TOKENS], device="cuda", dtype=torch.int32),
    )
    base_output_local, _ = base(x_local)
    actual, _ = wrapped(x_local)
    expected_adapter = torch.cat(
        [
            F.linear(F.linear(x_part, full_a), full_b)
            for x_part, (full_a, full_b) in zip(
                x_full.split([_LOCAL_TOKENS, _LOCAL_TOKENS], dim=0), full_weights, strict=True
            )
        ],
        dim=0,
    )
    expected_local = base_output_local.detach() + expected_adapter.chunk(_TP_SIZE, dim=0)[rank]
    torch.testing.assert_close(actual, expected_local, rtol=1e-5, atol=1e-5)

    actual.backward(grad_local)
    actual_grads: list[tuple[torch.Tensor, torch.Tensor]] = []
    for adapter in wrapped.adapters:
        actual_grads.append((_summed_grad(adapter.linear_in.weight), _summed_grad(adapter.linear_out.weight)))

    a_references = [full_a.detach().clone().requires_grad_() for full_a, _ in full_weights]
    b_references = [full_b.detach().clone().requires_grad_() for _, full_b in full_weights]
    reference_adapter = torch.cat(
        [
            F.linear(F.linear(x_part, a_reference), b_reference)
            for x_part, a_reference, b_reference in zip(
                x_full.split([_LOCAL_TOKENS, _LOCAL_TOKENS], dim=0),
                a_references,
                b_references,
                strict=True,
            )
        ],
        dim=0,
    )
    reference_grads = torch.autograd.grad(
        (reference_adapter * grad_full).sum(),
        (*a_references, *b_references),
    )

    for slot, adapter in enumerate(wrapped.adapters):
        full_a, full_b = full_weights[slot]
        _assert_replicated_parameter(adapter.linear_in.weight, full_a.shape)
        _assert_replicated_parameter(adapter.linear_out.weight, full_b.shape)
        torch.testing.assert_close(actual_grads[slot][0], reference_grads[slot], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            actual_grads[slot][1], reference_grads[len(wrapped.adapters) + slot], rtol=1e-5, atol=1e-5
        )
