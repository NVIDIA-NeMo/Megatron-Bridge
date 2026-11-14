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

"""Two-rank tests: multi-LoRA on tensor-parallel bases matches a dense single-device reference.

Each rank holds a shard of every adapter's A and B. The forward output, the
per-rank A/B gradient shards and the input gradient must all equal the
corresponding slices of an unsharded computation.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/peft/test_multi_lora_tp_distributed.py
"""

import os
from collections.abc import Iterator

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, set_tokens_per_adapter_slot
from megatron.bridge.peft.utils import init_method_normal


_TP_SIZE = 2
_HIDDEN_SIZE = 8
_OUTPUT_SIZE = 8
_LORA_RANK = 4
_LORA_ALPHA = 8
_TOKENS_PER_SLOT = (3, 2)


@pytest.fixture(scope="module", autouse=True)
def distributed_tp() -> Iterator[None]:
    if int(os.environ.get("WORLD_SIZE", "1")) != _TP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    owns_process_group = not dist.is_initialized()
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.backends.cuda.matmul.allow_tf32 = False
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


def _random(*shape: int, seed: int) -> torch.Tensor:
    """Draw the same tensor on every rank."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)).cuda()


def _shard(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor.chunk(_TP_SIZE, dim=dim)[dist.get_rank()].contiguous()


def _build(base_kind: str) -> MultiLoRALinear:
    config = TransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=2,
        tensor_model_parallel_size=_TP_SIZE,
        sequence_parallel=False,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
    )
    if base_kind == "column":
        base = ColumnParallelLinear(
            _HIDDEN_SIZE,
            _OUTPUT_SIZE,
            config=config,
            init_method=init_method_normal(0.02),
            bias=False,
            gather_output=False,
        )
        name = "linear_fc1"
    else:
        base = RowParallelLinear(
            _HIDDEN_SIZE,
            _OUTPUT_SIZE,
            config=config,
            init_method=init_method_normal(0.02),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
        )
        name = "linear_fc2"
    wrapped = MultiLoRALinear(
        to_wrap=base.cuda(),
        n_adapters=len(_TOKENS_PER_SLOT),
        dim=_LORA_RANK,
        alpha=_LORA_ALPHA,
        full_name=name,
        column_init_method="xavier",
        row_init_method="zero",
        dropout=0.0,
    )
    wrapped.adapters.cuda()
    return wrapped


def run_against_dense_reference(base_kind: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Run one TP-sharded forward/backward and its dense reference.

    Returns ``{name: (actual, expected)}`` for this rank's output, input gradient and
    adapter weight-gradient shards.
    """
    column = base_kind == "column"
    wrapped = _build(base_kind)

    base_weight = _random(_OUTPUT_SIZE, _HIDDEN_SIZE, seed=1)
    with torch.no_grad():
        wrapped.to_wrap.weight.copy_(_shard(base_weight, dim=0 if column else 1))

    full_weights = []
    for slot, adapter in enumerate(wrapped.adapters):
        full_a = _random(_LORA_RANK, _HIDDEN_SIZE, seed=10 + slot)
        full_b = _random(_OUTPUT_SIZE, _LORA_RANK, seed=20 + slot)
        with torch.no_grad():
            adapter.linear_in.weight.copy_(_shard(full_a, dim=0 if column else 1))
            adapter.linear_out.weight.copy_(_shard(full_b, dim=0))
        wrapped.init_adapter_slot(slot, rank=_LORA_RANK, alpha=_LORA_ALPHA)
        full_weights.append((full_a, full_b))

    total_tokens = sum(_TOKENS_PER_SLOT)
    x_full = _random(total_tokens, 1, _HIDDEN_SIZE, seed=30)
    grad_full = _random(total_tokens, 1, _OUTPUT_SIZE, seed=31)
    x_local = (x_full if column else _shard(x_full, dim=-1)).clone().requires_grad_()
    grad_local = _shard(grad_full, dim=-1) if column else grad_full

    set_tokens_per_adapter_slot(wrapped, torch.tensor(_TOKENS_PER_SLOT, device="cuda", dtype=torch.int32))
    actual, _ = wrapped(x_local)
    actual.backward(grad_local)

    x_reference = x_full.clone().requires_grad_()
    a_references = [full_a.clone().requires_grad_() for full_a, _ in full_weights]
    b_references = [full_b.clone().requires_grad_() for _, full_b in full_weights]
    scaling = _LORA_ALPHA / _LORA_RANK
    adapter_reference = torch.cat(
        [
            scaling * F.linear(F.linear(x_part, a_reference), b_reference)
            for x_part, a_reference, b_reference in zip(
                x_reference.split(_TOKENS_PER_SLOT, dim=0), a_references, b_references, strict=True
            )
        ],
        dim=0,
    )
    reference = F.linear(x_reference, base_weight) + adapter_reference
    x_grad, *weight_grads = torch.autograd.grad(
        (reference * grad_full).sum(), (x_reference, *a_references, *b_references)
    )
    a_grads = weight_grads[: len(full_weights)]
    b_grads = weight_grads[len(full_weights) :]

    results = {
        "output": (actual, _shard(reference, dim=-1) if column else reference),
        "input_grad": (x_local.grad, x_grad if column else _shard(x_grad, dim=-1)),
    }
    for slot, adapter in enumerate(wrapped.adapters):
        results[f"lora_A[{slot}]_grad"] = (
            adapter.linear_in.weight.grad,
            _shard(a_grads[slot], dim=0 if column else 1),
        )
        results[f"lora_B[{slot}]_grad"] = (adapter.linear_out.weight.grad, _shard(b_grads[slot], dim=0))
    return results


@pytest.mark.gpu
@pytest.mark.parametrize("base_kind", ["column", "row"])
def test_multi_lora_tp_sharded_matches_dense_forward_and_gradients(base_kind: str) -> None:
    for name, (actual, expected) in run_against_dense_reference(base_kind).items():
        assert actual is not None, name
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5, msg=lambda m, n=name: f"{n}: {m}")
