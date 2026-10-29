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

"""Two-rank parity test for LoRA sequence-parallel input re-gather.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/peft/test_lora_sp_input_regather_distributed.py
"""

import os

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.peft.utils import ParallelLinearAdapter


_TP_SIZE = 2


def _make_adapter(
    pg_collection: ProcessGroupCollection,
    *,
    sequence_parallel_input_regather: bool,
) -> ParallelLinearAdapter:
    """Construct the real MCore-backed adapter used by the parity test."""
    config = ModelParallelConfig(
        tensor_model_parallel_size=_TP_SIZE,
        sequence_parallel=True,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=True,
    )
    return ParallelLinearAdapter(
        in_features=8,
        out_features=8,
        dim=4,
        base_linear_name="decoder.layers.0.self_attention.linear_qkv",
        activation="identity",
        input_is_parallel=False,
        model_parallel_config=config,
        disable_sequence_parallel_comm=False,
        sequence_parallel_input_regather=sequence_parallel_input_regather,
        pg_collection=pg_collection,
    )


def _set_nonzero_weights(adapter: ParallelLinearAdapter) -> None:
    """Set deterministic nonzero weights so backward exercises both LoRA projections."""
    with torch.no_grad():
        for index, parameter in enumerate(adapter.parameters(), start=1):
            values = torch.arange(1, parameter.numel() + 1, device=parameter.device, dtype=parameter.dtype)
            parameter.copy_(values.reshape_as(parameter) * (0.01 * index))


def _run_microbatches(
    adapter: ParallelLinearAdapter,
    local_inputs: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Run two backward passes and snapshot fused main gradients after each pass."""
    for parameter in adapter.parameters():
        parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)

    outputs: list[torch.Tensor] = []
    input_grads: list[torch.Tensor] = []
    main_grads: list[dict[str, torch.Tensor]] = []
    for microbatch, local_input in enumerate(local_inputs, start=1):
        x = local_input.detach().clone().requires_grad_(True)
        output = adapter(x)
        output_grad = torch.arange(
            1,
            output.numel() + 1,
            device=output.device,
            dtype=output.dtype,
        ).reshape_as(output)
        output.backward(output_grad * microbatch)

        outputs.append(output.detach().clone())
        input_grads.append(x.grad.detach().clone())
        main_grads.append(
            {name: parameter.main_grad.detach().clone() for name, parameter in adapter.named_parameters()}
        )

    return outputs, input_grads, main_grads


@pytest.mark.gpu
def test_sequence_parallel_input_regather_distributed_backward_parity() -> None:
    """Input re-gather should match the baseline through fused two-microbatch backward."""
    if int(os.environ.get("WORLD_SIZE", "1")) != _TP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    owns_process_group = not dist.is_initialized()
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if owns_process_group:
        dist.init_process_group(backend="nccl")
    if dist.get_world_size() != _TP_SIZE:
        pytest.skip("requires a two-rank process group")

    try:
        if owns_model_parallel:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=_TP_SIZE,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
            )
        model_parallel_cuda_manual_seed(2026, force_reset_rng=True)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        baseline = _make_adapter(pg_collection, sequence_parallel_input_regather=False)
        regather = _make_adapter(pg_collection, sequence_parallel_input_regather=True)
        _set_nonzero_weights(baseline)
        regather.load_state_dict(baseline.state_dict())

        rank = dist.get_rank()
        input_values = torch.arange(1, 49, device="cuda", dtype=torch.float32).reshape(3, 2, 8)
        local_inputs = [input_values + rank, input_values.flip(0) + rank + 0.5]

        baseline_results = _run_microbatches(baseline, local_inputs)
        regather_results = _run_microbatches(regather, local_inputs)

        for baseline_tensors, regather_tensors in zip(baseline_results[:2], regather_results[:2], strict=True):
            for baseline_tensor, regather_tensor in zip(baseline_tensors, regather_tensors, strict=True):
                torch.testing.assert_close(regather_tensor, baseline_tensor, rtol=0, atol=0)

        baseline_main_grads = baseline_results[2]
        regather_main_grads = regather_results[2]
        for microbatch in range(2):
            assert baseline_main_grads[microbatch].keys() == regather_main_grads[microbatch].keys()
            for name in baseline_main_grads[microbatch]:
                torch.testing.assert_close(
                    regather_main_grads[microbatch][name],
                    baseline_main_grads[microbatch][name],
                    rtol=0,
                    atol=0,
                )

        assert any(
            not torch.equal(baseline_main_grads[0][name], baseline_main_grads[1][name])
            for name in baseline_main_grads[0]
        ), "the second microbatch must accumulate into fused main_grad"
        assert all(parameter.grad is None for parameter in baseline.parameters())
        assert all(parameter.grad is None for parameter in regather.parameters())
    finally:
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
