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

"""Two-rank ModelOpt export parity across TP, PP, and EP collectives.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/models/test_modelopt_utils_distributed.py
"""

import copy
import os
from types import SimpleNamespace

import modelopt.torch.quantization as mtq
import pytest
import torch
import torch.distributed as dist
from modelopt.torch.export.quantized_weight import (
    build_hf_quantization_config,
    capture_quantized_weight_export_state,
    export_quantized_weight,
)
from modelopt.torch.utils.distributed import ParallelState

from megatron.bridge.models.conversion import model_bridge as model_bridge_utils
from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
from megatron.bridge.models.conversion.modelopt_utils import build_modelopt_export_plan
from megatron.bridge.models.conversion.param_mapping import AutoMapping


_WORLD_SIZE = 2


def _quantized_linear(
    weight: torch.Tensor,
    *,
    weight_amax: float,
    input_amax: float,
) -> torch.nn.Module:
    module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype)
    mtq.quantize(
        module,
        copy.deepcopy(mtq.NVFP4_DEFAULT_CFG),
        lambda model: model(torch.ones(1, weight.shape[1], device=weight.device, dtype=weight.dtype)),
    )
    with torch.no_grad():
        module.weight.copy_(weight)
    module.weight_quantizer.amax = torch.tensor(weight_amax, device=weight.device)
    module.input_quantizer.amax = torch.tensor(input_amax, device=weight.device)
    return module


def _task(
    module: torch.nn.Module,
    global_name: str,
    hf_name: str,
    *,
    local_name: str | None = None,
    grouped: bool = False,
) -> WeightConversionTask:
    return WeightConversionTask(
        param_name=local_name or global_name,
        global_param_name=global_name,
        mapping=SimpleNamespace(
            hf_param=hf_name,
            is_grouped_export=grouped,
            ep_size=_WORLD_SIZE if grouped else 1,
            transpose_on_export=False,
        ),
        megatron_module=module,
        param_weight=module.weight,
    )


def _model(*, num_moe_experts: int | None = None) -> list[torch.nn.Module]:
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_moe_experts=num_moe_experts)
    return [model]


def _capture_state(
    weight: torch.Tensor,
    *,
    weight_amax: float,
    input_amax: float,
):
    module = _quantized_linear(
        weight,
        weight_amax=weight_amax,
        input_amax=input_amax,
    )
    return capture_quantized_weight_export_state(module)


def _canonical_export(
    hf_name: str,
    weight: torch.Tensor,
    *,
    weight_amax: float,
    input_amax: float,
) -> dict[str, torch.Tensor]:
    state = _capture_state(
        weight,
        weight_amax=weight_amax,
        input_amax=input_amax,
    )
    prefix = hf_name.removesuffix("weight")
    return {
        f"{prefix}{name}": tensor
        for name, tensor in export_quantized_weight(
            weight,
            state,
            dtype=weight.dtype,
        )
        .named_tensors()
        .items()
    }


def _assert_tensors_equal(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        assert torch.equal(actual[name], expected[name]), name


@pytest.mark.gpu
def test_modelopt_export_tp2_pp2_ep2_matches_canonical_export(monkeypatch) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != _WORLD_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    owns_process_group = not dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if owns_process_group:
        dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        singleton_groups = [dist.new_group(ranks=[group_rank]) for group_rank in range(_WORLD_SIZE)]
        local_group = singleton_groups[rank]
        world_group = dist.group.WORLD

        # TP: each rank owns one input shard, while scalar calibration state is max-reduced.
        local_tp_weight = torch.arange(64, device="cuda", dtype=torch.bfloat16).reshape(4, 16) / 8 + rank
        tp_module = _quantized_linear(
            local_tp_weight,
            weight_amax=(2.0, 5.0)[rank],
            input_amax=(3.0, 7.0)[rank],
        )
        tp_module.parallel_state = ParallelState(
            data_parallel_group=local_group,
            tensor_parallel_group=world_group,
        )
        tp_hf_name = "model.layers.0.mlp.up_proj.weight"
        tp_task = _task(tp_module, "decoder.layers.0.mlp.linear_fc1.weight", tp_hf_name)
        monkeypatch.setattr(model_bridge_utils, "_get_pp_group", lambda _model: local_group)
        monkeypatch.setattr(model_bridge_utils, "_get_ep_group", lambda _model: local_group)
        tp_plan = build_modelopt_export_plan([tp_task], model=_model())
        gathered_tp_weights = [torch.empty_like(local_tp_weight) for _ in range(_WORLD_SIZE)]
        dist.all_gather(gathered_tp_weights, local_tp_weight)
        canonical_tp_weight = torch.cat(gathered_tp_weights, dim=1)

        tp_stream = tp_plan.conversion_tasks[0].export_hook(tp_hf_name, canonical_tp_weight)
        assert not isinstance(tp_stream, tuple)
        _assert_tensors_equal(
            dict(tp_stream),
            _canonical_export(
                tp_hf_name,
                canonical_tp_weight,
                weight_amax=5.0,
                input_amax=7.0,
            ),
        )

        # PP: disjoint canonical layers and their config records are exchanged across stages.
        pp_weight = torch.arange(64, device="cuda", dtype=torch.bfloat16).reshape(4, 16) / 8 + rank
        pp_module = _quantized_linear(
            pp_weight,
            weight_amax=2.0 + rank,
            input_amax=3.0 + rank,
        )
        pp_global_name = f"decoder.layers.{rank}.mlp.linear_fc1.weight"
        pp_hf_name = f"model.layers.{rank}.mlp.up_proj.weight"
        pp_task = _task(pp_module, pp_global_name, pp_hf_name)
        monkeypatch.setattr(model_bridge_utils, "_get_pp_group", lambda _model: world_group)
        monkeypatch.setattr(model_bridge_utils, "_get_ep_group", lambda _model: local_group)
        pp_plan = build_modelopt_export_plan([pp_task], model=_model())
        expected_config = build_hf_quantization_config(
            {
                f"model.layers.{layer}.mlp.up_proj": _capture_state(
                    torch.ones(4, 16, device="cuda", dtype=torch.bfloat16),
                    weight_amax=2.0 + layer,
                    input_amax=3.0 + layer,
                )
                for layer in range(_WORLD_SIZE)
            }
        )
        assert pp_plan.quantization_config == expected_config
        _assert_tensors_equal(
            dict(pp_plan.conversion_tasks[0].export_hook(pp_hf_name, pp_weight)),
            _canonical_export(
                pp_hf_name,
                pp_weight,
                weight_amax=2.0 + rank,
                input_amax=3.0 + rank,
            ),
        )

        # EP: expert calibration remains distinct through exchange and packing happens per expert.
        expert_weight = torch.arange(64, device="cuda", dtype=torch.bfloat16).reshape(4, 16) / 8 + rank
        expert_module = _quantized_linear(
            expert_weight,
            weight_amax=2.0 + rank,
            input_amax=3.0 + rank,
        )
        expert_global_name = f"decoder.layers.0.mlp.experts.linear_fc2.weight{rank}"
        expert_hf_name = f"model.layers.0.mlp.experts.{rank}.down_proj.weight"
        expert_task = WeightConversionTask(
            param_name="decoder.layers.0.mlp.experts.linear_fc2.weight0",
            global_param_name=expert_global_name,
            mapping=AutoMapping(expert_global_name, expert_hf_name),
            megatron_module=expert_module,
            param_weight=expert_module.weight,
        )
        monkeypatch.setattr(model_bridge_utils, "_get_pp_group", lambda _model: local_group)
        monkeypatch.setattr(model_bridge_utils, "_get_ep_group", lambda _model: world_group)
        ep_plan = build_modelopt_export_plan(
            [expert_task],
            model=_model(num_moe_experts=_WORLD_SIZE),
        )
        gathered_expert_weights = [torch.empty_like(expert_weight) for _ in range(_WORLD_SIZE)]
        dist.all_gather(gathered_expert_weights, expert_weight)
        ep_tensors = dict(
            ep_plan.conversion_tasks[0].export_hook(
                expert_hf_name,
                expert_weight,
            )
        )
        terminal_experts = [
            _canonical_export(
                f"model.layers.0.mlp.experts.{expert}.down_proj.weight",
                gathered_expert_weights[expert],
                weight_amax=2.0 + expert,
                input_amax=3.0 + expert,
            )
            for expert in range(_WORLD_SIZE)
        ]
        expected_ep_tensors = {name: tensor for expert in terminal_experts for name, tensor in expert.items()}
        _assert_tensors_equal(ep_tensors, expected_ep_tensors)

        # Grouped EP: exchange per-expert state, pack each expert independently,
        # then stack every canonical tensor family under the grouped HF name.
        grouped_hf_name = "model.layers.0.mlp.experts.down_proj.weight"
        grouped_task = _task(
            expert_module,
            expert_global_name,
            grouped_hf_name,
            grouped=True,
        )
        grouped_plan = build_modelopt_export_plan(
            [grouped_task],
            model=_model(num_moe_experts=_WORLD_SIZE),
        )
        grouped_bridge = object.__new__(model_bridge_utils.MegatronModelBridge)
        grouped_tensors = grouped_bridge._accumulate_grouped_export(
            grouped_plan.conversion_tasks[0],
            {grouped_hf_name: torch.stack(gathered_expert_weights)},
            SimpleNamespace(num_moe_experts=_WORLD_SIZE),
            {},
            {},
        )
        grouped_experts = [
            _canonical_export(
                grouped_hf_name,
                gathered_expert_weights[expert],
                weight_amax=2.0 + expert,
                input_amax=3.0 + expert,
            )
            for expert in range(_WORLD_SIZE)
        ]
        expected_grouped_tensors = {
            name: torch.stack([expert[name] for expert in grouped_experts]) for name in grouped_experts[0]
        }
        assert grouped_tensors is not None
        _assert_tensors_equal(grouped_tensors, expected_grouped_tensors)
    finally:
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
