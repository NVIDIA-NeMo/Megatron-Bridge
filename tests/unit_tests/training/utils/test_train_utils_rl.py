# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import torch

from megatron.bridge.training.utils.train_utils import LinearForLastLayer, freeze_moe_router, make_value_model


def test_linear_for_last_layer_matches_megatron_output_contract():
    layer = LinearForLastLayer(input_size=3, output_size=2, sequence_parallel=False)
    output, bias = layer(torch.ones(4, 3), weight=torch.empty(0), runtime_gather_output=True)

    assert output.shape == (4, 2)
    assert output.dtype == torch.float32
    assert bias is None


def test_linear_for_last_layer_gathers_sequence_parallel_output():
    layer = LinearForLastLayer(input_size=3, output_size=2, sequence_parallel=True)
    gathered = torch.randn(8, 2)

    with patch(
        "megatron.bridge.training.utils.train_utils.tensor_parallel.gather_from_sequence_parallel_region",
        return_value=gathered,
    ) as gather:
        output, _ = layer(torch.ones(4, 3))

    assert output is gathered
    assert layer.weight.sequence_parallel is True
    gather.assert_called_once()
    assert gather.call_args.kwargs == {"tensor_parallel_output_grad": False}


def test_freeze_moe_router_freezes_router_and_shared_expert_gate():
    router = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(1)), bias=torch.nn.Parameter(torch.ones(1)))
    shared = SimpleNamespace(
        gate_weight=torch.nn.Parameter(torch.ones(1)),
        gate_bias=torch.nn.Parameter(torch.ones(1)),
    )
    model = SimpleNamespace(
        decoder=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(router=router, shared_experts=shared))])
    )

    result = freeze_moe_router(model)

    assert result == [model]
    assert not router.weight.requires_grad
    assert not router.bias.requires_grad
    assert not shared.gate_weight.requires_grad
    assert not shared.gate_bias.requires_grad


def test_make_value_model_replaces_output_layer_on_last_pipeline_stage():
    model = SimpleNamespace(output_layer=None)

    with (
        patch("megatron.bridge.training.utils.train_utils._register_linear_for_last_layer_mapping"),
        patch("megatron.core.parallel_state.get_pipeline_model_parallel_world_size", return_value=1),
        patch("megatron.core.parallel_state.is_pipeline_last_stage", return_value=True),
    ):
        result = make_value_model(hidden_size=4, sequence_parallel=False)(model)

    assert result == [model]
    assert isinstance(model.output_layer, LinearForLastLayer)
    assert model.output_layer.in_features == 4
    assert model.output_layer.out_features == 1
