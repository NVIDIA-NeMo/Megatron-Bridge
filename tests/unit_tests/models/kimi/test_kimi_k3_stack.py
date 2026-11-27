# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU regression coverage for Kimi depth state, recomputation and stage boundaries."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from megatron.bridge.models.kimi.kimi_k3_ops import KimiRMSNorm
from megatron.bridge.models.kimi.kimi_k3_pipeline import unpack_stage_boundary
from megatron.bridge.models.kimi.modeling_kimi_k3 import stack as stack_module


pytestmark = pytest.mark.unit
_PATTERN = "K-KE+EKEKE+E"


class _Branch(nn.Module):
    def __init__(self, config, *, layer_number=0, **kwargs):
        super().__init__()
        self.layer_number = layer_number
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)

    def set_layer_number(self, layer_number):
        self.layer_number = layer_number

    def forward(self, inputs, **kwargs):
        return torch.tanh(self.projection(inputs)) / self.layer_number, None


def _norm(*, config, **kwargs):
    return KimiRMSNorm(config.hidden_size, config.layernorm_epsilon, device="cpu", dtype=torch.float32)


@pytest.fixture
def cpu_stack_builders(monkeypatch):
    """Replace CUDA kernels/builders, retaining the production stack/layer math."""

    def residual_parameters(module, config, branch):
        module.add_module(f"{branch}_res_norm", _norm(config=config))
        module.add_module(f"{branch}_res_proj", nn.Linear(config.hidden_size, 1, bias=False))

    def specs(config, **kwargs):
        return [
            SimpleNamespace(
                submodules=SimpleNamespace(
                    input_layernorm=None,
                    pre_mlp_layernorm=None,
                    self_attention=None,
                    mlp=partial(_Branch, submodules=SimpleNamespace(linear_fc1=None)),
                )
            )
            for _ in range(config.num_layers)
        ]

    def build(spec, **kwargs):
        return _Branch(**kwargs) if "layer_number" in kwargs else _norm(**kwargs)

    checkpoint_calls = []

    def checkpoint(run, distribute, prefix, bank):
        assert not distribute
        checkpoint_calls.append(bank.shape[-2])
        return torch_checkpoint(run, prefix, bank, use_reentrant=True)

    monkeypatch.setattr(stack_module, "_residual_parameters", residual_parameters)
    monkeypatch.setattr(stack_module, "get_gpt_decoder_layer_specs", specs)
    monkeypatch.setattr(stack_module, "build_module", build)
    monkeypatch.setattr(stack_module, "TENorm", _norm)
    monkeypatch.setattr(stack_module, "_with_situ_activation", lambda spec: spec)
    monkeypatch.setattr(stack_module, "checkpoint", checkpoint)
    return checkpoint_calls


def _stack(*, offset=0, blocks=6, method=None, chunk=3):
    config = SimpleNamespace(
        hidden_size=8,
        layernorm_epsilon=1e-5,
        params_dtype=torch.float32,
        kimi_attn_res_block_size=2,
        recompute_granularity="full" if method else None,
        recompute_method=method,
        recompute_num_layers=chunk,
        distribute_saved_activations=False,
    )
    return stack_module.KimiK3Stack(
        config,
        pattern=_PATTERN,
        layer_config_list=[None] * (2 * blocks),
        pp_layer_offset=2 * offset,
        pre_process=offset == 0,
        post_process=offset + blocks == 6,
        pg_collection=SimpleNamespace(tp=None),
    )


def _copy_stage_weights(control, stage):
    for local, layer in enumerate(stage.layers):
        layer.load_state_dict(control.layers[stage.pp_layer_offset + local].state_dict())
    if stage.post_process:
        for name in ("output_attn_res_proj", "output_attn_res_norm", "final_norm"):
            getattr(stage, name).load_state_dict(getattr(control, name).state_dict())


def _logical_parameters(stages):
    parameters = {}
    for stage in stages:
        for name, parameter in stage.named_parameters():
            if name.startswith("layers."):
                _, index, suffix = name.split(".", 2)
                name = f"layers.{int(index) + stage.pp_layer_offset}.{suffix}"
            assert name not in parameters
            parameters[name] = parameter
    return parameters


@pytest.mark.parametrize("method,chunk", [(None, 3), ("uniform", 2), ("uniform", 3), ("block", 2), ("block", 3)])
@pytest.mark.parametrize("partition", [(6,), (3, 3), (4, 2)])
def test_depth_state_outputs_and_gradients_match_control(cpu_stack_builders, method, chunk, partition):
    torch.manual_seed(123)
    control = _stack()
    stages = []
    offset = 0
    for blocks in partition:
        stage = _stack(offset=offset, blocks=blocks, method=method, chunk=chunk)
        _copy_stage_weights(control, stage)
        assert [layer.source_index for layer in stage.layers] == [
            i // 2 for i in range(2 * offset, 2 * (offset + blocks))
        ]
        assert [
            layer.self_attention.layer_number if layer.attention else layer.mlp.layer_number for layer in stage.layers
        ] == [i // 2 + 1 for i in range(2 * offset, 2 * (offset + blocks))]
        stages.append(stage)
        offset += blocks

    source = torch.randn(3, 2, 8, requires_grad=True)
    candidate_input = source.detach().clone().requires_grad_()
    expected = control(source)
    actual = candidate_input
    for stage in stages:
        if not stage.pre_process:
            stage.set_input_tensor(actual)
        actual = stage(actual if stage.pre_process else None)
        if not stage.post_process:
            complete_blocks = stage.pp_layer_offset // 2 + len(stage.layers) // 2
            _, bank = unpack_stage_boundary(actual, 8, (complete_blocks + 1) // 2)
            assert bank.requires_grad
    cotangent = torch.randn_like(expected)
    (expected * cotangent).sum().backward()
    (actual * cotangent).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(candidate_input.grad, source.grad, rtol=1e-6, atol=1e-6)
    expected_parameters = dict(control.named_parameters())
    actual_parameters = _logical_parameters(stages)
    assert expected_parameters.keys() == actual_parameters.keys()
    for name, parameter in expected_parameters.items():
        other = actual_parameters[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all() and torch.isfinite(other.grad).all(), name
            torch.testing.assert_close(other.grad, parameter.grad, rtol=1e-6, atol=1e-6, msg=name)
    if method:
        assert cpu_stack_builders
        assert max(cpu_stack_builders) >= 1


def test_snapshot_bank_grows_only_at_attention_residual_boundaries(cpu_stack_builders):
    model = _stack()
    prefix = torch.randn(3, 2, 8, requires_grad=True)
    bank = prefix.new_empty(3, 2, 0, 8)
    for index, layer in enumerate(model.layers):
        previous = bank
        prefix, bank = layer(prefix, bank)
        assert bank.shape[-2] == index // 4 + 1
        if index % 4 == 0:
            assert bank is not previous
            torch.testing.assert_close(bank[..., :-1, :], previous)
        else:
            assert bank is previous
        assert bank.requires_grad
