# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from megatron.bridge.models.conversion.model_bridge import MegatronWeightTuple, WeightConversionTask
from megatron.bridge.models.conversion.peft_bridge import AdapterWeight, AdapterWeightConversionTask
from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_experts,rank,hidden_size,ffn_size,expert_tp_size",
    [(2, 3, 4, 3, 1), (3, 2, 5, 5, 1), (5, 4, 3, 2, 1), (3, 4, 6, 4, 2), (2, 3, 6, 6, 3)],
)
@pytest.mark.parametrize("with_megatron_names", [False, True])
def test_gpt_oss_expert_adapter_export_preserves_forward(
    monkeypatch, num_experts, rank, hidden_size, ffn_size, expert_tp_size, with_megatron_names
):
    """Export native FC1/FC2 factors, then compare real HF experts with native LoRA math."""
    generator = torch.Generator().manual_seed(42)

    def random_tensor(*shape):
        return torch.randn(*shape, generator=generator) * 0.2

    bridge = GPTOSSBridge()
    tasks = {}
    weights = {}
    factors = {}
    scale = 2
    for layer, in_features, out_features in [
        ("linear_fc1", hidden_size, 2 * ffn_size),
        ("linear_fc2", ffn_size, hidden_size),
    ]:
        prefix = f"decoder.layers.0.mlp.experts.{layer}"
        in_name = f"{prefix}.adapter.linear_in.weight"
        out_name = f"{prefix}.adapter.linear_out.weight"
        # Noncontiguous factors also occur when expert tensors are sliced/views.
        lora_a = random_tensor(num_experts, in_features, rank).transpose(1, 2)
        lora_b = random_tensor(num_experts, rank, out_features).transpose(1, 2)
        factors[layer] = (lora_a, lora_b)
        if layer == "linear_fc1":
            # Transport gathers each shard's [gate, up] rows in rank order.
            gate, up = lora_b.chunk(2, dim=1)
            materialized_b = torch.cat(
                [
                    torch.cat([gate_shard, up_shard], dim=1)
                    for gate_shard, up_shard in zip(gate.chunk(expert_tp_size, dim=1), up.chunk(expert_tp_size, dim=1))
                ],
                dim=1,
            )
        else:
            materialized_b = lora_b
        tasks[prefix] = [
            AdapterWeightConversionTask(
                global_base_prefix=prefix,
                adapter_key=None,
                alpha=scale * rank,
                dim=rank,
                linear_in_task=WeightConversionTask(param_name=in_name, global_param_name=in_name, mapping=Mock()),
                linear_out_task=WeightConversionTask(param_name=out_name, global_param_name=out_name, mapping=Mock()),
            )
        ]
        weights[prefix] = AdapterWeight(
            global_base_prefix=prefix,
            adapter_key=None,
            alpha=scale * rank,
            dim=rank,
            linear_in_weight=MegatronWeightTuple(in_name, lora_a, vp_stage=0),
            linear_out_weight=MegatronWeightTuple(out_name, materialized_b, vp_stage=0),
        )

    # Replace transport/materialization only: use the real registry, export stream,
    # projection dispatch, expert stacking, names, and optional source metadata.
    monkeypatch.setattr(bridge, "build_adapter_conversion_tasks", lambda *args, **kwargs: tasks)
    monkeypatch.setattr(
        bridge, "materialize_adapter_weights", lambda batch: [weights[t.global_base_prefix] for t in batch]
    )
    monkeypatch.setattr(
        "megatron.bridge.models.conversion.peft_bridge.parallel_state.get_expert_model_parallel_world_size", lambda: 1
    )
    exported = list(
        bridge.stream_adapter_weights_megatron_to_hf(
            [
                SimpleNamespace(
                    config=SimpleNamespace(num_moe_experts=num_experts, expert_tensor_parallel_size=expert_tp_size)
                )
            ],
            show_progress=False,
            with_megatron_names=with_megatron_names,
        )
    )
    tensors = {weight.param_name: weight.weight for weight in exported}
    assert len(tensors) == 4
    hf_prefix = "model.layers.0.mlp.experts"
    for layer, projection in [("linear_fc1", "gate_up_proj"), ("linear_fc2", "down_proj")]:
        torch.testing.assert_close(tensors[f"{hf_prefix}.{projection}.lora_A.weight"], factors[layer][0])
        if projection == "down_proj":
            torch.testing.assert_close(tensors[f"{hf_prefix}.{projection}.lora_B.weight"], factors[layer][1])
        if with_megatron_names:
            for side, source in [("A", "linear_in"), ("B", "linear_out")]:
                weight = next(w for w in exported if w.param_name == f"{hf_prefix}.{projection}.lora_{side}.weight")
                assert weight.megatron_param_names == (
                    f"decoder.layers.0.mlp.experts.{layer}.adapter.{source}.weight",
                )

    config = GptOssConfig(hidden_size=hidden_size, intermediate_size=ffn_size, num_local_experts=num_experts)
    config._experts_implementation = "eager"
    original = GptOssExperts(config).eval()
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.copy_(random_tensor(*parameter.shape))
    adapted = copy.deepcopy(original)
    with torch.no_grad():
        for projection in ["gate_up_proj", "down_proj"]:
            lora_a = tensors[f"{hf_prefix}.{projection}.lora_A.weight"]
            lora_b = tensors[f"{hf_prefix}.{projection}.lora_B.weight"]
            # HF GPT-OSS stores [E, in, out]; exported factors stay [E, r, in]/[E, out, r].
            getattr(adapted, projection).add_(scale * torch.bmm(lora_b, lora_a).transpose(1, 2))

    x = random_tensor(2 * num_experts, hidden_size)
    router_indices = torch.arange(2 * num_experts).remainder(num_experts).unsqueeze(1)
    routing_weights = torch.ones(2 * num_experts, 1)
    expected = torch.empty_like(x)
    fc1_a, fc1_b = factors["linear_fc1"]
    fc2_a, fc2_b = factors["linear_fc2"]
    for token, expert in enumerate(router_indices[:, 0]):
        # Independent oracle: Megatron FC1 emits [all gates, all ups]. Never
        # construct expected tensors using the exporter's row permutation.
        gate_delta, up_delta = (scale * (x[token] @ fc1_a[expert].T) @ fc1_b[expert].T).chunk(2)
        gate = x[token] @ original.gate_up_proj[expert, :, ::2] + original.gate_up_proj_bias[expert, ::2]
        up = x[token] @ original.gate_up_proj[expert, :, 1::2] + original.gate_up_proj_bias[expert, 1::2]
        gate = (gate + gate_delta).clamp(max=original.limit)
        up = (up + up_delta).clamp(min=-original.limit, max=original.limit)
        activated = (up + 1) * gate * torch.sigmoid(original.alpha * gate)
        expected[token] = (
            activated @ original.down_proj[expert]
            + original.down_proj_bias[expert]
            + scale * (activated @ fc2_a[expert].T) @ fc2_b[expert].T
        )
    torch.testing.assert_close(adapted(x, router_indices, routing_weights), expected, atol=1e-6, rtol=1e-5)
