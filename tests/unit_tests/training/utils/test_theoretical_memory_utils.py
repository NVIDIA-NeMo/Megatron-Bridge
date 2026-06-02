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

import sys
from types import ModuleType, SimpleNamespace

import pytest

from megatron.bridge.training.utils.theoretical_memory_utils import (
    estimate_training_memory,
    format_training_memory_estimate,
    report_theoretical_memory,
)


def _make_config(**model_overrides):
    model_defaults = {
        "num_layers": 4,
        "hidden_size": 16,
        "seq_length": 8,
        "ffn_hidden_size": 64,
        "num_attention_heads": 4,
        "num_query_groups": None,
        "kv_channels": 4,
        "vocab_size": 32,
        "make_vocab_size_divisible_by": 1,
        "should_pad_vocab": False,
        "share_embeddings_and_output_weights": True,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 2,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "num_moe_experts": None,
        "moe_layer_freq": 1,
        "moe_router_topk": 1,
        "moe_ffn_hidden_size": None,
        "moe_shared_expert_intermediate_size": None,
        "moe_latent_size": None,
        "mtp_num_layers": None,
        "gated_linear_unit": False,
        "activation_func": None,
        "sequence_parallel": True,
        "recompute_granularity": "selective",
    }
    for key, value in model_overrides.items():
        if key not in model_defaults:
            raise ValueError(f"Config has no field '{key}'")
        model_defaults[key] = value
    return SimpleNamespace(
        model=SimpleNamespace(**model_defaults),
        train=SimpleNamespace(micro_batch_size=2),
        optimizer=SimpleNamespace(use_distributed_optimizer=True),
        data_parallel_size=4,
    )


def _mock_megatron_mimo_provider(monkeypatch):
    module_name = "megatron.bridge.models.megatron_mimo.megatron_mimo_provider"
    module = ModuleType(module_name)
    module.MegatronMIMOProvider = type("MegatronMIMOProvider", (), {})
    monkeypatch.setitem(sys.modules, module_name, module)


@pytest.mark.unit
def test_dense_model_state_estimate_matches_legacy_arithmetic():
    config = _make_config()

    estimate = estimate_training_memory(config, include_activation=False)

    assert len(estimate.model_state_components) == 1
    dense_component = estimate.model_state_components[0]
    assert dense_component.parameter_count == 13088
    assert dense_component.parameter_count_per_gpu == 3400
    assert dense_component.bytes_per_parameter == 9
    assert estimate.weight_and_optimizer_bytes == 30600
    assert estimate.total_parameters == 13088


@pytest.mark.unit
def test_moe_estimate_accounts_for_expert_parallel_sharding():
    config = _make_config(
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        num_moe_experts=8,
        moe_layer_freq=2,
        moe_ffn_hidden_size=32,
        expert_model_parallel_size=4,
    )
    config.data_parallel_size = 8

    estimate = estimate_training_memory(config, include_activation=False)

    assert len(estimate.model_state_components) == 2
    dense_component, routed_component = estimate.model_state_components
    assert dense_component.parameter_count == 8992
    assert dense_component.parameter_count_per_gpu == 4496
    assert dense_component.bytes_per_parameter == 6.75
    assert dense_component.memory_bytes == 30348
    assert routed_component.parameter_count == 16384
    assert routed_component.parameter_count_per_gpu == 4096
    assert routed_component.bytes_per_parameter == 7.5
    assert routed_component.memory_bytes == 30720
    assert estimate.weight_and_optimizer_bytes == 61068


@pytest.mark.unit
def test_activation_estimate_is_partitioned_by_tensor_and_context_parallelism():
    config = _make_config(pipeline_model_parallel_size=1, context_parallel_size=2)

    estimate = estimate_training_memory(config, num_microbatches=1)

    assert estimate.activation is not None
    assert estimate.activation.memory_bytes == 9568
    assert estimate.total_memory_bytes == estimate.weight_and_optimizer_bytes + 9568


@pytest.mark.unit
def test_moe_list_pattern_shared_expert_latent_and_mtp_counts():
    config = _make_config(
        pipeline_model_parallel_size=1,
        num_moe_experts=4,
        moe_layer_freq=[0, 1, 0, 1],
        moe_ffn_hidden_size=32,
        moe_shared_expert_intermediate_size=8,
        moe_latent_size=4,
        mtp_num_layers=1,
    )

    estimate = estimate_training_memory(config, include_activation=False)

    dense_component, routed_component = estimate.model_state_components
    assert dense_component.parameter_count == 11232
    assert dense_component.parameter_count_per_gpu == 5616
    assert routed_component.parameter_count == 3072
    assert routed_component.parameter_count_per_gpu == 3072
    assert estimate.total_parameters == 14304


@pytest.mark.unit
def test_activation_estimate_applies_virtual_pipeline_penalty():
    config = _make_config(virtual_pipeline_model_parallel_size=2)

    estimate = estimate_training_memory(config, num_microbatches=4)

    assert estimate.activation is not None
    assert estimate.activation.memory_bytes == 22240


@pytest.mark.unit
def test_format_training_memory_estimate_reports_components_and_units():
    config = _make_config()
    estimate = estimate_training_memory(config, include_activation=False)

    formatted = format_training_memory_estimate(estimate, unit="GB")

    assert "dense parameters and optimizer=0.00 GB" in formatted
    assert "total=0.00 GB" in formatted


@pytest.mark.unit
def test_format_training_memory_estimate_rejects_unknown_unit():
    estimate = estimate_training_memory(_make_config(), include_activation=False)

    with pytest.raises(ValueError, match="Unsupported memory unit"):
        format_training_memory_estimate(estimate, unit="bytes")


@pytest.mark.unit
def test_report_theoretical_memory_uses_structured_estimate(monkeypatch, capsys):
    _mock_megatron_mimo_provider(monkeypatch)
    config = _make_config()

    report_theoretical_memory(config, num_microbatches=4)

    captured = capsys.readouterr()
    assert "Theoretical memory footprints: weight and optimizer=0.03 MB" in captured.out
    assert "activation=0.02 MB" in captured.out
    assert "total=0.05 MB" in captured.out
