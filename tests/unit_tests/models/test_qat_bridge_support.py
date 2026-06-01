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

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.conversion import model_bridge
from megatron.bridge.models.conversion import param_mapping as param_mapping_module
from megatron.bridge.models.conversion.param_mapping import AutoMapping, DirectMapping
from megatron.bridge.models.conversion.peft_bridge import MegatronPeftBridge
from megatron.bridge.models.conversion.utils import extract_sort_key
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


pytestmark = pytest.mark.unit


class _FakeGroup:
    def __init__(self, size: int, rank: int = 0) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


class _FakeModel(torch.nn.Module):
    def __init__(self, *, pp_size: int = 1, ep_size: int = 1, ep_rank: int = 0) -> None:
        super().__init__()
        self.pg_collection = SimpleNamespace(
            pp=_FakeGroup(pp_size),
            ep=_FakeGroup(ep_size, ep_rank),
        )


def test_extract_expert_number_supports_sequential_mlp_local_experts() -> None:
    assert extract_expert_number_from_param("decoder.layers.0.mlp.experts.local_experts.3.linear_fc1.weight") == 3
    assert extract_expert_number_from_param("decoder.layers.0.mlp.experts.linear_fc1.weight7") == 7
    assert extract_expert_number_from_param("model.layers.0.mlp.experts.11.gate_proj.weight") == 11


def test_extract_sort_key_supports_local_experts() -> None:
    assert extract_sort_key("decoder.layers.12.mlp.experts.local_experts.3.linear_fc1.weight") == (
        [12, 3],
        "decoder.layers.12.mlp.experts.local_experts.3.linear_fc1.weight",
    )


def test_local_expert_name_maps_to_global_expert_rank(monkeypatch) -> None:
    monkeypatch.setattr(model_bridge, "get_pg_size", lambda group: group.size())
    model = _FakeModel(ep_size=4, ep_rank=2)
    config = SimpleNamespace(num_moe_experts=8)

    global_name = model_bridge._megatron_local_name_to_global(
        [model],
        config,
        "decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight",
    )

    assert global_name == "decoder.layers.0.mlp.experts.local_experts.5.linear_fc1.weight"


def test_nested_local_expert_name_maps_to_global_expert_rank(monkeypatch) -> None:
    monkeypatch.setattr(model_bridge, "get_pg_size", lambda group: group.size())
    model = _FakeModel(ep_size=2, ep_rank=1)
    config = SimpleNamespace(num_moe_experts=4)

    global_name = model_bridge._megatron_local_name_to_global(
        [model],
        config,
        "decoder.layers.0.mlp.vision_moe_layer.experts.local_experts.1.linear_fc1.weight",
    )

    assert global_name == "decoder.layers.0.mlp.vision_moe_layer.experts.local_experts.3.linear_fc1.weight"


def test_local_expert_name_mapping_skips_adapter_params(monkeypatch) -> None:
    monkeypatch.setattr(model_bridge, "get_pg_size", lambda group: group.size())
    model = _FakeModel(ep_size=4, ep_rank=2)
    config = SimpleNamespace(num_moe_experts=8)
    param_name = "decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.adapter.linear_in.weight"

    assert model_bridge._megatron_local_name_to_global([model], config, param_name) == param_name


def test_grouped_expert_name_mapping_ignores_quantizer_params(monkeypatch) -> None:
    monkeypatch.setattr(model_bridge, "get_pg_size", lambda group: group.size())
    model = _FakeModel(ep_size=4, ep_rank=2)
    config = SimpleNamespace(num_moe_experts=8)
    param_name = "decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax"

    assert model_bridge._megatron_local_name_to_global([model], config, param_name) == param_name


def test_grouped_expert_name_maps_numbered_weight_and_bias(monkeypatch) -> None:
    monkeypatch.setattr(model_bridge, "get_pg_size", lambda group: group.size())
    model = _FakeModel(ep_size=4, ep_rank=2)
    config = SimpleNamespace(num_moe_experts=8)

    assert (
        model_bridge._megatron_local_name_to_global(
            [model],
            config,
            "decoder.layers.0.mlp.experts.linear_fc1.weight1",
        )
        == "decoder.layers.0.mlp.experts.linear_fc1.weight5"
    )
    assert (
        model_bridge._megatron_local_name_to_global(
            [model],
            config,
            "decoder.layers.0.mlp.experts.linear_fc1.bias1",
        )
        == "decoder.layers.0.mlp.experts.linear_fc1.bias5"
    )


def test_ep_gather_supports_sequential_mlp_local_experts(monkeypatch) -> None:
    monkeypatch.setattr(param_mapping_module, "get_pg_size", lambda group: group.size())
    mapping = DirectMapping(
        "decoder.layers.0.mlp.experts.local_experts.5.linear_fc1.weight",
        "model.layers.0.mlp.experts.5.gate_proj.weight",
    )
    mapping.ep_group = _FakeGroup(4)
    monkeypatch.setattr(mapping, "broadcast_obj_from_pp_rank", lambda obj, cache_key=None: obj)

    def fake_all_gather(output: list[torch.Tensor], tensor: torch.Tensor, group: _FakeGroup) -> None:
        del tensor, group
        for index, gathered_tensor in enumerate(output):
            gathered_tensor.fill_(index)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    result = mapping.gather_from_ep_ranks(
        torch.tensor([9.0]),
        SimpleNamespace(config=SimpleNamespace(num_moe_experts=8)),
        "model.layers.0.mlp.experts.5.gate_proj.weight",
    )

    assert list(result) == [
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.experts.5.gate_proj.weight",
        "model.layers.0.mlp.experts.7.gate_proj.weight",
    ]
    assert [tensor.item() for tensor in result.values()] == [0.0, 1.0, 2.0, 3.0]


def test_adapter_filter_is_peft_only() -> None:
    bridge = MegatronPeftBridge()

    assert bridge._is_adapter_param_name("decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight")
    assert not bridge._is_adapter_param_name("decoder.layers.0.self_attention.linear_qkv.weight_quantizer._amax")
    assert not bridge._is_adapter_param_name("decoder.layers.0.self_attention.linear_qkv.weight")


def test_auto_mapping_detects_quantized_layernorm_column_parallel_modules() -> None:
    QuantLayerNormColumnParallelLinear = type("QuantLayerNormColumnParallelLinear", (torch.nn.Module,), {})

    weight_mapping = AutoMapping(
        megatron_param="decoder.layers.0.self_attention.linear_qkv.weight",
        hf_param="model.layers.0.self_attn.q_proj.weight",
    )
    norm_mapping = AutoMapping(
        megatron_param="decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
        hf_param="model.layers.0.input_layernorm.weight",
    )

    module = QuantLayerNormColumnParallelLinear()

    assert weight_mapping._detect_parallelism_type(module) == "column"
    assert norm_mapping._detect_parallelism_type(module) == "replicated"


def test_auto_mapping_detects_dynamic_layernorm_column_parallel_modules(monkeypatch) -> None:
    class FakeDynamicModule(torch.nn.Module):
        def get_original_cls_by_level(self, *, level: int) -> type[torch.nn.Module]:
            assert level == 0
            return type("QuantLayerNormColumnParallelLinear", (torch.nn.Module,), {})

    monkeypatch.setattr(param_mapping_module, "is_modelopt_dynamic_module", lambda _module: True)

    weight_mapping = AutoMapping(
        megatron_param="decoder.layers.0.self_attention.linear_qkv.weight",
        hf_param="model.layers.0.self_attn.q_proj.weight",
    )
    norm_mapping = AutoMapping(
        megatron_param="decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
        hf_param="model.layers.0.input_layernorm.weight",
    )

    module = FakeDynamicModule()

    assert weight_mapping._detect_parallelism_type(module) == "column"
    assert norm_mapping._detect_parallelism_type(module) == "replicated"


def test_auto_mapping_registers_quantized_parallel_linear_types() -> None:
    assert "QuantColumnParallelLinear" in AutoMapping._MODULE_TYPE_REGISTRY["column"]
    assert "QuantRowParallelLinear" in AutoMapping._MODULE_TYPE_REGISTRY["row"]


def test_gpt_provider_exposes_modelopt_transformer_layer_spec() -> None:
    from megatron.bridge.models.gpt_provider import modelopt_transformer_layer_spec

    assert callable(modelopt_transformer_layer_spec)
