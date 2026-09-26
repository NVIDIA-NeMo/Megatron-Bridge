# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import QKVMapping
from megatron.bridge.models.inkling import InklingBridge, InklingModelProvider
from megatron.bridge.models.inkling.inkling_mapping import (
    InklingExpertGatedMapping,
    InklingGatedMapping,
    InklingSharedExpertMapping,
    interleave_gate_up,
)


pytestmark = pytest.mark.unit
FIXTURE = Path(__file__).with_name("published_config.json")


@pytest.fixture
def raw_config():
    return json.loads(FIXTURE.read_text())


def source(config):
    return SimpleNamespace(config=SimpleNamespace(to_dict=lambda: config), _model=None)


def test_raw_config_preserves_published_expert_width(tmp_path, raw_config):
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    # Native Transformers renames fields and replaces raw intermediate_size.
    normalized = AutoConfig.from_pretrained(tmp_path)
    pretrained = SimpleNamespace(config=normalized, model_name_or_path=tmp_path, init_kwargs={}, _model=None)
    provider = InklingBridge().provider_bridge(pretrained)
    provider.finalize()
    assert isinstance(provider, InklingModelProvider)
    assert provider.ffn_hidden_size == 16384
    assert provider.moe_ffn_hidden_size == 2048
    assert provider.num_layers == 42
    assert provider.moe_router_topk == 6
    assert provider.num_moe_experts == 256
    assert provider.moe_layer_freq == [0, 0] + [1] * 40
    assert provider.inkling_layer_types[5] == "hybrid"
    assert provider.inkling_layer_types[0] == "hybrid_sliding"
    assert provider.inkling_logits_mup_width_multiplier == 16
    assert provider.vocab_size == 201024
    assert provider.inkling_unpadded_vocab_size == 200058
    assert provider.inkling_n_shared_experts == 2
    assert provider.moe_shared_expert_intermediate_size is None
    assert provider.share_embeddings_and_output_weights is False
    assert provider.mtp_num_layers is None
    assert provider.position_embedding_type == "none"
    assert InklingBridge.megatron_to_hf_config(provider) == raw_config


def test_materialized_transformers_state_is_rejected(raw_config):
    pretrained = source(raw_config)
    pretrained._model = object()
    with pytest.raises(ValueError, match="lazy raw checkpoint"):
        InklingBridge().provider_bridge(pretrained)


def test_quantized_checkpoint_is_rejected(raw_config):
    raw_config["quantization_config"] = {"quant_method": "modelopt"}
    with pytest.raises(ValueError, match="unquantized checkpoint"):
        InklingBridge().provider_bridge(source(raw_config))


@pytest.mark.parametrize(
    "field,value", [("q_bias", True), ("shared_expert_sink", False), ("gate_activation", "softmax")]
)
def test_unsupported_architecture_fails_explicitly(raw_config, field, value):
    raw_config["text_config"][field] = value
    with pytest.raises(ValueError, match=field):
        InklingBridge().provider_bridge(source(raw_config))


def test_raw_projection_mappings_and_native_lora_names():
    bridge = InklingBridge()
    registry = bridge.mapping_registry()
    qkv = registry.megatron_to_hf_lookup("decoder.layers.7.self_attention.linear_qkv.weight")
    assert isinstance(qkv, QKVMapping)
    assert list(qkv.hf_param.values()) == [
        f"model.llm.layers.7.attn.{name}.weight" for name in ("wq_du", "wk_dv", "wv_dv")
    ]
    for module, raw in [("linear_r", "wr_du"), ("linear_proj", "wo_ud")]:
        mapping = registry.megatron_to_hf_lookup(f"decoder.layers.7.self_attention.{module}.weight")
        assert mapping.hf_param == f"model.llm.layers.7.attn.{raw}.weight"
        a_names, b_names = bridge._build_lora_hf_names([mapping.hf_param])
        assert a_names == [f"model.llm.layers.7.attn.{raw}.lora_A.weight"]
        assert b_names == [f"model.llm.layers.7.attn.{raw}.lora_B.weight"]


@pytest.mark.parametrize("layer,groups,head_dim", [(0, 4, 8), (1, 2, 16)])
def test_qkv_adapter_export_uses_layer_geometry(layer, groups, head_dim):
    config = SimpleNamespace(
        inkling_layer_types=("hybrid_sliding", "hybrid"),
        inkling_swa_num_attention_heads=8,
        inkling_swa_num_query_groups=4,
        inkling_swa_kv_channels=8,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=16,
    )
    # Both layers have the same packed width, but different Q/K/V interleaving.
    packed = torch.arange(128 * 4).reshape(128, 4)
    names = [f"model.llm.layers.{layer}.attn.{name}.weight" for name in ("wq_du", "wk_dv", "wv_dv")]
    exported = InklingBridge()._get_fused_adapter_linear_out_slices([SimpleNamespace(config=config)], names, packed)
    blocks = packed.reshape(groups, 4, head_dim, 4)
    expected = (blocks[:, :2].reshape(64, 4), blocks[:, 2].reshape(32, 4), blocks[:, 3].reshape(32, 4))
    assert all(torch.equal(exported[name], value) for name, value in zip(names, expected))
    assert config.num_attention_heads == 4
    adapter = SimpleNamespace(
        alpha=8,
        dim=4,
        linear_in_weight=SimpleNamespace(weight=torch.randn(4, 16)),
        linear_out_weight=SimpleNamespace(weight=packed.float()),
    )
    base = {name: torch.randn(value.shape[0], 16) for name, value in zip(names, expected)}
    merged = InklingBridge()._merge_lora_adapter_weights([SimpleNamespace(config=config)], base, [adapter])
    for name, value in zip(names, expected):
        torch.testing.assert_close(merged[name], base[name] + 2 * (value.float() @ adapter.linear_in_weight.weight))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dense_interleaved_roundtrip(dtype):
    raw = torch.arange(96, dtype=dtype).reshape(12, 8)
    mapping = InklingGatedMapping("mlp.linear_fc1.weight", "raw.w13.weight")
    native = mapping.hf_to_megatron(raw, None)
    assert torch.equal(native[:6], raw[0::2])
    assert torch.equal(native[6:], raw[1::2])
    assert torch.equal(mapping.megatron_to_hf(native, None)["raw.w13.weight"], raw)
    assert mapping.local_hf_param_specs() == ()


@pytest.mark.parametrize("expert", [0, 3])
def test_routed_interleaved_roundtrip(expert):
    raw = torch.arange(4 * 12 * 8, dtype=torch.float32).reshape(4, 12, 8)
    module = torch.nn.Module()
    module.register_parameter("weight0", torch.nn.Parameter(torch.empty(12, 8)))
    mapping = InklingExpertGatedMapping(
        f"decoder.layers.1.mlp.experts.linear_fc1.weight{expert}", "raw.experts.w13_weight"
    )
    native = mapping.hf_to_megatron(raw, module)
    assert torch.equal(native[:6], raw[expert, 0::2])
    assert torch.equal(native[6:], raw[expert, 1::2])
    restored = mapping.megatron_to_hf(native, module)["raw.experts.w13_weight"]
    assert torch.equal(restored, raw[expert])
    assert mapping.local_hf_param_specs() == ()


@pytest.mark.parametrize("fc,shape", [(1, (2, 12, 8)), (2, (2, 8, 6))])
def test_shared_experts_stack_roundtrip(fc, shape):
    raw = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(shape)
    bridge = InklingBridge()
    registry = bridge.mapping_registry()
    buffers, sources = {}, {}
    output = None
    for expert in (1, 0):  # Explicitly verify export does not assume traversal order.
        name = f"decoder.layers.2.mlp.shared_experts.{expert}.linear_fc{fc}.weight"
        mapping = registry.megatron_to_hf_lookup(name)
        assert isinstance(mapping, InklingSharedExpertMapping)
        module = torch.nn.Linear(shape[-1], shape[-2], bias=False)
        native = mapping.hf_to_megatron(raw, module)
        converted = mapping.megatron_to_hf(native, module)
        task = SimpleNamespace(mapping=mapping, param_name=name)
        output = bridge._accumulate_grouped_export(
            task, converted, SimpleNamespace(inkling_n_shared_experts=2), buffers, {}, sources
        )
    assert output is not None
    assert torch.equal(next(iter(output.values())), raw)
    assert not buffers
    assert len(next(iter(sources.values()))) == 2


def test_interleave_preserves_ep_leading_dimension():
    gate = torch.arange(3 * 4 * 8).reshape(3, 4, 8)
    up = gate + 1000
    raw = interleave_gate_up(gate, up)
    assert raw.shape == (3, 8, 8)
    assert torch.equal(raw[:, 0::2], gate)
    assert torch.equal(raw[:, 1::2], up)


def test_auxiliary_tensors_are_preserved_without_becoming_training_parameters():
    tensors = {
        "model.audio.encoder.weight": torch.randn(2, 3),
        "model.visual.final_norm.weight": torch.randn(3),
        "model.mtp.layers.0.embed_norm.weight": torch.randn(3),
    }
    state = dict(tensors)

    class State(dict):
        source = SimpleNamespace(get_all_keys=lambda: list(tensors))

    with patch.object(MegatronModelBridge, "stream_weights_megatron_to_hf", return_value=iter(())):
        result = dict(
            InklingBridge().stream_weights_megatron_to_hf(None, SimpleNamespace(state=State(state)), cpu=True)
        )
    assert result.keys() == tensors.keys()
    assert all(torch.equal(result[key], value) for key, value in tensors.items())
