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

"""CPU conversion tests; transport mocks do not exercise distributed collectives."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import (
    MegatronWeightTuple,
    ModelConfigNotSupportedError,
    WeightConversionTask,
)
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.conversion.peft_bridge import AdapterWeight
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
from megatron.bridge.models.mimo_v2_flash.mimo_v2_bridge import MiMoV2Bridge, MiMoV2FusedQKVMapping
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge import MiMoV2FlashBridge
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_provider import MiMoV2FlashModelProvider


pytestmark = pytest.mark.unit


class _MiMoV2Config(PretrainedConfig):
    model_type = "mimo_v2"


@pytest.fixture
def hf_config():
    # MiMo-V2.6-Flash-RL revision 5711b268169967567844e1e560e8a3966da959b1:
    # retain attention/routing semantics while reducing layers, width and experts.
    return _MiMoV2Config(
        architectures=["MiMoV2ForCausalLM"],
        model_type="mimo_v2",
        name_or_path="XiaomiMiMo/MiMo-V2.6-Flash-RL",
        auto_map={
            "AutoConfig": "configuration_mimo_v2.MiMoV2Config",
            "AutoModelForCausalLM": "modeling_mimo_v2.MiMoV2ForCausalLM",
        },
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=192,
        v_head_dim=128,
        vocab_size=128,
        max_position_embeddings=1048576,
        rope_theta=10000000,
        swa_rope_theta=10000,
        partial_rotary_factor=0.334,
        rms_norm_eps=1e-5,
        layernorm_epsilon=1e-5,
        hidden_act="silu",
        tie_word_embeddings=False,
        attention_bias=False,
        attention_projection_layout="fused_qkv",
        attention_value_scale=0.707,
        hybrid_layer_pattern=[0, 1],
        sliding_window_size=128,
        swa_num_key_value_heads=8,
        swa_num_attention_heads=8,
        swa_head_dim=192,
        swa_v_head_dim=128,
        add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False,
        moe_layer_freq=[0, 1],
        n_routed_experts=8,
        moe_intermediate_size=32,
        num_experts_per_tok=2,
        n_shared_experts=None,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=None,
        moe_router_dtype="bfloat16",
        quantization_config={"quant_method": "fp8", "store_dtype": "mxfp4", "weight_block_size": [128, 128]},
        vision_config={"hidden_size": 16},
        audio_config={"hidden_size": 8},
    )


def _projection_rows(groups, width=2):
    # Separate ranges expose Q/K/V swaps, and unique rows expose head permutations.
    sizes = (8 * 192, groups * 192, groups * 128)
    return tuple(
        torch.arange(size * width).reshape(size, width) + offset for size, offset in zip(sizes, (0, 10000, 20000))
    )


def _pack_rows(projections, chunks):
    return torch.cat([part for shard in zip(*(p.chunk(chunks) for p in projections)) for part in shard])


def _conversion_task(name, weight_dtype=None):
    return WeightConversionTask("weight", "weight", AutoMapping("weight", name), weight_dtype=weight_dtype)


@pytest.fixture
def megatron_model(hf_config):
    provider = MiMoV2Bridge().provider_bridge(SimpleNamespace(config=hf_config))
    return [SimpleNamespace(config=provider)]


@pytest.mark.parametrize("groups", [4, 8])
@pytest.mark.parametrize("checkpoint_tp_size", [1, 4])
def test_fused_qkv_import_and_export_exact_rows(groups, checkpoint_tp_size):
    projections = _projection_rows(groups)
    raw = _pack_rows(projections, checkpoint_tp_size)
    expected = _pack_rows(projections, groups)
    mapping = MiMoV2FusedQKVMapping("decoder.layers.0.self_attention.linear_qkv.weight", "qkv", checkpoint_tp_size)
    config = SimpleNamespace(num_attention_heads=8, num_query_groups=groups, kv_channels=192, v_head_dim=128)

    with (
        patch.object(type(mapping), "tp_rank", property(lambda self: 0)),
        patch.object(mapping, "_get_config", return_value=config),
        patch.object(AutoMapping, "hf_to_megatron", side_effect=lambda weight, module: weight),
        patch.object(AutoMapping, "megatron_to_hf", side_effect=lambda weight, module: {"qkv": weight}),
        patch.object(mapping, "broadcast_obj_from_pp_rank", side_effect=lambda value, name: value),
    ):
        converted = mapping.hf_to_megatron(raw, Mock())
        torch.testing.assert_close(converted, expected, rtol=0, atol=0)
        torch.testing.assert_close(mapping.megatron_to_hf(converted, Mock())["qkv"], raw, rtol=0, atol=0)


def test_checkpoint_partition_metadata_survives_wildcard_resolution(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"tp_size": 4}, "weight_map": {}}))
    source = Mock(spec=SafeTensorsStateSource)
    source.path = tmp_path
    bridge = MiMoV2Bridge()
    bridge.hf_pretrained = SimpleNamespace(state=SimpleNamespace(source=source))
    mapping = next(m for m in bridge.mapping_registry().mappings if isinstance(m, MiMoV2FusedQKVMapping))
    resolved = mapping.resolve(("3",))
    assert resolved.checkpoint_tp_size == 4
    assert resolved.megatron_param == "decoder.layers.3.self_attention.linear_qkv.weight"
    assert resolved.hf_param == "model.layers.3.self_attn.qkv_proj.weight"
    assert not any(m.megatron_param.startswith("mtp.") for m in bridge.mapping_registry().mappings)


def test_config_only_provider_preserves_v2_metadata_and_reference_router(hf_config):
    with patch("megatron.bridge.models.hf_pretrained.base.SafeTensorsStateSource") as state_source:
        auto_bridge = AutoBridge.from_hf_config(hf_config)
        provider = auto_bridge.to_megatron_provider(load_weights=False)
        with pytest.raises(ModelConfigNotSupportedError):
            auto_bridge.get_model_config()
    state_source.assert_not_called()
    assert isinstance(provider, MiMoV2FlashModelProvider)
    assert provider.mtp_num_layers == 0
    assert provider.moe_router_dtype == "fp32"
    assert provider.moe_router_bias_update_rate == 0.0
    assert provider.rotary_base == (10000, 10000000)
    assert provider.full_attn_num_query_groups == 4
    assert provider.swa_num_query_groups == 8
    exported = MiMoV2Bridge.megatron_to_hf_config(provider)
    assert exported == hf_config.to_dict()
    assert exported["model_type"] == "mimo_v2"
    exported["auto_map"]["AutoConfig"] = "changed"
    assert MiMoV2Bridge.megatron_to_hf_config(provider) == hf_config.to_dict()


@pytest.mark.parametrize("layer,groups", [(0, 4), (1, 8)])
def test_nonzero_lora_b_exports_in_canonical_hf_order(megatron_model, layer, groups):
    bridge = MiMoV2Bridge()
    assert bridge.hf_config is None  # Adapter-only dispatch has no source HF model.
    projections = tuple(p.float() / 32768 for p in _projection_rows(groups, width=3))
    linear_out = _pack_rows(projections, groups)
    name = f"model.layers.{layer}.self_attn.qkv_proj.weight"
    with patch.object(bridge, "_source_tp_size", return_value=4):
        exported = bridge._get_fused_adapter_linear_out_slices(
            megatron_model=megatron_model, base_hf_weight_names=[name], linear_out_tensor=linear_out, is_expert=False
        )
    assert set(exported) == {name}
    torch.testing.assert_close(exported[name], torch.cat(projections), rtol=0, atol=0)


def test_nonzero_lora_merge_restores_checkpoint_partition_order(megatron_model):
    bridge = MiMoV2Bridge()
    name = "model.layers.1.self_attn.qkv_proj.weight"
    projections = tuple(p.float() / 32768 for p in _projection_rows(8, width=3))
    linear_out = _pack_rows(projections, 8)
    linear_in = torch.tensor([[1, -1], [2, 1], [-1, 3]], dtype=torch.float32)
    base = torch.full((linear_out.shape[0], 2), 0.25)
    adapter = AdapterWeight(
        global_base_prefix="decoder.layers.1.self_attention.linear_qkv",
        adapter_key=None,
        alpha=6,
        dim=3,
        linear_in_weight=MegatronWeightTuple("in", linear_in, vp_stage=0),
        linear_out_weight=MegatronWeightTuple("out", linear_out, vp_stage=0),
    )
    with patch.object(bridge, "_source_tp_size", return_value=4):
        exported = bridge._merge_lora_adapter_weights(megatron_model, {name: base}, [adapter])
    expected = base + 2 * (_pack_rows(projections, 4) @ linear_in)
    torch.testing.assert_close(exported[name], expected, rtol=0, atol=0)


def test_mxfp4_import_and_export_preserve_signed_nibbles_and_e8m0_scales():
    bridge = MiMoV2Bridge()
    name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    packed = torch.tensor([0xF7, 0xE6, 0xD5, 0xC4, 0xB3, 0xA2, 0x91, 0x00], dtype=torch.uint8).repeat(2, 4)
    scales = torch.tensor([[127, 128], [126, 129]], dtype=torch.uint8)
    state = {name: packed, f"{name}_scale": scales}
    # Independent E2M1 oracle: low nibble first; each E8M0 value scales 32 columns.
    values = torch.tensor([6, -6, 4, -4, 3, -3, 2, -2, 1.5, -1.5, 1, -1, 0.5, -0.5, 0, 0]).repeat(2, 4)
    expected = (values * torch.tensor([[1, 2], [0.5, 4]]).repeat_interleave(32, dim=1)).bfloat16()
    decoded = bridge.maybe_modify_loaded_hf_weight(name, state)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    exported = bridge.maybe_modify_converted_hf_weight(_conversion_task(name), {name: decoded}, state)
    assert exported[name].dtype == torch.uint8
    assert exported[f"{name}_scale"].dtype == torch.uint8
    torch.testing.assert_close(exported[name], packed, rtol=0, atol=0)
    torch.testing.assert_close(exported[f"{name}_scale"], scales, rtol=0, atol=0)
    assert set(exported) == set(state)


def test_fp8_checkpoint_shards_have_independently_padded_scale_blocks():
    bridge = MiMoV2Bridge()
    name = "model.layers.0.self_attn.qkv_proj.weight"
    # Published global QKV: (64*192 + 4*192 + 4*128)/4 = 3392 rows per shard.
    # Four independently padded chunks require 108 scale rows, not ceil(13568/128)=106.
    rows_per_chunk, chunks, cols = 3392, 4, 16
    weight = torch.ones(rows_per_chunk * chunks, cols).to(torch.float8_e4m3fn)
    scales = torch.arange(1, 109, dtype=torch.float32).reshape(108, 1)
    state = {name: weight, f"{name}_scale_inv": scales}
    expected = torch.cat([scale.repeat_interleave(128, dim=0)[:rows_per_chunk] for scale in scales.chunk(chunks)])
    expected = expected.expand(-1, cols).bfloat16()
    with patch.object(bridge, "_source_tp_size", return_value=chunks):
        decoded = bridge.maybe_modify_loaded_hf_weight(name, state)
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
        exported = bridge.maybe_modify_converted_hf_weight(_conversion_task(name), {name: decoded}, state)
        assert exported[name].dtype == torch.float8_e4m3fn
        assert exported[f"{name}_scale_inv"].shape == scales.shape
        torch.testing.assert_close(bridge.maybe_modify_loaded_hf_weight(name, exported), expected, rtol=0, atol=0)
        with pytest.raises(ValueError, match="expected FP8 scales"):
            bridge.maybe_modify_loaded_hf_weight(name, {name: weight, f"{name}_scale_inv": scales[:106]})


def test_import_dependencies_include_both_quantization_sidecars_and_plain_weights():
    bridge = MiMoV2Bridge()
    names = {"q": "model.layers.0.self_attn.qkv_proj.weight", "gate": "model.layers.1.mlp.experts.0.gate_proj.weight"}
    available = {names["q"], f"{names['q']}_scale_inv", names["gate"], f"{names['gate']}_scale"}
    assert set(bridge.get_hf_import_param_names(names, available)) == available
    weight = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    assert bridge.maybe_modify_loaded_hf_weight("weight", {"weight": weight}) is weight


def test_text_export_preserves_untrained_auxiliary_weights(hf_config):
    weights = {
        "visual.patch_embed.weight": torch.ones(2, 2),
        "audio_encoder.conv.weight": torch.full((2, 2), 2),
        "speech_embeddings.weight": torch.full((2, 2), 3),
        "model.mtp.layers.0.self_attn.qkv_proj.weight": torch.full((2, 2), 4),
        "model.layers.0.self_attn.qkv_proj.weight": torch.zeros(2, 2),
    }
    state = Mock()
    state.source.get_all_keys.return_value = list(weights)
    state.__getitem__ = Mock(side_effect=weights.__getitem__)
    pretrained = SimpleNamespace(config=hf_config, state=state)
    bridge = MiMoV2Bridge()
    assert bridge.provider_bridge(pretrained).mtp_num_layers == 0
    with patch.object(MiMoV2FlashBridge, "stream_weights_megatron_to_hf", return_value=iter(())):
        exported = dict(bridge.stream_weights_megatron_to_hf([], pretrained))
    assert set(exported) == set(weights) - {"model.layers.0.self_attn.qkv_proj.weight"}
    for name, weight in exported.items():
        torch.testing.assert_close(weight, weights[name], rtol=0, atol=0)


@pytest.mark.parametrize("prebuilt_task", [False, True])
def test_quantized_source_export_rejects_dtype_override(hf_config, prebuilt_task):
    bridge = MiMoV2Bridge()
    with pytest.raises(ValueError, match="preserves quantization"):
        if prebuilt_task:
            bridge.maybe_modify_converted_hf_weight(
                _conversion_task("lm_head.weight", weight_dtype=torch.float32),
                {"lm_head.weight": torch.ones(2, 32)},
                {"lm_head.weight_scale": torch.ones(2, 1, dtype=torch.uint8)},
            )
        else:
            list(
                bridge.stream_weights_megatron_to_hf([], SimpleNamespace(config=hf_config), weight_dtype=torch.float32)
            )
