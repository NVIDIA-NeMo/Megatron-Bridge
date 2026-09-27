# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.deepseek.deepseek_v41_bridge import (
    DeepSeekV41Bridge,
    _EngramEmbeddingMapping,
    _ReplicatedBufferMapping,
)
from megatron.bridge.models.deepseek.deepseek_v41_provider import DeepSeekV41ModelProvider
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict


def _bridge() -> DeepSeekV41Bridge:
    bridge = DeepSeekV41Bridge()
    bridge.hf_config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=2,
            engram_layer_ids=[1],
            num_nextn_predict_layers=1,
        ),
        vision_config=SimpleNamespace(num_hidden_layers=1),
    )
    return bridge


def test_provider_freezes_v41_checkpoint_router_biases():
    assert DeepSeekV41ModelProvider().moe_router_bias_update_rate == 0.0


def test_provider_bridge_disables_dspark_for_actor(monkeypatch):
    seen = {}

    class NativeConfig:
        dspark_config = object()

        @classmethod
        def from_hf(cls, hf_dict):
            seen.update(hf_dict)
            return cls()

    hf_pretrained = SimpleNamespace(
        config=SimpleNamespace(
            to_dict=lambda: {
                "model_type": "deepseek_v41",
                "text_config": {"vocab_size": 128, "max_position_embeddings": 64},
                "vision_config": None,
            }
        ),
        model_name_or_path="test",
        init_kwargs={},
    )
    monkeypatch.setattr(
        "megatron.bridge.models.deepseek.deepseek_v41_bridge._load_deepseek_v41_config_class",
        lambda: NativeConfig,
    )

    provider = DeepSeekV41Bridge().provider_bridge(hf_pretrained)

    assert provider.dspark_config is None
    assert seen["vision_config"] is None


def test_provider_bridge_restores_flat_vllm_vision_config(monkeypatch):
    seen = {}

    class NativeConfig:
        dspark_config = object()

        @classmethod
        def from_hf(cls, hf_dict):
            seen.update(hf_dict)
            return cls()

    hf_pretrained = SimpleNamespace(
        config=SimpleNamespace(
            to_dict=lambda: {
                "model_type": "deepseek_v41",
                "vocab_size": 128,
                "max_position_embeddings": 64,
                "num_hidden_layers": 2,
                "vision_n_layers": 1,
                "vision_dim": 1024,
                "vision_n_heads": 16,
                "vision_inter_dim": 2816,
                "vision_patch_size": 14,
                "vision_rope_theta": 10000.0,
                "vision_downsample_ratio": 3,
                "vision_max_n_token": 1024,
                "vision_min_pixels": 295936,
                "vision_max_wh_ratio": 200,
            }
        ),
        model_name_or_path="test",
        init_kwargs={},
    )
    monkeypatch.setattr(
        "megatron.bridge.models.deepseek.deepseek_v41_bridge._load_deepseek_v41_config_class",
        lambda: NativeConfig,
    )

    DeepSeekV41Bridge().provider_bridge(hf_pretrained)

    assert seen["vision_config"] == {
        "num_hidden_layers": 1,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "intermediate_size": 2816,
        "patch_size": 14,
        "rope_theta": 10000.0,
        "downsample_ratio": 3,
        "max_image_tokens": 1024,
        "min_pixels": 295936,
        "max_wh_ratio": 200,
    }


def test_mapping_registry_covers_physical_backbone_and_extra_components():
    registry = _bridge().mapping_registry()

    expected = {
        "decoder.layers.0.inner_layer.self_attention.linear_kv_proj.weight": "layers.0.attn.wkv.weight",
        "decoder.layers.1.inner_layer.mlp.router.text_balance.expert_bias": "layers.0.ffn.gate.bias",
        "decoder.layers.1.inner_layer.mlp.router.image_balance.expert_bias": "layers.0.ffn.gate.bias_vl",
        "decoder.layers.2.engram.embed.tables.0.weight": "layers.1.engram.embed.weight",
        "vision.blocks.0.attn.wqkv.weight": "vision.blocks.0.attn.wqkv.weight",
        "dspark.decoder.layers.0.inner_layer.self_attention.linear_q_down_proj.weight": "mtp.0.attn.wq_a.weight",
        "dspark.decoder.layers.1.inner_layer.mlp.router.expert_bias": "mtp.0.ffn.gate.bias",
        "dspark.confidence_head.weight": "mtp.0.confidence_head.proj.weight",
    }
    for megatron_name, hf_name in expected.items():
        mapping = registry.megatron_to_hf_lookup(megatron_name)
        assert mapping is not None, megatron_name
        assert mapping.hf_param == hf_name

    # DSpark drafts text tokens only; the released visual router bias is inert.
    assert registry.hf_to_megatron_lookup("mtp.0.ffn.gate.bias_vl") is None


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="PyTorch lacks E8M0 scales")
def test_fp8_import_uses_released_32_element_blocks():
    bridge = _bridge()
    weight = torch.ones((32, 64), dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[2.0, 4.0]], dtype=torch.float8_e8m0fnu)

    result = bridge.maybe_modify_loaded_hf_weight(
        "layers.0.attn.wkv.weight",
        {"layers.0.attn.wkv.weight": weight, "layers.0.attn.wkv.scale": scale},
    )

    assert result.dtype == torch.bfloat16
    assert torch.all(result[:, :32] == 2)
    assert torch.all(result[:, 32:] == 4)


def test_packed_fp4_import_restores_the_unpacked_expert_width():
    bridge = _bridge()
    packed = torch.zeros((2, 16), dtype=torch.int8)
    scale = torch.ones((2, 1), dtype=torch.float32)

    result = bridge.maybe_modify_loaded_hf_weight(
        "layers.0.ffn.experts.0.w1.weight",
        {
            "layers.0.ffn.experts.0.w1.weight": packed,
            "layers.0.ffn.experts.0.w1.scale": scale,
        },
    )

    assert result.shape == (2, 32)
    assert result.dtype == torch.bfloat16
    assert torch.count_nonzero(result) == 0


def test_quantized_import_fails_when_scale_is_missing():
    bridge = _bridge()

    with pytest.raises(ValueError, match="missing"):
        bridge.maybe_modify_loaded_hf_weight(
            "layers.0.ffn.experts.0.w1.weight",
            {"layers.0.ffn.experts.0.w1.weight": torch.zeros((1, 16), dtype=torch.int8)},
        )


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="PyTorch lacks E8M0 scales")
def test_engram_import_reads_only_the_owner_rows_from_safetensors(tmp_path):
    weight_name = "layers.1.engram.embed.weight"
    scale_name = "layers.1.engram.embed.scale"
    weight = torch.ones((5, 64), dtype=torch.float8_e4m3fn)
    scale = torch.tensor(
        [[1.0, 2.0], [2.0, 4.0], [4.0, 8.0], [8.0, 16.0], [16.0, 32.0]],
        dtype=torch.float8_e8m0fnu,
    )
    save_file({weight_name: weight, scale_name: scale}, tmp_path / "model.safetensors")
    source = SafeTensorsStateSource(tmp_path)
    _ = source.key_to_filename_map
    state = StateDict(source)
    bridge = _bridge()
    bridge._engram_import_ranges = {weight_name: (1, 3, True)}

    result = bridge.maybe_modify_loaded_hf_weight(weight_name, state)

    assert result.shape == (2, 64)
    assert torch.all(result[0, :32] == 2)
    assert torch.all(result[0, 32:] == 4)
    assert torch.all(result[1, :32] == 4)
    assert torch.all(result[1, 32:] == 8)


def test_engram_row_reader_reuses_one_safetensor_handle(tmp_path, monkeypatch):
    import safetensors

    weight_name = "layers.1.engram.embed.weight"
    scale_name = "layers.1.engram.embed.scale"
    save_file(
        {
            weight_name: torch.ones((5, 64), dtype=torch.float32),
            scale_name: torch.ones((5, 2), dtype=torch.float32),
        },
        tmp_path / "model.safetensors",
    )
    source = SafeTensorsStateSource(tmp_path)
    _ = source.key_to_filename_map
    state = StateDict(source)
    original_safe_open = safetensors.safe_open
    open_count = 0

    def counted_safe_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return original_safe_open(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", counted_safe_open)
    bridge = _bridge()
    with bridge._state_row_reader(state, [weight_name, scale_name]) as read_rows:
        assert read_rows(weight_name, 0, 2).shape == (2, 64)
        assert read_rows(weight_name, 2, 4).shape == (2, 64)
        assert read_rows(scale_name, 0, 4).shape == (4, 2)

    assert open_count == 1


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="PyTorch lacks E8M0 scales")
def test_engram_import_streams_bounded_row_chunks_into_the_target(monkeypatch):
    weight_name = "layers.1.engram.embed.weight"
    scale_name = "layers.1.engram.embed.scale"
    weight = torch.ones((7, 64), dtype=torch.float8_e4m3fn)
    scale = torch.tensor(
        [[1.0, 2.0], [2.0, 4.0], [4.0, 8.0], [8.0, 16.0], [16.0, 32.0], [32.0, 64.0], [64.0, 128.0]],
        dtype=torch.float8_e8m0fnu,
    )
    state = {weight_name: weight, scale_name: scale}

    class EngramTable(nn.Module):
        def __init__(self):
            super().__init__()
            self.global_num_embeddings = 7
            self.row_start = 1
            self.row_end = 6
            self.weight = nn.Parameter(torch.empty((5, 64), dtype=torch.bfloat16))

    module = EngramTable()
    mapping = _EngramEmbeddingMapping("engram.embed.tables.0.weight", weight_name)
    task = WeightConversionTask(
        param_name="engram.embed.tables.0.weight",
        global_param_name="decoder.layers.2.engram.embed.tables.0.weight",
        mapping=mapping,
        megatron_module=module,
        param_weight=module.weight,
    )
    bridge = _bridge()
    row_reads = []
    load_rows = bridge._load_state_rows

    def record_rows(hf_state_dict, name, row_start, row_end):
        row_reads.append((name, row_start, row_end))
        return load_rows(hf_state_dict, name, row_start, row_end)

    monkeypatch.setattr(bridge, "_load_state_rows", record_rows)
    bridge._load_engram_tasks_streaming(state, [task], chunk_bytes=2 * 64 * 2)

    assert row_reads == [
        (weight_name, 1, 3),
        (scale_name, 1, 3),
        (weight_name, 3, 5),
        (scale_name, 3, 5),
        (weight_name, 5, 6),
        (scale_name, 5, 6),
    ]
    assert torch.all(module.weight[0, :32] == 2)
    assert torch.all(module.weight[0, 32:] == 4)
    assert torch.all(module.weight[-1, :32] == 32)
    assert torch.all(module.weight[-1, 32:] == 64)


def test_engram_export_is_streamed_outside_the_base_bridge(monkeypatch):
    regular_task = WeightConversionTask(
        param_name="decoder.layers.0.weight",
        global_param_name="decoder.layers.0.weight",
        mapping=SimpleNamespace(hf_param="layers.0.weight"),
    )
    engram_module = SimpleNamespace(global_num_embeddings=4, row_start=0, row_end=4)
    engram_task = WeightConversionTask(
        param_name="decoder.layers.0.engram.embed.tables.0.weight",
        global_param_name="decoder.layers.0.engram.embed.tables.0.weight",
        mapping=_EngramEmbeddingMapping(
            "decoder.layers.0.engram.embed.tables.0.weight", "layers.0.engram.embed.weight"
        ),
        megatron_module=engram_module,
        param_weight=torch.zeros((4, 2), dtype=torch.bfloat16),
    )
    seen = {}

    def fake_base_stream(self, *args, **kwargs):
        seen["tasks"] = kwargs["conversion_tasks"]
        yield ("layers.0.weight", torch.ones(1))

    monkeypatch.setattr(MegatronModelBridge, "stream_weights_megatron_to_hf", fake_base_stream)

    output = list(
        _bridge().stream_weights_megatron_to_hf(
            object(),
            object(),
            conversion_tasks=[regular_task, engram_task],
        )
    )

    assert seen["tasks"] == [regular_task]
    assert [name for name, _ in output] == ["layers.0.weight", "layers.0.engram.embed.weight.__rows_0"]


def test_replicated_buffer_mapping_targets_parameter_free_module():
    module = nn.Module()
    module.register_buffer("expert_bias", torch.zeros(3, dtype=torch.float32))
    mapping = _ReplicatedBufferMapping("router.expert_bias", "gate.bias")

    result = mapping.hf_to_megatron(torch.tensor([1.0, 2.0, 3.0]), module)

    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))


def test_quantized_export_restores_packed_fp4_sidecar():
    bridge = _bridge()
    hf_name = "layers.0.ffn.experts.0.w1.weight"
    scale_name = "layers.0.ffn.experts.0.w1.scale"
    weight = torch.zeros((2, 32), dtype=torch.bfloat16)
    source_scale = torch.ones((2, 1), dtype=torch.float32)
    task = SimpleNamespace(weight_dtype=None)

    result = bridge.maybe_modify_converted_hf_weight(
        task,
        {hf_name: weight},
        {scale_name: source_scale},
    )

    assert result[hf_name].dtype == torch.int8
    assert result[hf_name].shape == (2, 16)
    assert result[scale_name].shape == source_scale.shape
    restored = quantization_utils.dequantize_mxfp4_e2m1_packed(result[hf_name], result[scale_name])
    assert torch.count_nonzero(restored) == 0
