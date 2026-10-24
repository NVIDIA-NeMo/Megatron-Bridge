import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping, QKVMapping
from megatron.bridge.models.minicpmo import MiniCPMOBridge, MiniCPMOModelConfig


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_dense_config_round_trip(trust_remote_code):
    config = PretrainedConfig(
        model_type="minicpmo",
        architectures=["MiniCPMO"],
        version="4.5",
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151748,
        max_position_embeddings=40960,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
        init_vision=True,
        init_audio=True,
        vision_config={"hidden_size": 1152},
        audio_config={"d_model": 1024},
        init_tts=True,
    )

    # PretrainedConfig serializes its class-level model_type; use a local
    # subclass to exercise the same identity as the checkpoint configuration.
    class MiniCPMOConfig(PretrainedConfig):
        model_type = "minicpmo"

    config = MiniCPMOConfig.from_dict(config.to_dict())
    original = deepcopy(config.to_dict())
    assert AutoBridge.supports(config)
    bridge = AutoBridge.from_hf_config(config)
    bridge.trust_remote_code = trust_remote_code
    converted = bridge.get_model_config()
    assert converted.trust_remote_code is trust_remote_code
    assert isinstance(converted, MiniCPMOModelConfig)
    assert converted.transformer.num_layers == 36
    assert converted.transformer.num_query_groups == 8
    assert converted.transformer.qk_layernorm is True
    exported = MiniCPMOBridge.megatron_to_hf_config(converted)
    assert exported["architectures"] == ["MiniCPMO"]
    assert exported["vision_config"] == original["vision_config"]
    assert exported["audio_config"] == original["audio_config"]
    assert exported["init_tts"] is False
    assert config.to_dict() == original


def test_mappings_keep_model_namespaces_and_exclude_talker():
    registry = MiniCPMOBridge().mapping_registry()
    names = [m.megatron_param for m in registry.mappings]
    assert "language_model.decoder.layers.*.self_attention.linear_qkv.weight" in names
    assert "language_model.decoder.layers.*.mlp.linear_fc1.weight" in names
    assert "perception.vpm.**" in names
    assert "perception.apm.**" in names
    assert not any("tts" in n or "mtp" in n for n in names)


@pytest.mark.unit
def test_inherited_qkv_mapping_uses_grouped_query_layout_in_each_direction():
    mapping = (
        MiniCPMOBridge()
        .mapping_registry()
        .megatron_to_hf_lookup("language_model.decoder.layers.0.self_attention.linear_qkv.weight")
    )
    assert type(mapping) is QKVMapping
    assert mapping.hf_param == {name: f"llm.model.layers.0.self_attn.{name}_proj.weight" for name in ("q", "k", "v")}
    module = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=4, num_query_groups=2, kv_channels=2, hidden_size=4)
    )
    q = torch.arange(32, dtype=torch.float32).reshape(8, 4) + 10
    k = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 100
    v = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 200
    # Independent MCore layout: each query group is Q heads, K head, V head.
    expected = torch.cat([q[:4], k[:2], v[:2], q[4:], k[2:], v[2:]])
    with patch.object(mapping._tp_mapping, "hf_to_megatron", side_effect=lambda weight, _module: weight):
        actual = mapping.hf_to_megatron({"q": q, "k": k, "v": v}, module)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Export a hand-built layout, not the importer output: a symmetric swap
    # in both directions must not pass this test.
    with (
        patch.object(mapping, "broadcast_obj_from_pp_rank", side_effect=lambda value, _key: value),
        patch.object(mapping._tp_mapping, "megatron_to_hf", return_value={mapping.megatron_param: expected}),
    ):
        restored = mapping.megatron_to_hf(expected, module)
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        torch.testing.assert_close(restored[mapping.hf_param[name]], tensor, rtol=0, atol=0)


@pytest.mark.unit
def test_inherited_dense_mlp_mapping_keeps_gate_before_up_in_each_direction():
    mapping = (
        MiniCPMOBridge()
        .mapping_registry()
        .megatron_to_hf_lookup("language_model.decoder.layers.0.mlp.linear_fc1.weight")
    )
    assert type(mapping) is GatedMLPMapping
    assert mapping.hf_param == {name: f"llm.model.layers.0.mlp.{name}_proj.weight" for name in ("gate", "up")}
    gate = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 10
    up = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100
    expected = torch.cat([gate, up])
    module = SimpleNamespace()
    actual = mapping.hf_to_megatron({"gate": gate, "up": up}, module)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with patch.object(mapping, "broadcast_from_pp_rank", side_effect=lambda value, cache_key=None: value):
        restored = mapping.megatron_to_hf(expected, module)
    for name, tensor in (("gate", gate), ("up", up)):
        torch.testing.assert_close(restored[mapping.hf_param[name]], tensor, rtol=0, atol=0)


@pytest.mark.unit
def test_export_preserves_remote_tokenizer_schema_without_reverting_model_config(tmp_path):
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

    source = tmp_path / "source"
    source.mkdir()
    original = {
        "tokenizer_config.json": {"extra_special_tokens": {}, "tokenizer_class": "MiniCPMOTokenizerFast"},
        "tokenizer.json": {"added_tokens": [{"id": 7, "content": "<image>"}]},
        "preprocessor_config.json": {"processor_class": "MiniCPMOProcessor", "version": "4.5"},
    }
    for name, data in original.items():
        (source / name).write_text(json.dumps(data))
    # A source config must never overwrite the understanding-only export config.
    (source / "config.json").write_text(json.dumps({"init_tts": True}))

    class NewerArtifactSerializer:
        def save_pretrained(self, directory):
            # Reproduce the observed cross-version metadata change independently
            # in required and optional artifact serialization.
            for name in original:
                (directory / name).write_text(json.dumps({"extra_special_tokens": ["<image>"]}))
            (directory / "processor_config.json").write_text(
                json.dumps(
                    {
                        "audio_processor": {"feature_size": 80},
                        "image_processor": {"patch_size": 14},
                        "tokenizer": {"extra_special_tokens": ["<image>"]},
                        "audio_pool_step": 5,
                        "processor_class": "MiniCPMOProcessor",
                    }
                )
            )

    wrapper = PreTrainedCausalLM(str(source), device="cpu")
    wrapper.config = PretrainedConfig(
        init_tts=False, rope_parameters={"rope_type": "default", "rope_theta": 1000000}, dtype="bfloat16"
    )
    wrapper.tokenizer = NewerArtifactSerializer()
    wrapper.processor = NewerArtifactSerializer()
    wrapper.generation_config = None
    wrapper.image_processor = None
    output = tmp_path / "exported"
    wrapper.save_artifacts(
        output,
        original_source_path=source,
        additional_files=MiniCPMOBridge.ADDITIONAL_FILE_PATTERNS,
    )
    bridge = MiniCPMOBridge()
    bridge.postprocess_hf_export_artifacts(output)
    for name in original:
        assert (output / name).read_bytes() == (source / name).read_bytes()
    assert isinstance(json.loads((output / "tokenizer_config.json").read_text())["extra_special_tokens"], dict)
    config = json.loads((output / "config.json").read_text())
    assert config["init_tts"] is False
    assert config["rope_theta"] == config["rope_parameters"]["rope_theta"] == 1000000
    assert config["torch_dtype"] == config["dtype"] == "bfloat16"
    expected = {"audio_pool_step": 5, "processor_class": "MiniCPMOProcessor"}
    assert json.loads((output / "processor_config.json").read_text()) == expected
    bridge.postprocess_hf_export_artifacts(output)
    assert json.loads((output / "processor_config.json").read_text()) == expected
