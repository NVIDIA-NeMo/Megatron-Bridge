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

from megatron.bridge.models.diffusion_gemma.diffusion_gemma_bridge import (
    DiffusionGemmaBridge,
    _normalized_diffusion_config,
    _TiedReplicatedMapping,
)
from megatron.bridge.models.diffusion_gemma.diffusion_gemma_provider import DiffusionGemmaModelProvider
from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge


pytestmark = pytest.mark.unit


def _config():
    text = SimpleNamespace(
        num_experts=128,
        top_k_experts=8,
        moe_intermediate_size=704,
        num_global_key_value_heads=2,
        hidden_size_per_layer_input=0,
    )
    return SimpleNamespace(model_type="diffusion_gemma", text_config=text, audio_config=None)


LAYER_TYPES = (["sliding_attention"] * 5 + ["full_attention"]) * 5


def _checkpoint_keys():
    """Tensor schema from google/diffusiongemma-26B-A4B-it revision 0f28bc4."""
    keys = {
        "model.decoder.embed_tokens.weight",
        "model.decoder.norm.weight",
        "model.decoder.self_conditioning.down_proj.weight",
        "model.decoder.self_conditioning.gate_proj.weight",
        "model.decoder.self_conditioning.pre_norm.weight",
        "model.decoder.self_conditioning.up_proj.weight",
        "model.encoder.embed_vision.embedding_projection.weight",
        "model.encoder.vision_tower.patch_embedder.input_proj.weight",
        "model.encoder.vision_tower.patch_embedder.position_embedding_table",
        "model.encoder.vision_tower.std_bias",
        "model.encoder.vision_tower.std_scale",
    }
    decoder_suffixes = [
        "experts.down_proj",
        "experts.gate_up_proj",
        "input_layernorm.weight",
        "layer_scalar",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "post_feedforward_layernorm_1.weight",
        "post_feedforward_layernorm_2.weight",
        "pre_feedforward_layernorm.weight",
        "pre_feedforward_layernorm_2.weight",
        "router.per_expert_scale",
        "router.proj.weight",
        "router.scale",
        "self_attn.k_norm.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.weight",
    ]
    for layer, layer_type in enumerate(LAYER_TYPES):
        keys.update(f"model.decoder.layers.{layer}.{suffix}" for suffix in decoder_suffixes)
        keys.add(f"model.encoder.language_model.layers.{layer}.layer_scalar")
        if layer_type == "sliding_attention":
            keys.add(f"model.decoder.layers.{layer}.self_attn.v_proj.weight")

    vision_suffixes = [
        "input_layernorm.weight",
        "mlp.down_proj.linear.weight",
        "mlp.gate_proj.linear.weight",
        "mlp.up_proj.linear.weight",
        "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "self_attn.k_norm.weight",
        "self_attn.k_proj.linear.weight",
        "self_attn.o_proj.linear.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.linear.weight",
        "self_attn.v_proj.linear.weight",
    ]
    for layer in range(27):
        keys.update(f"model.encoder.vision_tower.encoder.layers.{layer}.{suffix}" for suffix in vision_suffixes)
    return keys


def test_normalized_config_makes_implicit_architecture_explicit_without_mutating_source():
    source = _config()
    normalized = _normalized_diffusion_config(source)
    assert normalized.text_config.enable_moe_block is True
    assert normalized.text_config.attention_k_eq_v is True
    assert not hasattr(source.text_config, "enable_moe_block")


def test_registry_uses_decoder_text_weights_and_encoder_vision_weights():
    bridge = DiffusionGemmaBridge()
    bridge.hf_config = _config()
    registry = bridge.mapping_registry()
    assert registry.megatron_to_hf_lookup("language_model.embedding.word_embeddings.weight").hf_param == (
        "model.decoder.embed_tokens.weight"
    )
    assert registry.megatron_to_hf_lookup("vision_tower.patch_embedder.input_proj.weight").hf_param == (
        "model.encoder.vision_tower.patch_embedder.input_proj.weight"
    )
    assert registry.megatron_to_hf_lookup("self_conditioning.gate_proj.weight").hf_param == (
        "model.decoder.self_conditioning.gate_proj.weight"
    )
    scalar = registry.megatron_to_hf_lookup("language_model.decoder.layers.3.layer_scalar")
    assert scalar.hf_param == {
        "decoder": "model.decoder.layers.3.layer_scalar",
        "encoder": "model.encoder.language_model.layers.3.layer_scalar",
    }


def test_every_pinned_checkpoint_tensor_has_a_mapping():
    bridge = DiffusionGemmaBridge()
    bridge.hf_config = _config()
    registry = bridge.mapping_registry()
    keys = _checkpoint_keys()
    assert len(keys) == 1047
    assert [key for key in sorted(keys) if registry.hf_to_megatron_lookup(key) is None] == []


def test_tied_mapping_rejects_mismatched_copies():
    mapping = _TiedReplicatedMapping("x", decoder="decoder.x", encoder="encoder.x")
    with pytest.raises(ValueError, match="tied tensors differ"):
        mapping.hf_to_megatron({"decoder": torch.ones(1), "encoder": torch.zeros(1)}, torch.nn.Linear(1, 1))


def test_normalized_config_infers_global_kv_heads_from_first_full_attention_layer():
    source = _config()
    del source.text_config.num_global_key_value_heads
    source.text_config.layer_types = ["sliding_attention", "full_attention", "full_attention"]
    source.text_config.per_layer_config = [
        SimpleNamespace(num_key_value_heads=8),
        SimpleNamespace(num_key_value_heads=3),
        SimpleNamespace(num_key_value_heads=1),
    ]
    assert _normalized_diffusion_config(source).text_config.num_global_key_value_heads == 3
    assert not hasattr(source.text_config, "num_global_key_value_heads")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: setattr(config, "model_type", "gemma4"), "Expected model_type='diffusion_gemma'"),
        (lambda config: setattr(config.text_config, "num_experts", None), "missing required fields"),
        (lambda config: delattr(config.text_config, "num_global_key_value_heads"), "num_global_key_value_heads"),
        (lambda config: setattr(config.text_config, "hidden_size_per_layer_input", 2), "per-layer embeddings"),
        (lambda config: setattr(config, "audio_config", object()), "audio inputs"),
    ],
)
def test_normalized_config_rejects_unsupported_or_incomplete_checkpoints(mutate, message):
    config = _config()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        _normalized_diffusion_config(config)


def test_tied_mapping_imports_once_exports_both_copies_and_resolves_wildcards(monkeypatch):
    monkeypatch.setattr(_TiedReplicatedMapping, "tp_size", property(lambda self: 1))
    mapping = _TiedReplicatedMapping(
        "language_model.decoder.layers.*.layer_scalar",
        decoder="model.decoder.layers.*.layer_scalar",
        encoder="model.encoder.language_model.layers.*.layer_scalar",
    )
    resolved = mapping.resolve(("4",))
    assert isinstance(resolved, _TiedReplicatedMapping)
    assert resolved.megatron_param == "language_model.decoder.layers.4.layer_scalar"
    assert resolved.hf_param == {
        "decoder": "model.decoder.layers.4.layer_scalar",
        "encoder": "model.encoder.language_model.layers.4.layer_scalar",
    }

    weight = torch.tensor([1.5])
    assert torch.equal(
        resolved.hf_to_megatron({"decoder": weight, "encoder": weight.clone()}, torch.nn.Linear(1, 1)), weight
    )
    broadcast_keys = []
    resolved.broadcast_from_pp_rank = lambda tensor, cache_key: broadcast_keys.append(cache_key) or tensor
    resolved.maybe_dequantize = lambda tensor: tensor * 2
    exported = resolved.megatron_to_hf(weight, None)
    assert broadcast_keys == [str(resolved.hf_param)]
    assert exported.keys() == {resolved.hf_param["decoder"], resolved.hf_param["encoder"]}
    assert all(torch.equal(value, weight * 2) for value in exported.values())
    resolved.broadcast_from_pp_rank = lambda tensor, cache_key: None
    assert resolved.megatron_to_hf(weight, None) == {}


def test_bridge_text_config_and_dense_registry_contracts(monkeypatch):
    bridge = DiffusionGemmaBridge()
    assert bridge._text_config() is None
    gemma_text = SimpleNamespace(marker=True)
    bridge.hf_config = SimpleNamespace(model_type="gemma4", text_config=gemma_text)
    assert bridge._text_config() is gemma_text
    bridge.hf_config = _config()
    assert bridge._text_config().enable_moe_block is True
    assert bridge._conversion_mode() == "vl"
    assert bridge._hf_layer_prefix() == "model.decoder."

    monkeypatch.setattr(bridge, "_is_dense_config", lambda: True)
    with pytest.raises(ValueError, match="only the 26B-A4B MoE"):
        bridge.mapping_registry()


def test_provider_bridge_rejects_unexpected_parent_provider(monkeypatch):
    monkeypatch.setattr(Gemma4VLBridge, "provider_bridge", lambda self, hf_pretrained: object())
    with pytest.raises(TypeError, match="Expected a Gemma4 VL MoE provider"):
        DiffusionGemmaBridge().provider_bridge(SimpleNamespace(config=_config()))


def test_export_config_restores_diffusion_gemma_schema(monkeypatch):
    def parent_config(cls, provider):
        return {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "audio_config": {"unused": True},
            "audio_token_id": 1,
            "video_token_id": 2,
            "text_config": {
                "architectures": ["Gemma4ForCausalLM"],
                "enable_moe_block": True,
                "attention_k_eq_v": True,
                "model_type": "gemma4_text",
                "hidden_size": 8,
            },
        }

    monkeypatch.setattr(Gemma4VLBridge, "megatron_to_hf_config", classmethod(parent_config))
    provider = SimpleNamespace(canvas_length=128)
    config = DiffusionGemmaBridge.megatron_to_hf_config(provider)

    assert config == {
        "architectures": ["DiffusionGemmaForBlockDiffusion"],
        "canvas_length": 128,
        "model_type": "diffusion_gemma",
        "text_config": {"model_type": "diffusion_gemma_text", "hidden_size": 8},
    }


def test_provider_rejects_audio_and_applies_all_freeze_flags(monkeypatch):
    from megatron.bridge.models.diffusion_gemma import diffusion_gemma_provider as provider_module

    created = []

    class _Model:
        def __init__(self, provider, **kwargs):
            self.kwargs = kwargs
            self.freeze_calls = []
            created.append(self)

        def freeze(self, **kwargs):
            self.freeze_calls.append(kwargs)

    monkeypatch.setattr(provider_module, "DiffusionGemmaModel", _Model)
    provider = DiffusionGemmaModelProvider.__new__(DiffusionGemmaModelProvider)
    provider.audio_config = object()
    with pytest.raises(ValueError, match="does not support audio"):
        provider.provide()

    provider.audio_config = None
    provider.freeze_language_model = False
    provider.freeze_vision_model = False
    provider.freeze_vision_projection = False
    provider.freeze_self_conditioning = False
    model = provider.provide(pre_process=True, post_process=False, vp_stage=3)
    assert model.kwargs == {"pre_process": True, "post_process": False, "vp_stage": 3}
    assert model.freeze_calls == []

    provider.freeze_self_conditioning = True
    model = provider.provide()
    assert model.freeze_calls == [
        {
            "freeze_language_model": False,
            "freeze_vision_model": False,
            "freeze_vision_projection": False,
            "freeze_self_conditioning": True,
        }
    ]
    assert len(created) == 2
