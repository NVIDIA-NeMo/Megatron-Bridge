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
