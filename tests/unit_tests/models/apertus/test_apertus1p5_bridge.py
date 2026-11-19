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

from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.apertus import APERTUS1P5_AVAILABLE


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not APERTUS1P5_AVAILABLE, reason="transformers has no Apertus1p5ForConditionalGeneration"),
]

if APERTUS1P5_AVAILABLE:
    from transformers import Apertus1p5Config, Apertus1p5ForConditionalGeneration

    from megatron.bridge.models.apertus.apertus_bridge import get_apertus_decoder_block_spec, xielu_placeholder
    from megatron.bridge.models.apertus.apertus1p5_bridge import (
        Apertus1p5Bridge,
        _TruncatedVocabEmbeddingMapping,
    )
    from megatron.bridge.models.conversion.param_mapping import AutoMapping, ReplicatedMapping
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

VOCAB, OUTPUT_VOCAB = 266752, 131072


def _config(output_vocab_size=OUTPUT_VOCAB, **text_overrides):
    text_config = dict(
        vocab_size=VOCAB,
        output_vocab_size=output_vocab_size,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_act="xielu",
        max_position_embeddings=65536,
        rms_norm_eps=1e-5,
        attention_bias=False,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "llama3",
            "rope_theta": 12000000.0,
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        },
    )
    text_config.update(text_overrides)
    return Apertus1p5Config(text_config=text_config, tie_word_embeddings=False)


def _pretrained(config):
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = config
    pretrained.generation_config = Mock()
    pretrained.model = Mock(spec=Apertus1p5ForConditionalGeneration)
    return pretrained


def _bridge(config):
    bridge = Apertus1p5Bridge()
    bridge.hf_config = config
    return bridge


def test_provider_reads_text_config():
    provider = Apertus1p5Bridge().provider_bridge(_pretrained(_config()))

    assert provider.num_layers == 2
    assert provider.hidden_size == 1024
    assert provider.ffn_hidden_size == 4096
    assert provider.num_attention_heads == 8
    assert provider.num_query_groups == 4
    assert provider.qk_layernorm is True
    assert provider.gated_linear_unit is False
    assert provider.rotary_base == 12000000.0
    assert provider.rope_scaling_factor == 8.0
    assert provider.activation_func is xielu_placeholder
    assert provider.transformer_layer_spec is get_apertus_decoder_block_spec


def test_pruned_head_sets_vocab_to_output_vocab():
    assert Apertus1p5Bridge().provider_bridge(_pretrained(_config())).vocab_size == OUTPUT_VOCAB
    assert Apertus1p5Bridge().provider_bridge(_pretrained(_config(output_vocab_size=None))).vocab_size == VOCAB


def test_provider_rejects_non_xielu():
    with pytest.raises(ValueError, match="hidden_act='xielu'"):
        Apertus1p5Bridge().provider_bridge(_pretrained(_config(hidden_act="silu")))


def test_hf_config_export_nests_text_config():
    config = _config()
    provider = Apertus1p5Bridge().provider_bridge(_pretrained(config))
    hf_config = Apertus1p5Bridge.megatron_to_hf_config(provider)

    text = hf_config["text_config"]
    assert text["hidden_act"] == "xielu"
    assert text["rope_parameters"]["rope_theta"] == 12000000.0


def test_mappings_use_language_model_prefix():
    registry = _bridge(_config(output_vocab_size=None)).mapping_registry()

    qkv_out = registry.megatron_to_hf_lookup("decoder.layers.1.self_attention.linear_proj.weight")
    assert qkv_out.hf_param == "model.language_model.layers.1.self_attn.o_proj.weight"
    alpha = registry.megatron_to_hf_lookup("decoder.layers.1.mlp.activation_func.alpha_p")
    assert isinstance(alpha, ReplicatedMapping)
    assert alpha.hf_param == "model.language_model.layers.1.mlp.act_fn.alpha_p"
    assert registry.megatron_to_hf_lookup("output_layer.weight").hf_param == "lm_head.weight"


def test_pruned_head_truncates_input_embedding():
    registry = _bridge(_config()).mapping_registry()
    mapping = registry.megatron_to_hf_lookup("embedding.word_embeddings.weight")

    assert isinstance(mapping, _TruncatedVocabEmbeddingMapping)
    assert mapping.hf_param == "model.language_model.embed_tokens.weight"

    seen = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            AutoMapping, "hf_to_megatron", lambda self, weights, module: seen.setdefault("shape", tuple(weights.shape))
        )
        mapping.hf_to_megatron(torch.zeros(VOCAB, 4), None)
    assert seen["shape"] == (OUTPUT_VOCAB, 4)
