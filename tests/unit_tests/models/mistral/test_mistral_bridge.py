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

from dataclasses import fields
from unittest.mock import Mock, patch

import pytest
import torch.nn.functional as F
from transformers import MistralConfig

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mistral.mistral_bridge import MistralBridge
from megatron.bridge.models.mistral.mistral_provider import MistralModelProvider


pytestmark = pytest.mark.unit

_YARN = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 1024,
    "beta_fast": 24.0,
    "beta_slow": 2.0,
    "mscale": 0.8,
    "mscale_all_dim": 0.5,
    "truncate": False,
}


def _mistral_config(**overrides) -> MistralConfig:
    values = dict(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        vocab_size=128,
        rope_theta=1000000.0,
        torch_dtype="bfloat16",
    )
    values.update(overrides)
    config = MistralConfig(**values)
    config.architectures = ["MistralForCausalLM"]
    return config


def _provider(config) -> MistralModelProvider:
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = config
    with pytest.warns(FutureWarning, match=r"deprecated.*get_model_config.*get_model"):
        return MistralBridge().provider_bridge(pretrained)


def test_provider_bridge_is_inherited_compatibility_only():
    assert "provider_bridge" not in MistralBridge.__dict__


def test_conversion_uses_builder_config():
    assert MistralBridge.MODEL_CONFIG_CLASS is BridgeGPTModelConfig
    assert MistralBridge.USE_MODEL_CONFIG_FOR_CONVERSION is True
    assert MistralBridge.PROVIDER_CLASS is MistralModelProvider


def test_hf_config_to_model_config_uses_direct_mapping():
    config = _mistral_config(sliding_window=32, head_dim=32)
    bridge = MistralBridge()
    with (
        patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
        patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
    ):
        model_config = bridge.hf_config_to_model_config(config)

    assert isinstance(model_config, BridgeGPTModelConfig)
    assert model_config.normalization == "RMSNorm"
    assert model_config.activation_func is F.silu
    assert model_config.gated_linear_unit is True
    assert model_config.add_bias_linear is False
    assert model_config.position_embedding_type == "rope"
    assert model_config.rotary_base == 1000000.0
    assert model_config.kv_channels == 32
    assert list(model_config.window_size) == [31, 0]
    assert model_config.cp_comm_type == "a2a"


@pytest.mark.parametrize(
    "overrides",
    [
        {"sliding_window": 32},
        {"sliding_window": None, "head_dim": 32},
        {"sliding_window": None, "tie_word_embeddings": True, "torch_dtype": "float16"},
        {"sliding_window": None, "rope_scaling": _YARN},
    ],
    ids=["ministral_like", "nemo_like", "tied_fp16", "yarn"],
)
def test_model_config_matches_provider_runtime_config(overrides):
    config = _mistral_config(**overrides)
    model_config = MistralBridge().hf_config_to_model_config(config)
    provider = _provider(config)

    assert isinstance(provider, MistralModelProvider)
    model_config_fields = {field.name for field in fields(model_config)}
    model_config_fields.update(field.name for field in fields(model_config.transformer))
    shared = ({field.name for field in fields(provider)} & model_config_fields) - {"transformer_layer_spec"}
    for field_name in sorted(shared):
        assert getattr(model_config, field_name) == getattr(provider, field_name), field_name


def test_yarn_rope_scaling_maps_to_yarn_fields_on_both_paths():
    """YaRN checkpoints build on both paths (previously a TypeError, #5627; test case from #6319)."""
    config = _mistral_config(sliding_window=None, rope_scaling=_YARN)

    for built in (MistralBridge().hf_config_to_model_config(config), _provider(config)):
        assert built.position_embedding_type == "yarn"
        assert built.yarn_rotary_scaling_factor == 4.0
        assert built.yarn_original_max_position_embeddings == 1024
        assert built.yarn_beta_fast == 24.0
        assert built.yarn_beta_slow == 2.0
        assert built.yarn_mscale == 0.8
        assert built.yarn_mscale_all_dim == 0.5
        assert built.yarn_correction_range_round_to_int is False


@pytest.mark.parametrize("sliding_window", [32, None])
def test_megatron_to_hf_config_from_model_config(sliding_window):
    model_config = MistralBridge().hf_config_to_model_config(_mistral_config(sliding_window=sliding_window))

    hf_config = MistralBridge.megatron_to_hf_config(model_config)

    assert hf_config["sliding_window"] == sliding_window
    assert hf_config["num_hidden_layers"] == 2
