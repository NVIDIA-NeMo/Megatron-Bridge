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
from transformers import MistralConfig

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mistral.mistral_bridge import MistralBridge
from megatron.bridge.models.mistral.mistral_provider import MistralModelProvider


@pytest.mark.unit
def test_provider_bridge_yarn_rope_scaling() -> None:
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = MistralConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 1024,
            "beta_fast": 24.0,
            "beta_slow": 2.0,
            "mscale": 0.8,
            "mscale_all_dim": 0.5,
            "truncate": False,
        },
    )

    try:
        provider = MistralBridge().provider_bridge(pretrained)
    except TypeError as error:
        raise AssertionError(f"YaRN provider construction failed: {error}") from error

    assert isinstance(provider, MistralModelProvider)
    assert provider.position_embedding_type == "yarn"
    assert provider.yarn_rotary_scaling_factor == 4.0
    assert provider.yarn_original_max_position_embeddings == 1024
    assert provider.yarn_beta_fast == 24.0
    assert provider.yarn_beta_slow == 2.0
    assert provider.yarn_mscale == 0.8
    assert provider.yarn_mscale_all_dim == 0.5
    assert provider.yarn_correction_range_round_to_int is False
