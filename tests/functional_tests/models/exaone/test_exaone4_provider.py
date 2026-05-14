# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.exaone import Exaone4ModelProvider1P2B
from tests.functional_tests.utils import compare_provider_configs


class TestExaone4ProviderMapping:
    """Test EXAONE 4.0 config-only AutoBridge provider mapping."""

    def test_bridge_vs_predefined_provider_config_from_config_only(self):
        cfg = PretrainedConfig(
            architectures=["Exaone4ForCausalLM"],
            hidden_size=2048,
            initializer_range=0.02,
            intermediate_size=4096,
            max_position_embeddings=65536,
            model_type="exaone4",
            num_attention_heads=32,
            num_hidden_layers=30,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            rope_scaling={
                "rope_type": "llama3",
                "factor": 16.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
            rope_theta=1000000.0,
            tie_word_embeddings=True,
            torch_dtype=torch.bfloat16,
            vocab_size=102400,
            head_dim=64,
        )

        bridge = AutoBridge.from_hf_config(cfg)
        converted_provider = bridge.to_megatron_provider(load_weights=False)
        converted_provider.finalize()

        predefined_provider = Exaone4ModelProvider1P2B()
        predefined_provider.finalize()

        compare_provider_configs(converted_provider, predefined_provider, "exaone4-config-only")
