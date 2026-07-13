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

import pytest

from megatron.bridge.models.conversion.utils import conform_config_to_reference
from megatron.bridge.models.qwen3_asr.hf_qwen3_asr.configuration_qwen3_asr import (
    Qwen3ASRConfig,
    Qwen3ASRThinkerConfig,
)
from megatron.bridge.training.config import ConfigContainer


pytestmark = [pytest.mark.unit]


def test_qwen3_asr_config_default_constructs_thinker_config():
    config = Qwen3ASRConfig()

    assert isinstance(config.thinker_config, Qwen3ASRThinkerConfig)
    assert config.get_text_config() is config.thinker_config.text_config


def test_qwen3_asr_config_from_dict_constructs_thinker_config():
    config = Qwen3ASRConfig.from_dict(
        {
            "model_type": "qwen3_asr",
            "thinker_config": {
                "audio_config": {"encoder_layers": 2},
                "text_config": {
                    "hidden_size": 128,
                    "intermediate_size": 256,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "vocab_size": 512,
                },
            },
        }
    )

    assert isinstance(config.thinker_config, Qwen3ASRThinkerConfig)
    assert config.thinker_config.audio_config.encoder_layers == 2
    assert config.thinker_config.text_config.hidden_size == 128


def test_qwen3_asr_config_conforming_preserves_reference_audio_subconfig():
    reference_config = Qwen3ASRConfig.from_dict(
        {
            "model_type": "qwen3_asr",
            "architectures": ["Qwen3ASRForConditionalGeneration"],
            "thinker_config": {
                "audio_config": {
                    "d_model": 1024,
                    "encoder_layers": 32,
                    "encoder_attention_heads": 16,
                    "encoder_ffn_dim": 4096,
                },
                "text_config": {
                    "hidden_size": 2048,
                    "intermediate_size": 8192,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "vocab_size": 151936,
                },
            },
        }
    )
    megatron_derived_config = {
        "model_type": "qwen3_asr",
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "thinker_config": {
            "text_config": {
                "hidden_size": 3584,
                "intermediate_size": 18944,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "vocab_size": 151936,
            },
        },
    }

    conformed_config = conform_config_to_reference(megatron_derived_config, reference_config.to_dict())
    config = Qwen3ASRConfig(**conformed_config)

    assert config.thinker_config.audio_config.d_model == 1024
    assert config.thinker_config.audio_config.encoder_attention_heads == 16
    assert config.thinker_config.text_config.hidden_size == 3584


def test_qwen3_asr_thinker_config_serializes_nested_subconfigs_for_run_config():
    config = Qwen3ASRThinkerConfig(
        audio_config={
            "d_model": 1024,
            "encoder_layers": 24,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "output_dim": 2048,
        },
        text_config={
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
        },
    )

    serialized = ConfigContainer._convert_value_to_dict(config)

    assert serialized["audio_config"]["d_model"] == 1024
    assert serialized["text_config"]["hidden_size"] == 2048
    assert serialized["_target_"].endswith("Qwen3ASRThinkerConfig")
