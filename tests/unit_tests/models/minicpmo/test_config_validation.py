# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from megatron.bridge.models.minicpmo.config_validation import validate_minicpmo_config


@pytest.fixture
def config():
    return dict(
        model_type="minicpmo",
        version="4.5",
        architectures=["MiniCPMO"],
        init_vision=True,
        init_audio=True,
        init_tts=True,
        stream_input=True,
        vision_config={"hidden_size": 1152},
        audio_config={"d_model": 1024},
    )


def test_understanding_config_is_independent(config):
    original = deepcopy(config)
    result = validate_minicpmo_config(config)
    assert result["init_tts"] is False
    assert result["stream_input"] is False
    assert result["use_cache"] is False
    result["vision_config"]["hidden_size"] = 1
    assert config == original


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", "2.6"),
        ("model_type", "qwen3_omni_moe"),
        ("architectures", ["Qwen3ForCausalLM"]),
        ("init_audio", "true"),
        ("vision_config", None),
        ("audio_config", None),
    ],
)
def test_reject_incompatible_checkpoint(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        validate_minicpmo_config(config)


def test_disabled_encoder_can_omit_config(config):
    config["init_audio"] = False
    del config["audio_config"]
    assert validate_minicpmo_config(config)["init_audio"] is False
