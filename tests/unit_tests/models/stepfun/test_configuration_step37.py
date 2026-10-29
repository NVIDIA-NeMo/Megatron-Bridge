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

import json

import pytest

from megatron.bridge.models.hf_pretrained.safe_config_loader import safe_load_config_with_retry
from megatron.bridge.models.stepfun.step35_bridge import Step35Config
from megatron.bridge.models.stepfun.step37_bridge import Step37Config


pytestmark = pytest.mark.unit


def _write_config(path, config):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))


def test_step37_config_load_and_roundtrip_preserve_step35_text_fields(tmp_path):
    step37_path = tmp_path / "step37"
    step37_config = {
        "architectures": ["Step3p7ForConditionalGeneration"],
        "auto_map": {
            "AutoConfig": "configuration_step3p7.Step3p7Config",
            "AutoProcessor": "processing_step3.Step3VLProcessor",
            "AutoModelForCausalLM": "modeling_step3p7.Step3p7ForConditionalGeneration",
        },
        "model_type": "step3p7",
        "image_token_id": 128001,
        "understand_projector_stride": 2,
        "projector_bias": False,
        "vision_config": {
            "model_type": "perception_encoder",
            "image_size": 728,
            "patch_size": 14,
            "width": 1536,
            "layers": 47,
            "heads": 16,
        },
        "text_config": {
            "architectures": ["Step3p5ForCausalLM"],
            "model_type": "step3p5",
            "hidden_size": 4096,
            "max_seq_len": 262144,
            "max_position_embeddings": 262144,
            "vocab_size": 128896,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 2.0,
                "original_max_position_embeddings": 131072,
                "low_freq_factor": 1.0,
                "high_freq_factor": 32.0,
            },
            "yarn_only_types": ["full_attention"],
        },
    }
    _write_config(step37_path, step37_config)

    loaded_step37 = safe_load_config_with_retry(step37_path, max_retries=0)

    assert isinstance(loaded_step37, Step37Config)
    assert loaded_step37.auto_map["AutoConfig"] == "configuration_step3p7.Step3p7Config"
    assert loaded_step37.max_position_embeddings == 262144
    assert loaded_step37.text_config.max_position_embeddings == 262144
    assert loaded_step37.text_config.yarn_only_types == ["full_attention"]

    roundtrip_path = tmp_path / "step37-roundtrip"
    loaded_step37.save_pretrained(roundtrip_path)
    reloaded_step37 = safe_load_config_with_retry(roundtrip_path, max_retries=0)
    assert reloaded_step37.text_config.max_position_embeddings == 262144
    assert reloaded_step37.text_config.yarn_only_types == ["full_attention"]
    loaded_dict = loaded_step37.to_dict()
    reloaded_dict = reloaded_step37.to_dict()
    loaded_dict.pop("_name_or_path")
    reloaded_dict.pop("_name_or_path")
    assert reloaded_dict == loaded_dict


def test_step35_config_load_preserves_inherited_fields(tmp_path):
    step35_path = tmp_path / "step35"
    step35_config = {
        "architectures": ["Step3p5ForCausalLM"],
        "model_type": "step3p5",
        "hidden_size": 2048,
        "max_position_embeddings": 131072,
        "step35_only_flag": "preserved",
    }
    _write_config(step35_path, step35_config)

    loaded_step35 = safe_load_config_with_retry(step35_path, max_retries=0)
    assert isinstance(loaded_step35, Step35Config)
    assert loaded_step35.hidden_size == 2048
    assert loaded_step35.max_position_embeddings == 131072
    assert loaded_step35.step35_only_flag == "preserved"
