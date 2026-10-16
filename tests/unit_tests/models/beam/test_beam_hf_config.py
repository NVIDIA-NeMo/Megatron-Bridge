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

"""Public Beam metadata must load without a Transformers Beam model implementation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from transformers import AutoConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.beam.beam_config import BeamModelConfig
from megatron.bridge.models.beam.configuration_beam import BeamConfig


pytestmark = pytest.mark.unit


@pytest.fixture
def checkpoint_metadata(tmp_path):
    # reflection/Beam-1.0-501B-A23B revision 06641076793dad51b8907f7eb2a14f4e08fd3ec8.
    config = json.loads((Path(__file__).parent / "fixtures" / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path, config


def test_public_config_loads_through_autobridge_without_weights(checkpoint_metadata):
    path, source = checkpoint_metadata
    bridge = AutoBridge.from_hf_pretrained(path)
    cfg = bridge.get_model_config()
    assert isinstance(cfg, BeamModelConfig)
    assert cfg.hybrid_layer_pattern.count("*") == source["num_hidden_layers"]
    assert cfg.hybrid_layer_pattern.count("E") == 49
    assert cfg.num_moe_experts == source["num_experts"]
    assert not cfg.share_embeddings_and_output_weights
    assert cfg.no_rope_freq[10] == 1 and cfg.no_rope_freq[0] == 0
    assert cfg.post_self_attn_layernorm and cfg.post_mlp_layernorm and cfg.depth_scaled_residual
    assert cfg.moe_router_bias_update_rate == cfg.moe_aux_loss_coeff == 0


@pytest.mark.parametrize("config_class", [AutoConfig, BeamConfig])
def test_config_roundtrip_preserves_published_fields(checkpoint_metadata, tmp_path, config_class):
    path, source = checkpoint_metadata
    cfg = config_class.from_pretrained(path)
    for key, value in source.items():
        assert getattr(cfg, key) == value, key
    output = tmp_path / "roundtrip"
    cfg.save_pretrained(output)
    restored = config_class.from_pretrained(output)
    for key, value in source.items():
        assert getattr(restored, key) == value, key
    assert set(restored.rope_parameters) == {"sliding_attention", "full_attention"}
    assert restored.rope_parameters["full_attention"] is None


def test_compatibility_config_retains_native_attention_and_mlp_defaults():
    cfg = BeamConfig()
    assert len(cfg.layer_types) == len(cfg.mlp_layer_types) == cfg.num_hidden_layers == 52
    assert [i for i, kind in enumerate(cfg.layer_types) if kind == "full_attention"] == [
        5,
        11,
        17,
        23,
        29,
        35,
        41,
        47,
        51,
    ]
    assert cfg.mlp_layer_types == ["dense"] * 3 + ["sparse"] * 49
    assert cfg.rope_parameters["full_attention"] is None
    assert cfg.rope_parameters["sliding_attention"]["rope_type"] == "default"
    assert not cfg.tie_word_embeddings


def test_bridge_import_preserves_an_existing_transformers_registration():
    code = """
from transformers import AutoConfig, PreTrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

class ExistingBeamConfig(PreTrainedConfig):
    model_type = "beam"

AutoConfig.register("beam", ExistingBeamConfig, exist_ok=True)
import megatron.bridge.models.beam
assert CONFIG_MAPPING["beam"] is ExistingBeamConfig
assert type(AutoConfig.for_model("beam")) is ExistingBeamConfig
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
