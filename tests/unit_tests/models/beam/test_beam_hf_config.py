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

"""Public Beam metadata loads and round-trips through native Transformers configuration."""

import json
from pathlib import Path

import pytest
from transformers import AutoConfig, BeamConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.beam.beam_config import BeamModelConfig


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
    assert isinstance(AutoConfig.from_pretrained(path), BeamConfig)
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
