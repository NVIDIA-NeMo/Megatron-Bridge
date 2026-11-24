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

"""Tests of the RoPE base in the ERNIE 5.5 Hugging Face config's ``rope_parameters`` (``Ernie55Config``).

``conftest.py`` leaves ``test_ernie55_*.py`` out of the default collection, because building an ERNIE 5.5 model needs
Megatron-Core dev and CI runs Megatron-Core main. The cases here build no model and map no config: they construct,
save and reload an ``Ernie55Config``. They live outside that pattern so that CI runs them, and can go back into
``test_ernie55_bridge_config.py`` once the exclusion is lifted.

The case that exports a config from a Megatron checkpoint is
``test_checkpoint_export_config_carries_the_rope_base`` in ``test_ernie55_bridge_config.py``.
"""

from __future__ import annotations

import json

import pytest
from transformers import AutoConfig

from tests.unit_tests.models.ernie.ernie55_hf_configs import PROXY_CONFIG, proxy_config
from tests.unit_tests.models.ernie.ernie55_test_utils import allow_conversion_imports_without_te


# Bridge's conversion modules import its LoRA layers, which need Transformer Engine at import time.
allow_conversion_imports_without_te()

from megatron.bridge.models.ernie.configuration_ernie55 import Ernie55Config  # noqa: E402


pytestmark = pytest.mark.unit


def _hf(config: dict) -> Ernie55Config:
    return Ernie55Config(**{key: value for key, value in config.items() if key != "model_type"})


def test_rope_parameters_carry_the_rope_base(tmp_path):
    """transformers 5 keeps ``rope_scaling`` as ``rope_parameters``; its readers take the RoPE base from there."""
    # A released config.json: rope_theta at the top level, the H layers' YaRN settings in rope_scaling.
    released = proxy_config(
        drop=("rotary_scaling_factor", "original_max_position_embeddings"),
        hca_rope_type="yarn",
        rope_theta=8000000,
        rope_scaling={"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536},
    )
    (tmp_path / "config.json").write_text(json.dumps(released))
    config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(config, Ernie55Config)
    assert config.rope_parameters["rope_theta"] == config.rope_theta == 8000000
    assert config.rope_parameters["factor"] == 16

    config.save_pretrained(tmp_path / "saved")
    saved = json.loads((tmp_path / "saved" / "config.json").read_text())
    assert saved["rope_parameters"]["rope_theta"] == saved["rope_theta"] == 8000000
    reloaded = AutoConfig.from_pretrained(tmp_path / "saved")
    assert reloaded.rope_parameters == config.rope_parameters


def test_rope_parameters_follow_the_top_level_rope_base():
    rope_parameters = {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536}
    config = _hf(proxy_config(rope_theta=8e6, rope_parameters={**rope_parameters, "rope_theta": 10000.0}))
    assert config.rope_parameters["rope_theta"] == config.rope_theta == 8e6


def test_config_without_rope_scaling_gets_no_rope_parameters():
    assert "rope_parameters" not in _hf(PROXY_CONFIG).to_dict()
