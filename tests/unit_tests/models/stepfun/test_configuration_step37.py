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

from megatron.bridge.models.stepfun.configuration_step37 import (
    Step37Config,
    Step37TextConfig,
    Step37VisionConfig,
)


pytestmark = pytest.mark.unit


def test_vision_config_defaults():
    cfg = Step37VisionConfig()
    assert cfg.model_type == "perception_encoder"
    assert cfg.width == 1536
    assert cfg.layers == 47
    assert cfg.patch_size == 14
    assert cfg.image_size == 728
    assert cfg.use_cls_token is False
    assert cfg.use_rope2d is True
    assert cfg.ls_init_value == 0.1


def test_text_config_model_type():
    assert Step37TextConfig.model_type == "step3p5"


def test_top_config_defaults_and_subconfigs():
    cfg = Step37Config()
    assert cfg.model_type == "step3p7"
    assert cfg.understand_projector_stride == 2
    assert cfg.projector_bias is False
    assert cfg.image_token_id == 128001
    assert isinstance(cfg.vision_config, Step37VisionConfig)
    assert isinstance(cfg.text_config, Step37TextConfig)


def test_top_config_mirrors_text_fields():
    cfg = Step37Config()
    assert cfg.hidden_size == cfg.text_config.hidden_size
    assert cfg.max_position_embeddings == cfg.text_config.max_position_embeddings


def test_subconfigs_accept_dicts():
    cfg = Step37Config(vision_config={"width": 999}, text_config={})
    assert isinstance(cfg.vision_config, Step37VisionConfig)
    assert cfg.vision_config.width == 999
    assert isinstance(cfg.text_config, Step37TextConfig)
