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

import megatron.bridge.recipes.minimax.h100.minimax_m3 as _minimax_m3_module
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


class _FakeProvider:
    freeze_language_model = False
    freeze_vision_model = False
    freeze_vision_projection = False

    def finalize(self):
        return None


class _FakeAutoBridge:
    @staticmethod
    def from_hf_pretrained(_hf_path: str, *, trust_remote_code: bool):
        assert trust_remote_code is True
        return _FakeAutoBridge()

    def to_megatron_provider(self, *, load_weights: bool):
        assert load_weights is False
        return _FakeProvider()


@pytest.mark.parametrize(
    "recipe",
    [
        _minimax_m3_module.minimax_m3_pretrain_256gpu_h100_bf16_config,
        _minimax_m3_module.minimax_m3_sft_128gpu_h100_bf16_config,
    ],
)
def test_text_recipe_freezes_unused_vlm_modules(monkeypatch, recipe):
    patch_recipe_module_global(monkeypatch, _minimax_m3_module, "AutoBridge", _FakeAutoBridge)

    cfg = recipe()

    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is True
    assert cfg.model.freeze_language_model is not True
