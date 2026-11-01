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
from megatron.bridge.models.minimax_m3 import MiniMaxM3Bridge, MiniMaxM3ModelConfig
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


def _tiny_hf_config():
    configuration = pytest.importorskip("transformers.models.minimax_m3_vl.configuration_minimax_m3_vl")
    return configuration.MiniMaxM3VLConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 32,
            "dense_intermediate_size": 128,
            "shared_intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "rotary_dim": 8,
            "vocab_size": 1024,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "moe_layer_freq": [0, 1, 1, 1],
        },
        vision_config={"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4},
    )


class _BuilderOnlyBridge:
    @staticmethod
    def from_hf_pretrained(_hf_path: str, *, trust_remote_code: bool):
        assert trust_remote_code is True
        return _BuilderOnlyBridge()

    def get_model_config(self):
        return MiniMaxM3Bridge().hf_config_to_model_config(_tiny_hf_config())

    def to_megatron_provider(self, load_weights=False):
        raise AssertionError("recipes must use get_model_config()")


@pytest.mark.parametrize(
    "recipe",
    [
        _minimax_m3_module.minimax_m3_pretrain_256gpu_h100_bf16_config,
        _minimax_m3_module.minimax_m3_sft_128gpu_h100_bf16_config,
    ],
)
def test_text_recipe_uses_text_only_model_config(monkeypatch, recipe):
    patch_recipe_module_global(monkeypatch, _minimax_m3_module, "AutoBridge", _BuilderOnlyBridge)

    cfg = recipe()

    assert type(cfg.model) is MiniMaxM3ModelConfig
    assert not hasattr(cfg.model, "vision_config")
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_flex_dispatcher_num_sms == 16
    assert cfg.model.moe_permute_fusion_into_hybridep is False
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert cfg.env_vars["USE_MNNVL"] == 0
