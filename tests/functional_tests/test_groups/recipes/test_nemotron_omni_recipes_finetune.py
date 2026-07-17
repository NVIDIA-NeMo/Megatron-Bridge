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

"""Functional smoke tests for Nemotron Omni finetuning recipes."""

import copy
import os
from dataclasses import dataclass

import pytest

from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import NemotronOmniModelProvider
from megatron.bridge.recipes.nemotron_omni import nemotron_omni_cord_v2_sft_config
from megatron.bridge.training import nemotron_omni_step
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_vl_recipe_test


_DEFAULT_PROCESSOR_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"


@dataclass
class _TinyNemotronOmniModelProvider(NemotronOmniModelProvider):
    """Small Omni provider used only for functional recipe smoke tests."""

    has_sound: bool = False
    language_model_type: str = "nemotron6-moe"
    hidden_size: int = 128
    ffn_hidden_size: int = 256
    num_attention_heads: int = 4
    num_query_groups: int = 2
    kv_channels: int = 32
    mamba_num_heads: int = 4
    mamba_head_dim: int = 32
    mamba_num_groups: int = 1
    mamba_state_dim: int = 16
    hybrid_layer_pattern: str = "M"
    vocab_size: int = 131072
    seq_length: int = 1024
    image_token_index: int = 18
    img_start_token_id: int = 21
    img_end_token_id: int = 22
    sound_context_token_id: int = 27
    tokenizer_type: str = "nemotron6-moe"
    use_vision_backbone_fp8_arch: bool = False
    vision_proj_ffn_hidden_size: int = 256
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    sequence_parallel: bool = False

    def _build_vision_config(self, language_cfg):
        vision_cfg = copy.deepcopy(language_cfg)
        vision_cfg.sequence_parallel = False
        vision_cfg.context_parallel_size = 1
        vision_cfg.tp_comm_overlap = False
        vision_cfg.recompute_granularity = None
        vision_cfg.recompute_method = None
        vision_cfg.recompute_num_layers = None
        vision_cfg.mtp_num_layers = None
        vision_cfg.num_layers = 1
        vision_cfg.pipeline_model_parallel_size = 1
        vision_cfg.num_attention_heads = 4
        vision_cfg.add_bias_linear = True
        vision_cfg.add_qkv_bias = True
        vision_cfg.hidden_size = 128
        vision_cfg.ffn_hidden_size = 256
        vision_cfg.gated_linear_unit = False
        vision_cfg.kv_channels = 32
        vision_cfg.num_query_groups = 4
        vision_cfg.normalization = "LayerNorm"
        vision_cfg.qk_layernorm = False
        vision_cfg.layernorm_epsilon = 1e-6
        return vision_cfg


class _TinyAutoBridge:
    @staticmethod
    def from_hf_pretrained(*_, **__):
        return _TinyAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        return _TinyNemotronOmniModelProvider()


def _tiny_nemotron_omni_cord_v2_sft_config():
    return nemotron_omni_cord_v2_sft_config()


class TestNemotronOmniRecipes:
    @pytest.mark.run_only_on("GPU")
    def test_nemotron_omni_finetune_recipe(self, monkeypatch, tmp_path):
        import megatron.bridge.recipes.nemotron_omni.h100.nemotron_omni as recipe_module

        processor_id = os.environ.get("NEMOTRON_OMNI_PROCESSOR_MODEL", _DEFAULT_PROCESSOR_ID)
        monkeypatch.setattr(recipe_module, "AutoBridge", _TinyAutoBridge)
        monkeypatch.setattr(recipe_module, "_DEFAULT_HF_PATH", processor_id)

        run_pretrain_vl_recipe_test(
            _tiny_nemotron_omni_cord_v2_sft_config,
            "nemotron_omni_cord_v2_sft",
            tmp_path,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            model_overrides={
                "sequence_parallel": False,
                "hybrid_layer_pattern": "M",
                "num_layers": None,
            },
            dataset_overrides={"trust_remote_code": True},
            forward_step_func=nemotron_omni_step.forward_step,
        )
