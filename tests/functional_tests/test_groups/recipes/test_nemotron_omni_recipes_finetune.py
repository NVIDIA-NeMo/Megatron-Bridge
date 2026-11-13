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
from types import SimpleNamespace

import pytest

from megatron.bridge.models.nemotron_omni import nemotron_omni_builder
from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import NemotronOmniBridge
from megatron.bridge.recipes.nemotronh_multimodal import nemotron_omni_cord_v2_sft_config
from megatron.bridge.training import nemotron_omni_step
from tests.functional_tests.test_groups.recipes.utils import run_pretrain_vl_recipe_test


_DEFAULT_PROCESSOR_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"


def _tiny_hf_config() -> SimpleNamespace:
    """A small single-Mamba-layer Omni config for functional recipe smoke tests."""
    llm_config = SimpleNamespace(
        torch_dtype="bfloat16",
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        mamba_num_heads=4,
        mamba_head_dim=32,
        n_groups=1,
        ssm_state_size=16,
        hybrid_override_pattern="M",
        vocab_size=131072,
        max_position_embeddings=1024,
    )
    return SimpleNamespace(
        architectures=["NemotronH_Nano_Omni_Reasoning_V3"],
        llm_config=llm_config,
        img_context_token_id=18,
        projector_hidden_size=256,
    )


def _tiny_vision_config(settings, language_cfg):
    """A one-layer stand-in for the RADIO ViT-H vision encoder config."""
    del settings
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

    def get_model_config(self):
        return NemotronOmniBridge().hf_config_to_model_config(_tiny_hf_config())


def _tiny_nemotron_omni_cord_v2_sft_config():
    return nemotron_omni_cord_v2_sft_config()


class TestNemotronOmniRecipes:
    @pytest.mark.run_only_on("GPU")
    def test_nemotron_omni_finetune_recipe(self, monkeypatch, tmp_path):
        import megatron.bridge.recipes.nemotronh_multimodal.h100.nemotron_omni as recipe_module

        processor_id = os.environ.get("NEMOTRON_OMNI_PROCESSOR_MODEL", _DEFAULT_PROCESSOR_ID)
        monkeypatch.setattr(recipe_module, "AutoBridge", _TinyAutoBridge)
        monkeypatch.setattr(recipe_module, "_DEFAULT_HF_PATH", processor_id)
        monkeypatch.setattr(nemotron_omni_builder, "_make_omni_vision_config", _tiny_vision_config)

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
