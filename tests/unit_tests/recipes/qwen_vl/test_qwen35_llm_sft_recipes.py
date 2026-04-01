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

import importlib
from typing import Any, Callable

import pytest


_qwen35_llm_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen35_llm")

_LLM_SFT_FUNCS: list[Callable[..., Any]] = [
    _qwen35_llm_module.qwen35_llm_800m_sft_config,
    _qwen35_llm_module.qwen35_llm_2b_sft_config,
    _qwen35_llm_module.qwen35_llm_4b_sft_config,
    _qwen35_llm_module.qwen35_llm_9b_sft_config,
    _qwen35_llm_module.qwen35_llm_27b_sft_config,
    _qwen35_llm_module.qwen35_llm_35b_a3b_sft_config,
    _qwen35_llm_module.qwen35_llm_122b_a10b_sft_config,
    _qwen35_llm_module.qwen35_llm_397b_a17b_sft_config,
]


class _FakeModelCfg:
    def __init__(self):
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 2048
        self.freeze_language_model = False
        self.freeze_vision_model = True
        self.freeze_vision_projection = True
        self.mtp_num_layers = None

    def finalize(self):
        return None


class _FakeAutoBridge:
    @staticmethod
    def from_hf_pretrained(hf_path: str):
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False, skip_megatron_param_globs: list[str] | None = None):
        assert load_weights is True
        assert skip_megatron_param_globs == list(_qwen35_llm_module.QWEN35_LLM_SFT_SKIP_VISION_GLOBS)
        return _FakeModelCfg()


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.tokenizer.tokenizer_model is not None
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is True
    assert cfg.model.mtp_num_layers is None


@pytest.mark.parametrize("recipe_func", _LLM_SFT_FUNCS)
def test_each_qwen35_llm_sft_recipe_builds_config(recipe_func: Callable[..., Any], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_qwen35_llm_module, "AutoBridge", _FakeAutoBridge)
    cfg = recipe_func("dummy/hf/path")
    _assert_basic_config(cfg)


def test_qwen35_llm_skip_vision_globs_exported():
    assert "*vision_model*" in _qwen35_llm_module.QWEN35_LLM_SFT_SKIP_VISION_GLOBS
