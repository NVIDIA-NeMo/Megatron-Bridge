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

from types import SimpleNamespace

import pytest

from megatron.bridge.recipes.gemma.h100.gemma2 import gemma2_2b_pretrain_2gpu_h100_bf16_config
from megatron.bridge.recipes.gemma.h100.gemma_text import gemma_2b_pretrain_1gpu_h100_bf16_config
from megatron.bridge.recipes.utils import text_pretrain_utils


pytestmark = pytest.mark.unit


def test_build_text_pretrain_config_uses_pinned_lazy_bridge(monkeypatch):
    calls = []
    provider = SimpleNamespace()

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return provider

    def _from_hf_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return _Bridge()

    monkeypatch.setattr(text_pretrain_utils.AutoBridge, "from_hf_pretrained", _from_hf_pretrained)

    config = text_pretrain_utils.build_text_pretrain_config(
        hf_model_id="org/model",
        revision="a" * 40,
        tensor_parallelism=2,
        pipeline_parallelism=2,
        expert_parallelism=8,
        sequence_length=2048,
        trust_remote_code=True,
    )

    assert calls == [("org/model", {"revision": "a" * 40, "trust_remote_code": True})]
    assert config.model is provider
    assert config.model.tensor_model_parallel_size == 2
    assert config.model.pipeline_model_parallel_size == 2
    assert config.model.expert_model_parallel_size == 8
    assert config.model.sequence_parallel is True
    assert config.model.seq_length == 2048
    assert config.dataset.seq_length == 2048
    assert config.logger.log_interval == 1


def test_gemma_pretrain_configs_do_not_require_gated_hf_access(monkeypatch):
    def _unexpected_hf_access(*_args, **_kwargs):
        raise AssertionError("from-scratch Gemma pretrain must not access a gated HF config")

    monkeypatch.setattr(text_pretrain_utils.AutoBridge, "from_hf_pretrained", _unexpected_hf_access)

    gemma = gemma_2b_pretrain_1gpu_h100_bf16_config()
    gemma2 = gemma2_2b_pretrain_2gpu_h100_bf16_config()

    assert gemma.model.num_layers == 18
    assert gemma.model.hidden_size == 2048
    assert gemma2.model.num_layers == 26
    assert gemma2.model.hidden_size == 2304
    assert gemma2.model.query_pre_attn_scalar == 256
