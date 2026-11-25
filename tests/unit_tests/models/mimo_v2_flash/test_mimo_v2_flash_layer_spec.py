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

"""Unit tests for mimo_v2_flash_layer_spec with pipeline parallelism."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.models.gpt import gpt_layer_specs
from megatron.core.transformer import TransformerConfig

from megatron.bridge.models.gpt_provider import _callable_spec_kwargs
from megatron.bridge.models.mimo_v2_flash import modeling_mimo_v2_flash
from megatron.bridge.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashSelfAttention,
    MiMoV2FlashTEDotProductAttention,
    mimo_v2_flash_layer_spec,
)


def _layer_spec_stub(index):
    attention = SimpleNamespace(module=None, submodules=SimpleNamespace(core_attention=None))
    return SimpleNamespace(index=index, submodules=SimpleNamespace(self_attention=attention))


def test_provider_passes_vp_stage_and_pp_rank_to_the_block_spec(monkeypatch):
    """GPTModelProvider.provide() forwards only the arguments the spec callable declares."""
    calls = []

    def block_spec(config, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(layer_specs=[])

    monkeypatch.setattr(modeling_mimo_v2_flash, "get_gpt_decoder_block_spec", block_spec)
    config = SimpleNamespace()

    mimo_v2_flash_layer_spec(config, **_callable_spec_kwargs(mimo_v2_flash_layer_spec, config, vp_stage=1, pp_rank=0))

    assert calls == [{"use_transformer_engine": True, "vp_stage": 1, "pp_rank": 0}]


@pytest.mark.parametrize(
    "pp_rank, vp_stage, expected", [(0, 0, [0, 1]), (1, 0, [2, 3]), (0, 1, [4, 5]), (1, 1, [6, 7])]
)
def test_layer_spec_builds_its_virtual_pipeline_chunk(monkeypatch, pp_rank, vp_stage, expected):
    """With 8 layers, 2 pipeline stages and 2 virtual stages, each chunk gets its own two layers."""
    layer_specs = [_layer_spec_stub(index) for index in range(8)]
    monkeypatch.setattr(gpt_layer_specs, "get_gpt_decoder_layer_specs", lambda *args, **kwargs: layer_specs)
    config = TransformerConfig(
        num_layers=8,
        hidden_size=64,
        num_attention_heads=4,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
    )

    spec = mimo_v2_flash_layer_spec(config, vp_stage=vp_stage, pp_rank=pp_rank)

    assert [layer_spec.index for layer_spec in spec.layer_specs] == expected
    for layer_spec in spec.layer_specs:
        assert layer_spec.submodules.self_attention.module is MiMoV2FlashSelfAttention
        assert layer_spec.submodules.self_attention.submodules.core_attention is MiMoV2FlashTEDotProductAttention
