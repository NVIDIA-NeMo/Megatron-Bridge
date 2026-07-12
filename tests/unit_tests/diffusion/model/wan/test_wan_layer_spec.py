# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch.nn as nn
from megatron.core.transformer.transformer_layer import TransformerLayer

import megatron.bridge.diffusion.models.wan.wan_layer_spec as wan_layer_spec
from megatron.bridge.diffusion.models.wan.wan_layer_spec import (
    WanLayerWithAdaLN,
    WanWithAdaLNSubmodules,
    get_wan_block_with_transformer_engine_spec,
)


def test_get_wan_block_with_transformer_engine_spec_basic():
    spec = get_wan_block_with_transformer_engine_spec()
    # Basic structure checks
    assert hasattr(spec, "module")
    assert hasattr(spec, "submodules")
    sub = spec.submodules
    # Expected submodule fields exist
    for name in ["norm1", "norm2", "norm3", "full_self_attention", "cross_attention", "mlp"]:
        assert hasattr(sub, name), f"Missing submodule {name}"


def test_wan_layer_forwards_virtual_pipeline_stage(monkeypatch):
    captured = {}

    def fake_transformer_layer_init(
        self,
        *,
        config,
        submodules,
        layer_number,
        hidden_dropout,
        pg_collection=None,
        vp_stage=None,
    ):
        nn.Module.__init__(self)
        self.config = config
        self.cross_attention = nn.Identity()
        self.mlp = nn.Identity()
        captured["vp_stage"] = vp_stage

    monkeypatch.setattr(TransformerLayer, "__init__", fake_transformer_layer_init)
    monkeypatch.setattr(wan_layer_spec, "build_module", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(wan_layer_spec, "WanAdaLN", lambda config: nn.Identity())

    config = type(
        "Config",
        (),
        {"hidden_size": 8, "layernorm_epsilon": 1e-6, "sequence_parallel": False},
    )()
    WanLayerWithAdaLN(config=config, submodules=WanWithAdaLNSubmodules(), vp_stage=1)

    assert captured["vp_stage"] == 1
