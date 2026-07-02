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

import inspect

import torch

from megatron.bridge.diffusion.models.flux.flux_model import Flux
from megatron.bridge.diffusion.models.wan.wan_model import WanModel, _build_wan_layer_spec, sinusoidal_embedding_1d


def test_sinusoidal_embedding_1d_shape_and_dtype():
    dim = 16
    pos = torch.arange(10, dtype=torch.float32)
    emb = sinusoidal_embedding_1d(dim, pos)
    assert emb.shape == (pos.shape[0], dim)
    assert emb.dtype == torch.float32


def test_diffusion_model_constructor_positional_prefix_is_compatible():
    config = object()

    flux_bound = inspect.signature(Flux).bind(config, False, False, True, False)
    wan_bound = inspect.signature(WanModel).bind(config, False, False, True, False)

    assert flux_bound.arguments["pre_process"] is False
    assert flux_bound.arguments["post_process"] is False
    assert wan_bound.arguments["pre_process"] is False
    assert wan_bound.arguments["post_process"] is False


def test_wan_custom_layer_spec_factory_remains_zero_argument():
    sentinel = object()
    calls = []

    def custom_factory():
        calls.append(True)
        return sentinel

    assert _build_wan_layer_spec(custom_factory, layernorm_across_heads=True) is sentinel
    assert calls == [True]
