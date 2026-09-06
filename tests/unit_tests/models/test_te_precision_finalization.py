# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
import torch
from megatron.core.quantization.quant_config import GlobMatcher, RecipeConfig

from megatron.bridge.models.model_provider import _finalize_model_quantization


class _RecordingLinear(torch.nn.Module):
    def __init__(self, recipe: RecipeConfig) -> None:
        super().__init__()
        self.config = SimpleNamespace(quant_recipe=recipe)
        self.match: tuple[str | None, str | None] | None = None

    def finish_init(self, quantization_config) -> None:
        if quantization_config is None:
            self.match = (None, None)
        else:
            self.match = (
                quantization_config.config_key,
                quantization_config.match_input.module_path,
            )


class _Layer(torch.nn.Module):
    def __init__(self, recipe: RecipeConfig) -> None:
        super().__init__()
        self.self_attention = torch.nn.ModuleDict(
            {
                "q_proj": _RecordingLinear(recipe),
                "o_proj": _RecordingLinear(recipe),
            }
        )
        self.mlp = torch.nn.ModuleDict(
            {
                "linear_fc1": _RecordingLinear(recipe),
                "linear_fc2": _RecordingLinear(recipe),
                "router": _RecordingLinear(recipe),
                "shared_expert": _RecordingLinear(recipe),
            }
        )


class _ReplacementModel(torch.nn.Module):
    def __init__(self, recipe: RecipeConfig, mount_path: str) -> None:
        super().__init__()
        stack = torch.nn.ModuleList([_Layer(recipe) for _ in range(7)])
        if mount_path == "decoder":
            self.decoder = torch.nn.Module()
            self.decoder.layers = stack
        else:
            self.container = torch.nn.Module()
            self.container.backbone = torch.nn.Module()
            self.container.backbone.layers = stack


def _precision_recipe(first_bf16: int, last_bf16: int, qkvo_mxfp8: bool) -> RecipeConfig:
    matchers = []
    for layer_idx in list(range(first_bf16)) + list(range(7 - last_bf16, 7)):
        matchers.append(GlobMatcher(pattern=f"*layers.{layer_idx}.*", config_key="bf16"))
    if qkvo_mxfp8:
        matchers.extend(
            [
                GlobMatcher(pattern="*self_attention.q_proj", config_key="mxfp8"),
                GlobMatcher(pattern="*self_attention.o_proj", config_key="mxfp8"),
            ]
        )
    matchers.extend(
        [
            GlobMatcher(pattern="*mlp.linear_fc1", config_key="mxfp8"),
            GlobMatcher(pattern="*mlp.linear_fc2", config_key="mxfp8"),
            GlobMatcher(pattern="*", config_key="bf16"),
        ]
    )
    return RecipeConfig(
        matchers=matchers,
        config_dict={"bf16": {"precision": "bf16"}, "mxfp8": {"precision": "mxfp8"}},
    )


@pytest.mark.parametrize(
    ("first_bf16", "last_bf16", "qkvo_mxfp8", "expected"),
    [
        (0, 0, False, "EEEEEEE"),
        (2, 3, False, "BBEEBBB"),
        (1, 2, True, "BQQQQBB"),
        (4, 0, True, "BBBBQQQ"),
        (0, 5, False, "EEBBBBB"),
    ],
)
@pytest.mark.parametrize("mount_path", ["decoder", "container.backbone"])
def test_finalizes_replacement_subtree_with_complete_module_paths(
    first_bf16: int,
    last_bf16: int,
    qkvo_mxfp8: bool,
    expected: str,
    mount_path: str,
) -> None:
    recipe = _precision_recipe(first_bf16, last_bf16, qkvo_mxfp8)
    model = _ReplacementModel(recipe, mount_path)

    _finalize_model_quantization(model)

    layers = model.decoder.layers if mount_path == "decoder" else model.container.backbone.layers
    signature = ""
    for layer in layers:
        q_proj = layer.self_attention["q_proj"]
        fc1 = layer.mlp["linear_fc1"]
        router = layer.mlp["router"]
        shared_expert = layer.mlp["shared_expert"]
        assert router.match[0] == "bf16"
        assert shared_expert.match[0] == "bf16"
        assert q_proj.match[1].endswith("self_attention.q_proj")
        assert fc1.match[1].endswith("mlp.linear_fc1")
        if fc1.match[0] == "bf16":
            signature += "B"
        elif q_proj.match[0] == "mxfp8":
            signature += "Q"
        else:
            signature += "E"

    assert signature == expected
