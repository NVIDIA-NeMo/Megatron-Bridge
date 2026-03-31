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

import unittest.mock as mock

import torch

from megatron.bridge.models.conversion.model_bridge import (
    MegatronModelBridge,
    WeightConversionTask,
    megatron_param_matches_skip_globs,
)


def test_megatron_param_matches_skip_globs():
    assert not megatron_param_matches_skip_globs("decoder.layers.0.weight", "layers.0.weight", None)
    assert not megatron_param_matches_skip_globs("decoder.layers.0.weight", "layers.0.weight", [])
    assert megatron_param_matches_skip_globs(
        "vision_model.decoder.layers.0.weight", "layers.0.weight", ["*vision_model*"]
    )
    assert megatron_param_matches_skip_globs(
        "language_model.decoder.layers.0.weight", "vision_model.foo.weight", ["*vision_model*"]
    )
    assert not megatron_param_matches_skip_globs(
        "language_model.decoder.layers.0.weight", "layers.0.weight", ["*vision_model*"]
    )


def test_load_weights_hf_to_megatron_respects_skip_globs():
    """Skipped tasks must not copy HF weights into Megatron parameters."""

    class _DummyMapping:
        def __init__(self, param_weight: torch.Tensor, mname: str):
            self.hf_param = "hf.weight"
            self.megatron_param = mname
            self._param_weight = param_weight

        def hf_to_megatron(self, hf_weights, megatron_module):
            del hf_weights, megatron_module
            return torch.ones_like(self._param_weight)

    vision_param = torch.zeros(2, 3)
    lm_param = torch.zeros(2, 3)

    tasks = [
        WeightConversionTask(
            param_name="vision_model.weight",
            global_param_name="vision_model.weight",
            mapping=_DummyMapping(vision_param, "vision_model.weight"),
            megatron_module=mock.Mock(),
            param_weight=vision_param,
        ),
        WeightConversionTask(
            param_name="language_model.weight",
            global_param_name="language_model.weight",
            mapping=_DummyMapping(lm_param, "language_model.weight"),
            megatron_module=mock.Mock(),
            param_weight=lm_param,
        ),
    ]

    hf_pretrained = mock.Mock()
    hf_pretrained.state = {"hf.weight": torch.ones(2, 3)}
    hf_pretrained.model_name_or_path = "dummy"

    bridge = MegatronModelBridge.__new__(MegatronModelBridge)
    bridge.maybe_modify_loaded_hf_weight = lambda _hf_key, hf_sd: hf_sd["hf.weight"]  # type: ignore[method-assign]

    megatron_stages = [mock.MagicMock()]

    with (
        mock.patch.object(MegatronModelBridge, "build_conversion_tasks", return_value=tasks),
        mock.patch.object(MegatronModelBridge, "_with_progress_tracking", lambda self, ts, _desc: ts),
        mock.patch.object(MegatronModelBridge, "_broadcast_shared_embeddings", mock.Mock()),
    ):
        MegatronModelBridge.load_weights_hf_to_megatron(
            bridge,
            hf_pretrained,
            megatron_stages,
            skip_megatron_param_globs=["*vision_model*"],
        )

    assert torch.all(vision_param == 0)
    assert torch.all(lm_param == 1)
