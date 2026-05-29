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
import torch
import transformers.modeling_utils as modeling_utils

import megatron.bridge.models.conversion.transformers_compat  # noqa: F401


pytestmark = [pytest.mark.unit]


def test_transformers_tied_weight_keys_accept_nested_lists():
    model = torch.nn.Module()
    model._tied_weights_keys = [["lm_head.weight", ("model.embed_tokens.weight",)]]

    child = torch.nn.Module()
    child._tied_weights_keys = {"proj.weight": None}
    model.child = child

    assert modeling_utils._get_tied_weight_keys(model) == [
        "lm_head.weight",
        "model.embed_tokens.weight",
        "child.proj.weight",
    ]
