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

from megatron.bridge.diffusion.models.wan.wan_layer_spec import get_wan_block_with_transformer_engine_spec


def test_get_wan_block_with_transformer_engine_spec_basic():
    spec = get_wan_block_with_transformer_engine_spec()
    # Basic structure checks
    assert hasattr(spec, "module")
    assert hasattr(spec, "submodules")
    sub = spec.submodules
    # Expected submodule fields exist
    for name in ["norm1", "norm2", "norm3", "full_self_attention", "cross_attention", "mlp"]:
        assert hasattr(sub, name), f"Missing submodule {name}"
