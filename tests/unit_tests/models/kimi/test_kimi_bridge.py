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

from megatron.bridge.models.kimi.kimi_bridge import KimiK2Bridge


@pytest.fixture
def kimi_registry():
    """Mapping registry of a bare Kimi K2 bridge."""
    return KimiK2Bridge().mapping_registry()


class TestKimiK2BridgeMappingRegistry:
    """Mapping-registry checks for the Kimi K2 bridge."""

    def test_expert_bias_mapping_present(self, kimi_registry):
        """Kimi K2 enables moe_router_enable_expert_bias, so the router expert_bias
        must map to the HF gate.e_score_correction_bias key.

        Regression test for the mapping being absent from the DeepSeek common
        mapping list (https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/4199):
        without it, expert_bias stayed at its zero init after HF conversion and
        top-k routing diverged from the HF checkpoint.
        """
        mapping = kimi_registry.megatron_to_hf_lookup("decoder.layers.3.mlp.router.expert_bias")
        assert mapping is not None
        assert mapping.hf_param == "model.layers.3.mlp.gate.e_score_correction_bias"

    def test_expert_bias_reverse_lookup(self, kimi_registry):
        """The HF-side key must resolve back to the Megatron router expert_bias."""
        mapping = kimi_registry.hf_to_megatron_lookup("model.layers.7.mlp.gate.e_score_correction_bias")
        assert mapping is not None
        assert mapping.megatron_param == "decoder.layers.7.mlp.router.expert_bias"
