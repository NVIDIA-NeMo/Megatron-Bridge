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

from unittest.mock import patch

import pytest
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider


class TestMambaModelProviderCompatibility:
    def test_mamba_provider_is_hybrid_provider_wrapper(self):
        provider = MambaModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
        )

        assert isinstance(provider, HybridModelProvider)

    def test_mamba_stack_spec_maps_to_hybrid_model_kwarg(self):
        module_spec = ModuleSpec(module=object)
        provider = MambaModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            mamba_stack_spec=module_spec,
        )
        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch("megatron.bridge.models.hybrid.hybrid_provider.MCoreHybridModel") as mock_model:
            provider.provide(pre_process=True, post_process=True)

        assert provider.hybrid_stack_spec is module_spec
        assert provider.mamba_stack_spec is None
        assert mock_model.call_args.kwargs["hybrid_stack_spec"] is module_spec

    def test_rejects_hybrid_and_mamba_stack_spec_together(self):
        module_spec = ModuleSpec(module=object)

        with pytest.raises(ValueError, match="Cannot specify both hybrid_stack_spec and mamba_stack_spec"):
            MambaModelProvider(
                num_layers=2,
                hidden_size=128,
                num_attention_heads=1,
                hybrid_stack_spec=module_spec,
                mamba_stack_spec=module_spec,
            )

    def test_serialized_mamba_provider_uses_hybrid_stack_spec(self):
        module_spec = ModuleSpec(module=object)
        provider = MambaModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            mamba_stack_spec=module_spec,
        )

        data = provider.to_cfg_dict()

        hybrid_stack_spec = data["hybrid_stack_spec"]
        assert hybrid_stack_spec["_target_"] == "megatron.core.transformer.spec_utils.ModuleSpec"
        assert hybrid_stack_spec["module"] is object
        assert hybrid_stack_spec["params"] == {}
        assert hybrid_stack_spec["submodules"] is None
        assert hybrid_stack_spec["metainfo"] == {}
        assert "mamba_stack_spec" not in data
