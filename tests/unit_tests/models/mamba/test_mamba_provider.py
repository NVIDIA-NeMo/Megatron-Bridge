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
