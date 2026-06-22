from unittest.mock import Mock, patch

import pytest
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.common import ModelConfig
from megatron.bridge.models.hybrid.hybrid_builder import HybridModelBuilder, HybridModelConfig
from megatron.bridge.models.mamba.mamba_builder import MambaModelBuilder, MambaModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig


def _make_transformer(**kwargs):
    defaults = dict(num_layers=2, hidden_size=128, num_attention_heads=1)
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


class TestMambaModelBuilderCompatibility:
    def test_mamba_config_is_hybrid_config_wrapper(self):
        config = MambaModelConfig(transformer=_make_transformer(), vocab_size=32000)

        assert isinstance(config, HybridModelConfig)
        assert config.builder == "megatron.bridge.models.mamba.MambaModelBuilder"
        assert config.get_builder_cls() is MambaModelBuilder

    def test_mamba_builder_is_hybrid_builder_wrapper(self):
        config = MambaModelConfig(transformer=_make_transformer(), vocab_size=32000)
        builder = MambaModelBuilder(config)

        assert isinstance(builder, HybridModelBuilder)

    @patch("megatron.bridge.models.hybrid.hybrid_builder.MCoreHybridModel")
    def test_mamba_stack_spec_maps_to_hybrid_model_kwarg(self, mock_model):
        module_spec = ModuleSpec(module=object)
        config = MambaModelConfig(
            transformer=_make_transformer(),
            vocab_size=32000,
            mamba_stack_spec=module_spec,
        )
        builder = MambaModelBuilder(config)
        pg = Mock()
        pg.pp = Mock()

        builder.build_model(pg, pre_process=True, post_process=True)

        assert mock_model.call_args.kwargs["hybrid_stack_spec"] is module_spec

    def test_rejects_hybrid_and_mamba_stack_spec_together(self):
        module_spec = ModuleSpec(module=object)

        with pytest.raises(ValueError, match="Cannot specify both hybrid_stack_spec and mamba_stack_spec"):
            MambaModelConfig(
                transformer=_make_transformer(),
                hybrid_stack_spec=module_spec,
                mamba_stack_spec=module_spec,
            )

    def test_old_serialized_targets_resolve(self):
        config = MambaModelConfig(transformer=_make_transformer(), vocab_size=32000)
        data = config.to_cfg_dict()
        data["_target_"] = "megatron.bridge.models.mamba.MambaModelConfig"
        data["_builder_"] = "megatron.bridge.models.mamba.MambaModelBuilder"

        restored = ModelConfig.from_dict(data)

        assert isinstance(restored, MambaModelConfig)
        assert restored.get_builder_cls() is MambaModelBuilder

    def test_old_serialized_module_targets_resolve(self):
        config = MambaModelConfig(transformer=_make_transformer(), vocab_size=32000)
        data = config.to_cfg_dict()
        data["_target_"] = "megatron.bridge.models.mamba.mamba_builder.MambaModelConfig"
        data["_builder_"] = "megatron.bridge.models.mamba.mamba_builder.MambaModelBuilder"

        restored = ModelConfig.from_dict(data)

        assert isinstance(restored, MambaModelConfig)
        assert restored.get_builder_cls() is MambaModelBuilder
