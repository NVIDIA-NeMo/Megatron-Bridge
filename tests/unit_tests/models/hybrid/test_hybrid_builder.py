from unittest.mock import Mock, patch

import pytest
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.common import ModelConfig
from megatron.bridge.models.hybrid.hybrid_builder import (
    HybridModelBuilder,
    HybridModelConfig,
    get_default_hybrid_stack_spec,
    modelopt_hybrid_stack_spec,
    transformer_engine_hybrid_stack_spec,
)
from megatron.bridge.models.transformer_config import TransformerConfig


def _make_transformer(**kwargs):
    defaults = dict(num_layers=2, hidden_size=128, num_attention_heads=1)
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def _make_hybrid_config(**kwargs):
    defaults = dict(transformer=_make_transformer(), vocab_size=32000)
    defaults.update(kwargs)
    return HybridModelConfig(**defaults)


class TestHybridStackSpecs:
    def test_transformer_engine_spec_returns_default(self):
        with patch("megatron.bridge.models.hybrid.hybrid_builder.default_hybrid_stack_spec") as mock_spec:
            result = transformer_engine_hybrid_stack_spec()
        assert result is mock_spec

    def test_modelopt_spec_calls_modelopt_helper(self):
        mock_spec = Mock(spec=ModuleSpec)
        with patch(
            "megatron.bridge.models.hybrid.hybrid_builder.get_hybrid_stack_modelopt_spec",
            return_value=mock_spec,
        ) as mock_fn:
            result = modelopt_hybrid_stack_spec()
        mock_fn.assert_called_once_with(local_core_attention=False, remap_te_layernorm=False)
        assert result is mock_spec

    def test_default_spec_returns_te_spec(self):
        config = _make_hybrid_config()
        with patch(
            "megatron.bridge.models.hybrid.hybrid_builder.transformer_engine_hybrid_stack_spec",
            return_value=Mock(spec=ModuleSpec),
        ) as mock_fn:
            result = get_default_hybrid_stack_spec(config)
        mock_fn.assert_called_once()
        assert result is mock_fn.return_value

    def test_default_spec_returns_modelopt_spec(self):
        config = _make_hybrid_config()
        config.restore_modelopt_state = True
        with patch(
            "megatron.bridge.models.hybrid.hybrid_builder.modelopt_hybrid_stack_spec",
            return_value=Mock(spec=ModuleSpec),
        ) as mock_fn:
            result = get_default_hybrid_stack_spec(config)
        mock_fn.assert_called_once_with(config)
        assert result is mock_fn.return_value


class TestHybridModelConfig:
    def test_builder_classvar(self):
        assert HybridModelConfig.builder == "megatron.bridge.models.hybrid.HybridModelBuilder"

    def test_default_values(self):
        config = HybridModelConfig(transformer=_make_transformer())
        assert config.fp16_lm_cross_entropy is False
        assert config.parallel_output is True
        assert config.share_embeddings_and_output_weights is False
        assert config.hybrid_layer_pattern is None
        assert config.hybrid_stack_spec is None
        assert config.mamba_stack_spec is None
        assert config.seq_length == 8192
        assert config.position_embedding_type == "none"
        assert config.vocab_size is None

    def test_rejects_hybrid_and_mamba_stack_spec_together(self):
        module_spec = ModuleSpec(module=object)

        with pytest.raises(ValueError, match="Cannot specify both hybrid_stack_spec and mamba_stack_spec"):
            HybridModelConfig(
                transformer=_make_transformer(),
                hybrid_stack_spec=module_spec,
                mamba_stack_spec=module_spec,
            )

    def test_proxies_transformer_attribute(self):
        transformer = _make_transformer(hidden_size=256)
        config = HybridModelConfig(transformer=transformer, vocab_size=32000)

        assert config.hidden_size == 256
        config.hidden_size = 512
        assert transformer.hidden_size == 512

    def test_from_dict_with_hybrid_builder_target(self):
        config = HybridModelConfig(transformer=_make_transformer(), vocab_size=32000)
        data = config.to_cfg_dict()
        data["_target_"] = "megatron.bridge.models.hybrid.HybridModelConfig"
        data["_builder_"] = "megatron.bridge.models.hybrid.HybridModelBuilder"

        restored = ModelConfig.from_dict(data)

        assert isinstance(restored, HybridModelConfig)
        assert restored.get_builder_cls() is HybridModelBuilder


class TestHybridModelBuilder:
    def setup_method(self):
        self.config = _make_hybrid_config(vocab_size=32000)
        self.builder = HybridModelBuilder(self.config)
        self.pg = Mock()
        self.pg.pp = Mock()

    @patch("megatron.bridge.models.hybrid.hybrid_builder.MCoreHybridModel")
    def test_raises_when_vocab_size_none(self, mock_model):
        self.config.vocab_size = None
        with pytest.raises(AssertionError, match="vocab_size"):
            self.builder.build_model(self.pg)
        mock_model.assert_not_called()

    @patch("megatron.bridge.models.hybrid.hybrid_builder.MCoreHybridModel")
    def test_hybrid_stack_spec_used_directly(self, mock_model):
        module_spec = ModuleSpec(module=object)
        self.config.__dict__["hybrid_stack_spec"] = module_spec

        self.builder.build_model(self.pg, pre_process=True, post_process=True)

        assert mock_model.call_args.kwargs["hybrid_stack_spec"] is module_spec

    @patch("megatron.bridge.models.hybrid.hybrid_builder.MCoreHybridModel")
    def test_mamba_stack_spec_alias_used_directly(self, mock_model):
        module_spec = ModuleSpec(module=object)
        self.config.__dict__["mamba_stack_spec"] = module_spec

        self.builder.build_model(self.pg, pre_process=True, post_process=True)

        assert mock_model.call_args.kwargs["hybrid_stack_spec"] is module_spec

    @patch("megatron.bridge.models.hybrid.hybrid_builder.MCoreHybridModel")
    def test_stack_spec_callable_with_config_param(self, mock_model):
        returned_spec = ModuleSpec(module=object)
        received = []

        def one_param_fn(config):
            received.append(config)
            return returned_spec

        self.config.__dict__["hybrid_stack_spec"] = one_param_fn
        self.builder.build_model(self.pg, pre_process=True, post_process=True)

        assert received == [self.config]
        assert mock_model.call_args.kwargs["hybrid_stack_spec"] is returned_spec

    @patch("megatron.bridge.models.hybrid.hybrid_builder.calculate_padded_vocab_size", return_value=32128)
    @patch("megatron.bridge.models.hybrid.hybrid_builder.MCoreHybridModel")
    def test_vocab_padding_calls_calculate_padded_vocab_size(self, mock_model, mock_pad):
        self.config.__dict__["should_pad_vocab"] = True
        self.config.transformer.tensor_model_parallel_size = 2

        self.builder.build_model(self.pg, pre_process=True, post_process=True)

        mock_pad.assert_called_once_with(32000, 128, 2)
        assert mock_model.call_args.kwargs["vocab_size"] == 32128

    @patch("megatron.bridge.models.hybrid.hybrid_builder.is_pp_first_stage", return_value=True)
    @patch("megatron.bridge.models.hybrid.hybrid_builder.is_pp_last_stage", return_value=True)
    @patch("megatron.bridge.models.hybrid.hybrid_builder.MCoreHybridModel")
    def test_explicit_pre_post_process_passed_through(self, mock_model, *_):
        self.builder.build_model(self.pg, pre_process=False, post_process=True)

        assert mock_model.call_args.kwargs["pre_process"] is False
        assert mock_model.call_args.kwargs["post_process"] is True

    @patch("megatron.bridge.models.hybrid.hybrid_builder.MCoreHybridModel")
    def test_config_params_passed_to_mcore(self, mock_model):
        config = _make_hybrid_config(
            vocab_size=32000,
            seq_length=4096,
            hybrid_layer_pattern="M-A-",
            fp16_lm_cross_entropy=True,
            parallel_output=False,
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
        )
        builder = HybridModelBuilder(config)

        builder.build_model(self.pg, pre_process=True, post_process=True)

        kw = mock_model.call_args.kwargs
        assert kw["config"] is config.transformer
        assert kw["vocab_size"] == 32000
        assert kw["max_sequence_length"] == 4096
        assert kw["hybrid_layer_pattern"] == "M-A-"
        assert kw["fp16_lm_cross_entropy"] is True
        assert kw["parallel_output"] is False
        assert kw["share_embeddings_and_output_weights"] is True

    @patch("megatron.bridge.models.hybrid.hybrid_builder.compose_hooks")
    @patch("megatron.bridge.models.hybrid.hybrid_builder.unimodal_build_distributed_models")
    def test_build_distributed_models_delegates_to_unimodal(self, mock_unimodal, mock_compose):
        model_list = [Mock()]
        mock_unimodal.return_value = model_list
        mock_compose.return_value = Mock(return_value=None)

        result = self.builder.build_distributed_models(self.pg)

        assert result is model_list
        assert mock_unimodal.called
