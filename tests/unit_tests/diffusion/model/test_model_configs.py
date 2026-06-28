import ast
from pathlib import Path

import pytest
from megatron.core.transformer import TransformerConfig

from megatron.bridge.diffusion.models.flux.model_config import FluxModelBuilder, FluxModelConfig
from megatron.bridge.diffusion.models.llada15.model_config import LLaDA15ModelConfig
from megatron.bridge.diffusion.models.nemotron_labs_diffusion.model_config import NemotronLabsDiffusionModelConfig
from megatron.bridge.diffusion.models.wan.model_config import WanModelBuilder, WanModelConfig


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_class",
    [FluxModelConfig, WanModelConfig, LLaDA15ModelConfig, NemotronLabsDiffusionModelConfig],
)
def test_diffusion_model_configs_roundtrip_with_exact_mcore_transformer(config_class):
    transformer = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2)
    config = config_class(transformer=transformer, vocab_size=32)

    restored = config_class.from_dict(config.as_dict())

    assert type(restored.transformer) is TransformerConfig
    assert restored.get_builder_cls() is config.get_builder_cls()
    assert restored.as_dict() == config.as_dict()


@pytest.mark.unit
def test_diffusion_model_config_modules_are_provider_neutral():
    repository_root = Path(__file__).parents[4]
    modules = [
        "src/megatron/bridge/diffusion/models/flux/model_config.py",
        "src/megatron/bridge/diffusion/models/wan/model_config.py",
        "src/megatron/bridge/diffusion/models/llada15/model_config.py",
        "src/megatron/bridge/diffusion/models/nemotron_labs_diffusion/model_config.py",
        "src/megatron/bridge/diffusion/models/flux/__init__.py",
        "src/megatron/bridge/diffusion/models/llada15/__init__.py",
        "src/megatron/bridge/diffusion/models/nemotron_labs_diffusion/__init__.py",
    ]

    for module in modules:
        tree = ast.parse((repository_root / module).read_text())
        top_level_imports = [
            node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        assert all("provider" not in node.module for node in top_level_imports)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config_class", "builder_class", "target"),
    [
        (FluxModelConfig, FluxModelBuilder, "megatron.bridge.diffusion.models.flux.model_config.Flux"),
        (WanModelConfig, WanModelBuilder, "megatron.bridge.diffusion.models.wan.model_config.WanModel"),
    ],
)
def test_custom_diffusion_builders_pass_exact_transformer_and_outer_config(
    monkeypatch, config_class, builder_class, target
):
    transformer = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2)
    config = config_class(transformer=transformer, vocab_size=32)
    captured = {}

    def fake_model(transformer_config=None, *args, **kwargs):
        captured["transformer"] = kwargs.pop("config", transformer_config)
        captured["architecture_config"] = kwargs["architecture_config"]
        return object()

    monkeypatch.setattr(target, fake_model)
    builder_class(config).build_model(object(), pre_process=True, post_process=True)

    assert type(captured["transformer"]) is TransformerConfig
    assert captured["architecture_config"] is config
