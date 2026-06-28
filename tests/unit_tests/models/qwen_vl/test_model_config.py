import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer import TransformerConfig

from megatron.bridge.models.qwen3_asr.model_config import Qwen3ASRModelConfig
from megatron.bridge.models.qwen_audio.model_config import Qwen2AudioModelConfig
from megatron.bridge.models.qwen_omni.model_config import Qwen3OmniModelConfig, Qwen25OmniModelConfig
from megatron.bridge.models.qwen_vl.model_config import (
    Qwen3VLModelConfig,
    Qwen25VLModelBuilder,
    Qwen25VLModelConfig,
    Qwen35VLModelConfig,
)
from megatron.bridge.models.qwen_vl.qwen25_vl_bridge import Qwen25VLBridge


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_class",
    [
        Qwen25VLModelConfig,
        Qwen3VLModelConfig,
        Qwen35VLModelConfig,
        Qwen25OmniModelConfig,
        Qwen3OmniModelConfig,
        Qwen2AudioModelConfig,
        Qwen3ASRModelConfig,
    ],
)
def test_qwen_multimodal_model_configs_roundtrip(config_class):
    transformer = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2)
    config = config_class(transformer=transformer, vocab_size=32)

    restored = config_class.from_dict(config.as_dict())

    assert type(restored.transformer) is TransformerConfig
    assert restored.get_builder_cls() is config.get_builder_cls()
    assert restored.as_dict() == config.as_dict()


@pytest.mark.unit
def test_qwen25_vl_bridge_builds_exact_mcore_transformer_config():
    text_config = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_bias=True,
        tie_word_embeddings=True,
        rope_theta=1_000_000,
        hidden_act="silu",
        torch_dtype=torch.float32,
    )
    vision_config = SimpleNamespace(to_dict=lambda: {"hidden_size": 16})
    hf_config = SimpleNamespace(
        text_config=text_config,
        vision_config=vision_config,
        tie_word_embeddings=False,
    )

    config = Qwen25VLBridge().model_config_bridge(SimpleNamespace(config=hf_config))

    assert isinstance(config, Qwen25VLModelConfig)
    assert type(config.transformer) is TransformerConfig
    assert config.share_embeddings_and_output_weights is False
    assert config.vision_config == {"hidden_size": 16}


@pytest.mark.unit
def test_qwen_multimodal_model_config_modules_do_not_import_providers():
    model_modules = [
        "qwen_vl/model_config.py",
        "qwen_vl/modelling_qwen3_vl/__init__.py",
        "qwen_omni/model_config.py",
        "qwen_audio/model_config.py",
        "qwen3_asr/model_config.py",
    ]
    models_root = Path(__file__).parents[4] / "src/megatron/bridge/models"

    for relative_path in model_modules:
        tree = ast.parse((models_root / relative_path).read_text())
        imported_modules = [
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        assert all("provider" not in module for module in imported_modules)


@pytest.mark.unit
def test_qwen25_builder_binds_mrope_before_language_model_construction(monkeypatch):
    transformer = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2)
    config = Qwen25VLModelConfig(
        transformer=transformer,
        vocab_size=32,
        vision_config={"hidden_size": 16},
        mrope_section=[8, 4, 4],
    )
    transformer_state = dict(transformer.__dict__)
    serialized_config = config.as_dict()
    language_model = SimpleNamespace()

    def fake_build_language_model(self, pg_collection, pre_process, post_process, vp_stage):
        assert self._model_config.transformer.mrope_section == [8, 4, 4]
        return language_model

    built_model = SimpleNamespace()
    monkeypatch.setattr("megatron.training.models.gpt.GPTModelBuilder.build_model", fake_build_language_model)
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.model_config.Qwen25VLModel", lambda *args, **kwargs: built_model
    )
    monkeypatch.setattr("megatron.bridge.models.qwen_vl.model_config.Qwen2_5_VLVisionConfig", lambda **kwargs: kwargs)

    result = Qwen25VLModelBuilder(config).build_model(SimpleNamespace(pp=None), pre_process=True, post_process=True)

    assert result is built_model
    assert type(config.transformer) is TransformerConfig
    assert config.transformer.__dict__ == transformer_state
    assert config.as_dict() == serialized_config


@pytest.mark.unit
def test_qwen_builders_never_mutate_serialized_nested_transformer():
    models_root = Path(__file__).parents[4] / "src/megatron/bridge/models"
    modules = [
        "qwen_vl/model_config.py",
        "qwen_omni/model_config.py",
        "qwen_audio/model_config.py",
        "qwen3_asr/model_config.py",
    ]

    for relative_path in modules:
        tree = ast.parse((models_root / relative_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    rendered = ast.unparse(target)
                    assert not rendered.startswith("config.transformer.")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr":
                assert ast.unparse(node.args[0]) != "config.transformer"
