import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer import TransformerConfig

from megatron.bridge.models.qwen3_asr.model_config import Qwen3ASRModelBuilder, Qwen3ASRModelConfig
from megatron.bridge.models.qwen_audio.model_config import Qwen2AudioModelBuilder, Qwen2AudioModelConfig
from megatron.bridge.models.qwen_omni.model_config import (
    Qwen3OmniModelBuilder,
    Qwen3OmniModelConfig,
    Qwen25OmniModelBuilder,
    Qwen25OmniModelConfig,
)
from megatron.bridge.models.qwen_vl.model_config import (
    Qwen3VLModelBuilder,
    Qwen3VLModelConfig,
    Qwen25VLModelBuilder,
    Qwen25VLModelConfig,
    Qwen35VLModelBuilder,
    Qwen35VLModelConfig,
    build_qwen35_mimo_modality_specs,
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
@pytest.mark.parametrize(
    ("config_class", "payload"),
    [
        (Qwen25VLModelConfig, {"vision_config": {"hidden_size": 16, "depths": [1, 2]}}),
        (Qwen3VLModelConfig, {"vision_config": {"hidden_size": 16, "deepstack_visual_indexes": [1]}}),
        (Qwen35VLModelConfig, {"vision_config": {"hidden_size": 16, "out_hidden_size": 32}}),
        (Qwen25OmniModelConfig, {"thinker_config": {"audio_config": {"d_model": 8}}}),
        (Qwen3OmniModelConfig, {"thinker_config": {"vision_config": {"depth": 2}}}),
        (Qwen2AudioModelConfig, {"hf_config": {"audio_config": {"d_model": 8}}}),
        (Qwen3ASRModelConfig, {"thinker_config": {"audio_config": {"encoder_layers": 2}}}),
    ],
)
def test_qwen_multimodal_nested_payloads_roundtrip(config_class, payload):
    config = config_class(
        transformer=TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2),
        vocab_size=32,
        **payload,
    )

    restored = config_class.from_dict(config.as_dict())

    for field_name, expected in payload.items():
        assert getattr(restored, field_name) == expected
    assert type(restored.transformer) is TransformerConfig


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

    restored = Qwen25VLModelConfig.from_dict(config.as_dict())
    assert type(restored.transformer) is TransformerConfig
    assert restored.vision_config == {"hidden_size": 16}


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
@pytest.mark.parametrize(
    ("builder_class", "config_class", "module_name", "model_name"),
    [
        (Qwen3VLModelBuilder, Qwen3VLModelConfig, "qwen_vl", "Qwen3VLModel"),
        (Qwen35VLModelBuilder, Qwen35VLModelConfig, "qwen_vl", "Qwen3VLModel"),
        (Qwen25OmniModelBuilder, Qwen25OmniModelConfig, "qwen_omni", "Qwen25OmniModel"),
        (Qwen3OmniModelBuilder, Qwen3OmniModelConfig, "qwen_omni", "Qwen3OmniModel"),
        (Qwen3ASRModelBuilder, Qwen3ASRModelConfig, "qwen3_asr", "Qwen3ASRModel"),
    ],
)
def test_qwen_custom_builders_pass_outer_config_without_phantom_transformer_fields(
    monkeypatch, builder_class, config_class, module_name, model_name
):
    transformer = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2)
    config = config_class(transformer=transformer, vocab_size=32, mrope_section=[8, 4, 4])
    transformer_state = dict(transformer.__dict__)
    captured = {}

    def capture_model(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    module = f"megatron.bridge.models.{module_name}.model_config"
    monkeypatch.setattr(f"{module}.{model_name}", capture_model)
    if module_name == "qwen_vl":
        monkeypatch.setattr(f"{module}._vision_config_from_dict", lambda *args: SimpleNamespace())
        monkeypatch.setattr(f"{module}.get_gpt_layer_with_transformer_engine_spec", lambda **kwargs: SimpleNamespace())
        monkeypatch.setattr(
            f"{module}.get_transformer_block_with_experimental_attention_variant_spec",
            lambda *args, **kwargs: SimpleNamespace(),
        )
        monkeypatch.setattr(f"{module}.mtp_block_spec", lambda *args, **kwargs: None)
    elif module_name == "qwen_omni":
        monkeypatch.setattr(f"{module}.get_gpt_layer_with_transformer_engine_spec", lambda **kwargs: SimpleNamespace())
        monkeypatch.setattr(builder_class, "model_cls", staticmethod(capture_model))
        monkeypatch.setattr(builder_class, "thinker_cls", staticmethod(lambda **kwargs: SimpleNamespace()))
    else:
        monkeypatch.setattr(
            f"{module}.get_gpt_layer_with_transformer_engine_spec", lambda *args, **kwargs: SimpleNamespace()
        )
        monkeypatch.setattr(f"{module}.Qwen3ASRThinkerConfig", lambda **kwargs: SimpleNamespace())

    result = builder_class(config).build_model(SimpleNamespace(pp=None), pre_process=True, post_process=True)

    assert isinstance(result, SimpleNamespace)
    assert captured["model_config"] is config
    runtime_transformer = captured["language_transformer_config"]
    assert type(runtime_transformer) is TransformerConfig
    assert runtime_transformer.mrope_section == [8, 4, 4]
    assert set(runtime_transformer.__dict__) == set(transformer_state)
    assert config.transformer.__dict__ == transformer_state


@pytest.mark.unit
def test_qwen2_audio_builder_passes_outer_config_without_transformer_copy(monkeypatch):
    transformer = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2)
    config = Qwen2AudioModelConfig(transformer=transformer, vocab_size=32, audio_token_id=77, pad_token_id=0)
    language_model = SimpleNamespace(pre_process=True, post_process=True)
    captured = {}

    monkeypatch.setattr(
        "megatron.training.models.gpt.GPTModelBuilder.build_model", lambda *args, **kwargs: language_model
    )
    monkeypatch.setattr("megatron.bridge.models.qwen_audio.model_config.Qwen2AudioConfig", lambda **kwargs: kwargs)

    def capture_model(model_config, *args, **kwargs):
        captured["model_config"] = model_config
        return SimpleNamespace()

    monkeypatch.setattr("megatron.bridge.models.qwen_audio.model_config.Qwen2AudioModel", capture_model)

    Qwen2AudioModelBuilder(config).build_model(SimpleNamespace(), pre_process=True, post_process=True)

    assert captured["model_config"] is config
    assert type(config.transformer) is TransformerConfig
    assert "audio_token_id" not in config.transformer.__dict__
    assert "pad_token_id" not in config.transformer.__dict__


@pytest.mark.unit
def test_qwen35_mimo_modality_spec_uses_exact_mcore_transformer(monkeypatch):
    vision_config = SimpleNamespace(
        depth=2,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        patch_size=2,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
        num_position_embeddings=16,
        out_hidden_size=16,
        deepstack_visual_indexes=[],
    )
    config = Qwen35VLModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2),
        vocab_size=32,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.model_config._vision_config_from_dict",
        lambda *args: vision_config,
    )

    specs = build_qwen35_mimo_modality_specs(config)
    encoder_spec = specs["images"].submodules["encoders"]["qwen_visual"]
    transformer = encoder_spec.params["transformer_config"]

    assert type(transformer) is TransformerConfig
    assert encoder_spec.params["vision_config"] is vision_config
    for family_field in (
        "patch_size",
        "temporal_patch_size",
        "in_channels",
        "spatial_merge_size",
        "num_position_embeddings",
        "out_hidden_size",
        "deepstack_visual_indexes",
        "apply_rotary_pos_emb_in_fp32",
    ):
        assert family_field not in transformer.__dict__


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
                    assert not rendered.startswith("transformer.")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr":
                assert "transformer" not in ast.unparse(node.args[0])
