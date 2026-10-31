# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the builder-backed GLM-5.3-Flash config and builder."""

import dataclasses
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

from megatron.bridge.models.glm5next import glm53flash_builder
from megatron.bridge.models.glm5next.glm53flash_bridge import (
    GLM53FlashBridge,
    GLM53FlashModelProvider,
    glm5next_hybrid_stack_spec,
)
from megatron.bridge.models.glm5next.glm53flash_builder import GLM53FlashModelBuilder
from megatron.bridge.models.glm5next.glm53flash_config import GLM53FlashModelConfig, GLM53FlashTransformerConfig
from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig


pytestmark = pytest.mark.unit

_BUILDER_MODULE = "megatron.bridge.models.glm5next.glm53flash_builder"


def _hf_config(num_nextn_predict_layers: int = 0):
    config_module = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    return config_module.Glm5NextConfig(
        text_config={
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "indexed_attention", "linear_attention", "indexed_attention"],
            "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
            "indexer_types": ["full"] * 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "q_lora_rank": 64,
            "kv_lora_rank": 64,
            "qk_head_dim": 64,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 0,
            "v_head_dim": 64,
            "index_head_dim": 64,
            "index_n_heads": 4,
            "index_topk": 32,
            "index_kpool": 4,
            "linear_attn_config": {
                "num_heads": 4,
                "head_dim": 64,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
            "n_shared_experts": 1,
            "vocab_size": 2560,
            "pad_token_id": 0,
            "scoring_func": "sigmoid",
            "num_nextn_predict_layers": num_nextn_predict_layers,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 2,
            "out_hidden_size": 256,
        },
        image_token_id=17,
        video_start_token_id=18,
    )


def _model_config(**hf_kwargs) -> GLM53FlashModelConfig:
    return GLM53FlashBridge().hf_config_to_model_config(_hf_config(**hf_kwargs))


def _provider(**hf_kwargs) -> GLM53FlashModelProvider:
    return GLM53FlashBridge().provider_bridge(SimpleNamespace(config=_hf_config(**hf_kwargs)))


class TestGLM53FlashModelConfig:
    def test_bridge_builds_and_converts_through_the_model_config(self):
        assert GLM53FlashBridge.MODEL_CONFIG_CLASS is GLM53FlashModelConfig
        assert GLM53FlashBridge.USE_MODEL_CONFIG_FOR_CONVERSION is True
        assert "provider_bridge" not in GLM53FlashBridge.__dict__

    def test_config_types(self):
        model_config = _model_config()
        assert isinstance(model_config, BridgeHybridModelConfig)
        assert type(model_config.transformer) is GLM53FlashTransformerConfig
        assert model_config.get_builder_cls() is GLM53FlashModelBuilder
        assert model_config.hybrid_stack_spec is glm5next_hybrid_stack_spec

    def test_vision_language_fields(self):
        hf_config = _hf_config()
        model_config = GLM53FlashBridge().hf_config_to_model_config(hf_config)
        transformer = model_config.transformer
        assert transformer.vision_config is hf_config.vision_config
        assert transformer.spatial_merge_size == hf_config.vision_config.spatial_merge_size
        assert transformer.image_token_id == 17
        assert transformer.video_start_token_id == 18
        assert transformer.return_dict is True
        assert not model_config.freeze_language_model
        assert not model_config.freeze_vision_model
        assert not model_config.freeze_vision_projection

    def test_language_layout(self):
        model_config = _model_config()
        assert model_config.hybrid_layer_pattern == "K-DEKEDE"
        assert model_config.num_layers == 8
        assert model_config.moe_layer_freq == [0, 0, 0, 1, 0, 1, 0, 1]
        assert model_config.position_embedding_type == "none"
        assert model_config.qk_pos_emb_head_dim == 0
        assert model_config.kda_two_stage_gates and model_config.kda_safe_gate
        assert model_config.make_vocab_size_divisible_by == 1280

    @pytest.mark.parametrize("num_mtp", [0, 1])
    def test_builder_config_matches_legacy_provider(self, num_mtp):
        """Every field the provider and the builder config share has the same value."""
        provider = _provider(num_nextn_predict_layers=num_mtp)
        model_config = _model_config(num_nextn_predict_layers=num_mtp)

        assert type(provider) is GLM53FlashModelProvider
        mismatches = {}
        for owner in (model_config, model_config.transformer):
            for field in dataclasses.fields(owner):
                if field.name == "transformer" or not hasattr(provider, field.name):
                    continue
                expected, actual = getattr(provider, field.name), getattr(owner, field.name)
                # The Hugging Face vision config is the same object, not comparable by value.
                if actual is not expected and actual != expected:
                    mismatches[field.name] = (expected, actual)
        assert mismatches == {}
        # The builder always builds the embedding without SP scatter, the provider's default.
        assert provider.scatter_embedding_sequence_parallel is False

    def test_export_config_matches_legacy_provider(self):
        exported = GLM53FlashBridge.megatron_to_hf_config(_model_config())
        assert exported == GLM53FlashBridge.megatron_to_hf_config(_provider())
        assert exported["num_hidden_layers"] == 4

    def test_provider_kwargs_extend_the_builder_kwargs(self):
        hf_config = _hf_config()
        bridge = GLM53FlashBridge()
        model_kwargs = bridge.hf_config_to_model_config_kwargs(hf_config)
        provider_kwargs = bridge.hf_config_to_provider_kwargs(hf_config)
        assert provider_kwargs == {**model_kwargs, "keep_mtp_spec_in_bf16": True}

    def test_flat_overrides_route_to_their_owner(self):
        model_config = _model_config()
        model_config.freeze_vision_model = True
        model_config.image_token_id = 7
        model_config.tensor_model_parallel_size = 2

        assert "freeze_vision_model" not in vars(model_config.transformer)
        assert model_config.freeze_vision_model is True
        assert model_config.transformer.image_token_id == 7
        assert model_config.transformer.tensor_model_parallel_size == 2
        with pytest.raises(AttributeError):
            model_config.vision_tokens = 1

    @pytest.mark.parametrize(("pp_size", "vp_size", "segments"), [(1, None, 1), (2, None, 2), (2, 2, 4)])
    def test_finalize_adds_dsa_pipeline_separators(self, pp_size, vp_size, segments):
        model_config = _model_config()
        # Four DSA blocks give three full-indexer boundaries.
        model_config.hybrid_layer_pattern = "D-DEDEDE"
        model_config.pipeline_model_parallel_size = pp_size
        model_config.virtual_pipeline_model_parallel_size = vp_size
        with patch.object(BridgeHybridModelConfig, "finalize"):
            model_config.finalize()
        assert model_config.hybrid_layer_pattern.count(Symbols.PIPE) + 1 == segments

    def test_checkpoint_run_config_reload(self, tmp_path):
        """A config saved to run_config.yaml reloads as the same config and exports identically."""
        from megatron.bridge.recipes.gpt import vanilla_gpt_pretrain_config
        from megatron.bridge.training.model_load_save import load_model_config

        model_config = _model_config(num_nextn_predict_layers=1)
        model_config.freeze_vision_model = True
        run_config = vanilla_gpt_pretrain_config()
        run_config.model = model_config
        run_config.to_yaml(str(tmp_path / "run_config.yaml"))

        loaded, _ = load_model_config(str(tmp_path))

        assert type(loaded) is GLM53FlashModelConfig
        assert type(loaded.transformer) is GLM53FlashTransformerConfig
        assert loaded.get_builder_cls() is GLM53FlashModelBuilder
        assert loaded.hybrid_stack_spec is glm5next_hybrid_stack_spec
        assert loaded.freeze_vision_model is True
        assert loaded.mtp_num_layers == 1
        # The Hugging Face vision config is saved as provider checkpoints save it.
        assert type(loaded.vision_config) is type(model_config.vision_config)
        assert loaded.vision_config.hidden_size == model_config.vision_config.hidden_size
        assert GLM53FlashBridge.megatron_to_hf_config(loaded) == GLM53FlashBridge.megatron_to_hf_config(model_config)
        # Serialization leaves the live config untouched.
        assert model_config.vision_config is not None


class _FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _pg_collection(pp_rank: int = 0, pp_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(pp=_FakeGroup(pp_rank, pp_size))


@pytest.fixture
def model_class(monkeypatch):
    """Replace GLM53FlashModel in the builder, recording how the builder constructs it."""
    model_class = Mock(return_value=Mock())
    monkeypatch.setattr(f"{_BUILDER_MODULE}.GLM53FlashModel", model_class)
    return model_class


class TestGLM53FlashModelBuilder:
    def test_build_model_constructs_the_vision_language_model(self, model_class):
        model_config = _model_config()
        pg_collection = _pg_collection()
        builder = GLM53FlashModelBuilder(model_config)

        model = builder.build_model(pg_collection, pre_process=True, post_process=False, vp_stage=None)

        assert model is model_class.return_value
        args, kwargs = model_class.call_args
        assert args == (model_config.transformer,)
        assert kwargs["pre_process"] is True and kwargs["post_process"] is False
        assert kwargs["pg_collection"] is pg_collection
        build_language_model = kwargs["build_language_model"]
        assert isinstance(build_language_model, partial)
        assert build_language_model.func == builder._build_language_model
        assert build_language_model.args == (pg_collection,)
        model.freeze.assert_not_called()

    @pytest.mark.parametrize(
        ("pp_rank", "vp_stage", "vp_size", "expected"),
        [
            (0, None, None, (True, False)),
            (1, None, None, (False, True)),
            (0, 1, 2, (False, False)),
            (1, 1, 2, (False, True)),
        ],
    )
    def test_build_model_derives_stage_flags(self, model_class, monkeypatch, pp_rank, vp_stage, vp_size, expected):
        # Megatron-Core reads stage ranks from the process group only once distributed is initialized.
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        model_config = _model_config()
        model_config.virtual_pipeline_model_parallel_size = vp_size

        GLM53FlashModelBuilder(model_config).build_model(_pg_collection(pp_rank, 2), vp_stage=vp_stage)

        kwargs = model_class.call_args.kwargs
        assert (kwargs["pre_process"], kwargs["post_process"]) == expected
        assert kwargs["vp_stage"] == vp_stage

    def test_build_model_freezes_requested_modules(self, model_class):
        model_config = _model_config()
        model_config.freeze_language_model = True
        model_config.freeze_vision_projection = True

        model = GLM53FlashModelBuilder(model_config).build_model(_pg_collection(), pre_process=True, post_process=True)

        model.freeze.assert_called_once_with(
            freeze_language_model=True, freeze_vision_model=False, freeze_vision_projection=True
        )

    def test_builder_rejects_other_configs(self, model_class):
        other = Mock(spec=BridgeHybridModelConfig)
        with pytest.raises(TypeError, match="GLM53FlashModelConfig"):
            GLM53FlashModelBuilder(other).build_model(_pg_collection())

    @pytest.mark.parametrize(("should_pad_vocab", "expected_vocab"), [(False, 1300), (True, 2560)])
    def test_language_model_embeds_without_sequence_parallel_scatter(self, should_pad_vocab, expected_vocab):
        model_config = _model_config()
        # Padded to a multiple of make_vocab_size_divisible_by (1280) x TP (2).
        model_config.vocab_size = 1300
        model_config.should_pad_vocab = should_pad_vocab
        model_config.tensor_model_parallel_size = 2
        builder = GLM53FlashModelBuilder(model_config)
        stack_spec = object()
        pg_collection = _pg_collection()

        with (
            patch.object(builder, "_resolve_hybrid_stack_spec", return_value=stack_spec),
            patch.object(glm53flash_builder, "HybridModel") as hybrid_model,
        ):
            builder._build_language_model(pg_collection, pre_process=True, post_process=False, vp_stage=None)

        kwargs = hybrid_model.call_args.kwargs
        assert kwargs["config"] is model_config.transformer
        assert kwargs["hybrid_stack_spec"] is stack_spec
        assert kwargs["scatter_embedding_sequence_parallel"] is False
        assert kwargs["vocab_size"] == expected_vocab
        assert kwargs["hybrid_layer_pattern"] == model_config.hybrid_layer_pattern
        assert kwargs["position_embedding_type"] == "none"
        assert kwargs["pre_process"] is True and kwargs["post_process"] is False
        assert kwargs["pg_collection"] is pg_collection
