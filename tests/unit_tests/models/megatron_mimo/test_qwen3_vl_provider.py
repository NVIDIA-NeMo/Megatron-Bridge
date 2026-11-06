# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for Qwen3-VL Dense and MoE standard-provider MegatronMIMO hooks."""

import pytest
import torch
import yaml
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.megatron_mimo.conversion.mimo_model_io import _clear_derived_spec_fields
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.mimo import Qwen3VLMIMOModel, Qwen3VLMIMOVisionModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider, Qwen3VLMoEModelProvider
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.instantiate_utils import instantiate
from megatron.bridge.utils.yaml_utils import safe_yaml_representers


pytestmark = pytest.mark.unit


def _make_language_provider(provider_cls=Qwen3VLModelProvider, **overrides):
    """Build a small Qwen3-VL provider for unit-test shape checks."""
    kwargs = dict(
        num_layers=4, hidden_size=128, num_attention_heads=4, num_query_groups=4, kv_channels=32, vocab_size=256
    )
    if provider_cls is Qwen3VLMoEModelProvider:
        kwargs.update(num_moe_experts=8, moe_router_topk=2, moe_ffn_hidden_size=64)
    kwargs.update(overrides)
    return provider_cls(**kwargs)


def _make_parallelism_config() -> MegatronMIMOParallelismConfig:
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(tensor_model_parallel_size=1),
            "images": ModuleParallelismConfig(tensor_model_parallel_size=1),
        }
    )


@pytest.fixture(params=[Qwen3VLModelProvider, Qwen3VLMoEModelProvider], ids=["dense", "moe"])
def provider_cls(request):
    return request.param


class TestQwen3VLProviderMIMOAPI:
    """MIMO-facing API exposed by the standard Qwen provider."""

    def test_modality_metadata(self, provider_cls):
        provider = _make_language_provider(provider_cls)

        assert provider.modality_keys == {"images": "qwen_visual"}
        assert provider.special_token_ids == {"images": provider.image_token_id}

    def test_language_spec_uses_qwen_attention(self, provider_cls):
        provider = _make_language_provider(provider_cls)

        spec = provider.build_language_spec()

        assert spec.submodules.self_attention.module is Qwen3VLSelfAttention

    def test_language_model_spec_shape(self, provider_cls, monkeypatch):
        provider = _make_language_provider(provider_cls)
        sentinel_block_spec = ModuleSpec(module=object)
        monkeypatch.setattr(provider, "build_language_spec", lambda vp_stage=None, pp_rank=None: sentinel_block_spec)

        spec = provider.build_language_model_spec()

        assert isinstance(spec, ModuleSpec)
        assert spec.module is Qwen3VLGPTModel

        params = spec.params
        assert params is not None
        assert params["config"] is provider
        assert params["transformer_layer_spec"] is sentinel_block_spec
        assert params["vocab_size"] == provider.vocab_size
        assert params["max_sequence_length"] == provider.language_max_sequence_length
        assert params["logit_dtype"] is None
        assert params["position_embedding_type"] == "mrope"
        assert params["rotary_percent"] == provider.rotary_percent
        assert params["rotary_base"] == provider.rotary_base
        assert params["mtp_block_spec"] is None
        assert params["parallel_output"] is True
        assert params["scatter_embedding_sequence_parallel"] is False
        assert "pg_collection" not in params
        assert "pre_process" not in params
        assert "post_process" not in params

    def test_language_model_spec_propagates_logit_dtype(self, provider_cls, monkeypatch):
        provider = _make_language_provider(provider_cls, logit_dtype=torch.float32)
        monkeypatch.setattr(
            provider, "build_language_spec", lambda vp_stage=None, pp_rank=None: ModuleSpec(module=object)
        )

        spec = provider.build_language_model_spec()

        assert spec.params["logit_dtype"] is torch.float32

    @pytest.mark.parametrize("first_layers", [2, 3])
    def test_deepstack_pipeline_first_chunk_capacity(self, provider_cls, monkeypatch, first_layers):
        provider = _make_language_provider(provider_cls)
        provider.num_layers = 8
        provider.pipeline_model_parallel_size = 2
        provider.num_layers_in_first_pipeline_stage = first_layers
        monkeypatch.setattr(provider, "build_language_spec", lambda **kwargs: ModuleSpec(module=object))

        if first_layers == 3:
            assert provider.build_language_model_spec().module is Qwen3VLGPTModel
        else:
            with pytest.raises(ValueError, match="at least 3 decoder layers.*got 2"):
                provider.build_language_model_spec()

    @pytest.mark.parametrize("layout_type", ["list", "string", "parsed"])
    @pytest.mark.parametrize("first_layers", [2, 3])
    @pytest.mark.parametrize("virtual", [False, True])
    def test_deepstack_explicit_pipeline_layout(self, provider_cls, monkeypatch, layout_type, first_layers, virtual):
        layout = [["embedding"] + ["decoder"] * first_layers, ["decoder"] * 3]
        if virtual:
            layout.extend([["decoder"], ["loss"]])
        else:
            layout[-1].append("loss")
        if layout_type == "string":
            layout = f"Et*{first_layers}|t*3" + ("|t|L" if virtual else "L")
        elif layout_type == "parsed":
            layout = PipelineParallelLayerLayout(layout, pipeline_model_parallel_size=2)
        provider = _make_language_provider(
            provider_cls,
            num_layers=first_layers + 3 + int(virtual),
            pipeline_model_parallel_size=2,
            pipeline_model_parallel_layout=layout,
        )
        monkeypatch.setattr(provider, "build_language_spec", lambda **kwargs: ModuleSpec(module=object))

        if first_layers == 3:
            assert provider.build_language_model_spec().module is Qwen3VLGPTModel
        else:
            with pytest.raises(ValueError, match="at least 3 decoder layers.*got 2"):
                provider.build_language_model_spec()
        assert provider.pipeline_model_parallel_layout is layout

    @pytest.mark.parametrize(
        ("vocab_size", "should_pad_vocab", "expected_vocab_size"),
        [(257, True, 512), (257, False, 257), (512, True, 512)],
    )
    def test_language_model_spec_vocabulary_padding(
        self, provider_cls, monkeypatch, vocab_size, should_pad_vocab, expected_vocab_size
    ):
        provider = _make_language_provider(
            provider_cls,
            vocab_size=vocab_size,
            should_pad_vocab=should_pad_vocab,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=2,
        )
        monkeypatch.setattr(provider, "build_language_spec", lambda **kwargs: ModuleSpec(module=object))

        spec = provider.build_language_model_spec()

        assert spec.params["vocab_size"] == expected_vocab_size
        assert provider.vocab_size == vocab_size


class TestMegatronMIMOProviderFactory:
    def test_from_standard_provider_builds_base_provider(self, provider_cls, monkeypatch):
        language_provider = _make_language_provider(provider_cls)
        sentinel_block_spec = ModuleSpec(module=object)
        monkeypatch.setattr(
            language_provider,
            "build_language_spec",
            lambda vp_stage=None, pp_rank=None: sentinel_block_spec,
        )
        parallelism_config = _make_parallelism_config()

        provider = MegatronMIMOProvider.from_standard_provider(
            standard_provider=language_provider,
            megatron_mimo_parallelism_config=parallelism_config,
        )

        assert type(provider) is MegatronMIMOProvider
        assert provider.standard_provider is language_provider
        assert provider.megatron_mimo_parallelism_config is parallelism_config
        assert provider.language_model_spec.params["config"] is language_provider
        assert list(provider.modality_submodules_spec.keys()) == ["images"]
        assert provider.special_token_ids == {"images": language_provider.image_token_id}
        assert not hasattr(provider, "language_provider")
        assert language_provider.mimo_model_class is Qwen3VLMIMOModel

        images_spec = provider.modality_submodules_spec["images"]
        assert isinstance(images_spec, ModuleSpec)
        assert images_spec.module is VisionModalitySubmodules
        assert images_spec.params == {}

        submodules = images_spec.submodules
        assert "encoders" in submodules
        assert "input_projections" in submodules
        assert submodules["input_projections"] == []

        encoders = submodules["encoders"]
        assert list(encoders.keys()) == ["qwen_visual"]
        qwen_visual_spec = encoders["qwen_visual"]
        assert isinstance(qwen_visual_spec, ModuleSpec)
        assert qwen_visual_spec.module is Qwen3VLMIMOVisionModel
        assert "pg_collection" not in qwen_visual_spec.params
        vision_config = qwen_visual_spec.params["transformer_config"]
        assert vision_config.deepstack_visual_indexes == language_provider.vision_config.deepstack_visual_indexes

    def test_from_standard_provider_forces_mtp_off(self, provider_cls, monkeypatch):
        language_provider = _make_language_provider(provider_cls)
        language_provider.mtp_num_layers = 4
        monkeypatch.setattr(
            language_provider,
            "build_language_spec",
            lambda vp_stage=None, pp_rank=None: ModuleSpec(module=object),
        )

        MegatronMIMOProvider.from_standard_provider(
            standard_provider=language_provider,
            megatron_mimo_parallelism_config=_make_parallelism_config(),
        )

        assert language_provider.mtp_num_layers is None

    def test_from_standard_provider_canonicalizes_zero_mtp_to_none(self, provider_cls, monkeypatch):
        language_provider = _make_language_provider(provider_cls)
        language_provider.mtp_num_layers = 0
        monkeypatch.setattr(
            language_provider,
            "build_language_spec",
            lambda vp_stage=None, pp_rank=None: ModuleSpec(module=object),
        )

        MegatronMIMOProvider.from_standard_provider(
            standard_provider=language_provider,
            megatron_mimo_parallelism_config=_make_parallelism_config(),
        )

        assert language_provider.mtp_num_layers is None

    def test_yaml_round_trip_rebuilds_specs_after_clear(self, provider_cls, tmp_path):
        language_provider = _make_language_provider(provider_cls)
        provider = MegatronMIMOProvider.from_standard_provider(
            standard_provider=language_provider,
            megatron_mimo_parallelism_config=_make_parallelism_config(),
        )

        _clear_derived_spec_fields(provider)
        model_dict = ConfigContainer._convert_value_to_dict(provider)
        assert model_dict["language_model_spec"] is None
        assert model_dict["modality_submodules_spec"] == {}
        assert model_dict["special_token_ids"] == {}
        assert "standard_provider" in model_dict

        yaml_path = tmp_path / "run_config.yaml"
        with safe_yaml_representers():
            yaml_path.write_text(yaml.safe_dump({"model": model_dict}))

        loaded_dict = yaml.safe_load(yaml_path.read_text())
        loaded_provider = instantiate(loaded_dict["model"])

        assert isinstance(loaded_provider, MegatronMIMOProvider)
        assert isinstance(loaded_provider.standard_provider, provider_cls)
        assert loaded_provider.language_model_spec is not None
        assert loaded_provider.modality_submodules_spec
        assert loaded_provider.special_token_ids == {"images": language_provider.image_token_id}
