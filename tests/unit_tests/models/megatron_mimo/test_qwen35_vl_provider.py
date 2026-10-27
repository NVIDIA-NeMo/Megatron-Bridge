# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for Qwen3.5-VL standard-provider MegatronMIMO hooks."""

import pytest
import torch
import yaml
from megatron.core import parallel_state
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.megatron_mimo.conversion.mimo_model_io import _clear_derived_spec_fields
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    _TRANSFORMERS_HAS_QWEN3_5,
    Qwen35VLModelProvider,
    Qwen35VLMoEModelProvider,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.instantiate_utils import instantiate
from megatron.bridge.utils.yaml_utils import safe_yaml_representers


pytestmark = pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5, reason="transformers does not have qwen3_5 support")


def _make_language_provider(**overrides) -> Qwen35VLModelProvider:
    """Build a small Qwen35VLModelProvider for unit-test shape checks."""
    kwargs = dict(num_layers=64, hidden_size=5120, num_attention_heads=24, vocab_size=128)
    kwargs.update(overrides)
    provider = Qwen35VLModelProvider(**kwargs)
    provider.vision_config.deepstack_visual_indexes = []
    return provider


def _make_parallelism_config() -> MegatronMIMOParallelismConfig:
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(tensor_model_parallel_size=1),
            "images": ModuleParallelismConfig(tensor_model_parallel_size=1),
        }
    )


class TestQwen35VLProviderMIMOAPI:
    """MIMO-facing API exposed by the standard Qwen provider."""

    def test_modality_metadata(self):
        provider = _make_language_provider()

        assert provider.modality_keys == {"images": "qwen_visual"}
        assert provider.special_token_ids == {"images": provider.image_token_id}

    def test_language_model_spec_shape(self, monkeypatch):
        provider = _make_language_provider()
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

    def test_language_model_spec_propagates_logit_dtype(self, monkeypatch):
        provider = _make_language_provider(logit_dtype=torch.float32)
        monkeypatch.setattr(
            provider, "build_language_spec", lambda vp_stage=None, pp_rank=None: ModuleSpec(module=object)
        )

        spec = provider.build_language_model_spec()

        assert spec.params["logit_dtype"] is torch.float32

    @pytest.mark.parametrize("provider_cls", [Qwen35VLModelProvider, Qwen35VLMoEModelProvider])
    def test_advertises_mimo_mtp_support(self, provider_cls):
        assert provider_cls.mimo_supports_mtp is True

    def test_language_model_spec_builds_mtp_for_requested_stage(self, monkeypatch):
        provider = _make_language_provider(mtp_num_layers=1)
        monkeypatch.setattr(
            provider, "build_language_spec", lambda vp_stage=None, pp_rank=None: ModuleSpec(module=object)
        )
        mtp_spec_calls = []

        def mtp_block_spec(config, vp_stage=None, pp_rank=None):
            mtp_spec_calls.append((config, vp_stage, pp_rank))
            return None

        monkeypatch.setattr("megatron.bridge.models.gpt_provider.mtp_block_spec", mtp_block_spec)

        spec = provider.build_language_model_spec(pp_rank=3)

        assert mtp_spec_calls == [(provider, None, 3)]
        assert spec.params["mtp_block_spec"] is None


class TestQwen35VLMIMOMTPSpec:
    """The MIMO language spec builds the MTP block from an explicit pipeline rank."""

    @staticmethod
    def _tiny_provider(**overrides) -> Qwen35VLModelProvider:
        kwargs = dict(
            num_layers=4,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=16,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            mtp_num_layers=1,
        )
        kwargs.update(overrides)
        return _make_language_provider(**kwargs)

    @pytest.mark.parametrize("pp_size", [1, 2])
    def test_mtp_block_only_on_last_stage_without_mpu(self, monkeypatch, pp_size):
        def fail(*args, **kwargs):
            raise AssertionError("MIMO builds language specs before the MPU globals are set")

        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", fail)
        provider = self._tiny_provider(pipeline_model_parallel_size=pp_size)

        for pp_rank in range(pp_size):
            mtp_spec = provider.build_language_model_spec(pp_rank=pp_rank).params["mtp_block_spec"]
            if pp_rank < pp_size - 1:
                assert mtp_spec is None
                continue
            assert len(mtp_spec.layer_specs) == 1
            mtp_layer = mtp_spec.layer_specs[0].submodules.mtp_model_layer
            assert mtp_layer.submodules.self_attention.module is Qwen3VLSelfAttention

    def test_no_mtp_block_without_mtp_layers(self):
        provider = self._tiny_provider(mtp_num_layers=None)

        assert provider.build_language_model_spec(pp_rank=0).params["mtp_block_spec"] is None


class TestMegatronMIMOProviderFactory:
    def test_from_standard_provider_builds_base_provider(self, monkeypatch):
        language_provider = _make_language_provider()
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
        assert qwen_visual_spec.module is Qwen3VLVisionModel
        assert "pg_collection" not in qwen_visual_spec.params

    def test_from_standard_provider_keeps_mtp(self, monkeypatch):
        language_provider = _make_language_provider()
        language_provider.mtp_num_layers = 4
        monkeypatch.setattr(
            language_provider,
            "build_language_spec",
            lambda vp_stage=None, pp_rank=None: ModuleSpec(module=object),
        )
        mtp_spec = ModuleSpec(module=object)
        mtp_spec_ranks = []

        def build_mtp_spec(vp_stage=None, pp_rank=None):
            mtp_spec_ranks.append(pp_rank)
            return mtp_spec

        monkeypatch.setattr(language_provider, "build_mtp_spec", build_mtp_spec)

        provider = MegatronMIMOProvider.from_standard_provider(
            standard_provider=language_provider,
            megatron_mimo_parallelism_config=_make_parallelism_config(),
        )

        assert language_provider.mtp_num_layers == 4
        assert provider.language_model_spec.params["mtp_block_spec"] is mtp_spec
        assert mtp_spec_ranks == [0]

    def test_from_standard_provider_canonicalizes_zero_mtp_to_none(self, monkeypatch):
        language_provider = _make_language_provider()
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

    def test_from_standard_provider_default_modality_builder(self):
        class DummyProvider:
            modality_keys = {"images": "dummy_visual"}
            special_token_ids = {"images": 7}

            def build_language_model_spec(self):
                return ModuleSpec(module=object)

            def build_vision_encoder_spec(self):
                return ModuleSpec(module=object)

        provider = MegatronMIMOProvider.from_standard_provider(
            standard_provider=DummyProvider(),
            megatron_mimo_parallelism_config=_make_parallelism_config(),
        )

        images_spec = provider.modality_submodules_spec["images"]
        assert images_spec.module is VisionModalitySubmodules
        assert list(images_spec.submodules["encoders"].keys()) == ["dummy_visual"]

    def test_yaml_round_trip_rebuilds_specs_after_clear(self, tmp_path):
        language_provider = _make_language_provider()
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
        assert isinstance(loaded_provider.standard_provider, Qwen35VLModelProvider)
        assert loaded_provider.language_model_spec is not None
        assert loaded_provider.modality_submodules_spec
        assert loaded_provider.special_token_ids == {"images": language_provider.image_token_id}

    def test_from_standard_provider_requires_language_model_spec_api(self):
        with pytest.raises(TypeError, match="build_language_model_spec"):
            MegatronMIMOProvider.from_standard_provider(
                standard_provider=object(),
                megatron_mimo_parallelism_config=_make_parallelism_config(),
            )
