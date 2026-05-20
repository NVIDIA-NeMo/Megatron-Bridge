# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for Qwen35VLMegatronMIMOProvider."""

import pytest
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.megatron_mimo.qwen35_vl_provider import Qwen35VLMegatronMIMOProvider
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import _TRANSFORMERS_HAS_QWEN3_5, Qwen35VLModelProvider


pytestmark = pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5, reason="transformers does not have qwen3_5 support")


def _make_language_provider(**overrides) -> Qwen35VLModelProvider:
    """Build a small Qwen35VLModelProvider for unit-test shape checks."""
    kwargs = dict(num_layers=64, hidden_size=5120, num_attention_heads=24, vocab_size=128)
    kwargs.update(overrides)
    provider = Qwen35VLModelProvider(**kwargs)
    # ``get_vision_model_config`` reads deepstack_visual_indexes on the HF
    # vision config. The bare ``Qwen3_5VisionConfig()`` constructor does not
    # provide it; real loaded HF configs do. Set it explicitly for the test.
    provider.vision_config.deepstack_visual_indexes = []
    return provider


class TestQwen35VLMegatronMIMOProviderInit:
    """Initialization, validation, and rejection paths."""

    def test_rejects_mtp_enabled(self):
        lp = _make_language_provider()
        lp.mtp_num_layers = 1
        with pytest.raises(ValueError, match="does not support MTP"):
            Qwen35VLMegatronMIMOProvider(language_provider=lp)

    def test_canonicalizes_zero_mtp_to_none(self):
        """mtp_num_layers=0 is accepted and canonicalized to None.

        ``build_mtp_spec`` short-circuits on falsy mtp_num_layers; canonicalizing
        to None keeps downstream comparisons (``is None``) accurate.
        """
        lp = _make_language_provider()
        lp.mtp_num_layers = 0
        provider = Qwen35VLMegatronMIMOProvider(language_provider=lp)
        assert provider.language_provider.mtp_num_layers is None


class TestQwen35VLMegatronMIMOProviderSpecs:
    """The language_model_spec and modality_submodules_spec produced by the provider."""

    def test_language_model_spec_not_built_eagerly(self):
        """The language spec must be deferred to provide() — its construction
        calls into the experimental-attention block spec helper which queries
        parallel_state and is unsafe to invoke without torch.distributed init.
        """
        lp = _make_language_provider()
        provider = Qwen35VLMegatronMIMOProvider(language_provider=lp)
        assert provider.language_model_spec is None

    def test_language_model_spec_shape(self, monkeypatch):
        """Inspect the spec built by ``_build_language_model_spec`` directly.

        ``Qwen35VLModelProvider.build_language_spec`` requires an initialized
        parallel_state; monkeypatch it on the language provider so the test
        can verify the WRAPPER logic without needing torch.distributed.
        """
        lp = _make_language_provider()
        provider = Qwen35VLMegatronMIMOProvider(language_provider=lp)

        sentinel_block_spec = ModuleSpec(module=object)
        monkeypatch.setattr(lp, "build_language_spec", lambda vp_stage=None, pp_rank=None: sentinel_block_spec)

        spec = provider._build_language_model_spec()
        assert isinstance(spec, ModuleSpec)
        assert spec.module is Qwen3VLGPTModel

        params = spec.params
        assert params is not None
        assert params["config"] is lp
        assert params["transformer_layer_spec"] is sentinel_block_spec
        assert params["vocab_size"] == lp.vocab_size
        assert params["max_sequence_length"] == lp.language_max_sequence_length
        assert params["position_embedding_type"] == "mrope"
        assert params["rotary_percent"] == lp.rotary_percent
        assert params["rotary_base"] == lp.rotary_base
        # MTP must be disabled — MIMO v1 doesn't support it.
        assert params["mtp_block_spec"] is None
        assert params["parallel_output"] is True
        assert params["scatter_embedding_sequence_parallel"] is False
        # pg_collection / pre_process / post_process are injected by
        # MegatronMIMOProvider — they must NOT be baked in here.
        assert "pg_collection" not in params
        assert "pre_process" not in params
        assert "post_process" not in params

    def test_modality_submodules_spec_shape(self):
        lp = _make_language_provider()
        provider = Qwen35VLMegatronMIMOProvider(language_provider=lp)

        modality_specs = provider.modality_submodules_spec
        assert list(modality_specs.keys()) == ["images"]

        images_spec = modality_specs["images"]
        assert isinstance(images_spec, ModuleSpec)
        assert images_spec.module is VisionModalitySubmodules
        assert images_spec.params == {}

        submodules = images_spec.submodules
        assert "encoders" in submodules
        assert "input_projections" in submodules
        # Qwen3.5-VL bundles the patch merger inside the vision model.
        assert submodules["input_projections"] == []

        encoders = submodules["encoders"]
        assert list(encoders.keys()) == ["qwen_visual"]
        qwen_visual_spec = encoders["qwen_visual"]
        assert isinstance(qwen_visual_spec, ModuleSpec)
        assert qwen_visual_spec.module is Qwen3VLVisionModel
        # The encoder spec was built by Qwen35VLModelProvider.build_vision_encoder_spec
        # and must omit pg_collection so MegatronMIMOProvider injects it per rank.
        assert "pg_collection" not in qwen_visual_spec.params
