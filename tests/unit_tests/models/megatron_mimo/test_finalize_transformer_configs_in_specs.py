# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for ``_finalize_transformer_configs_in_specs``."""

from __future__ import annotations

from unittest.mock import Mock

from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import (
    _finalize_transformer_configs_in_specs,
)
from megatron.bridge.models.transformer_config import TransformerConfig


def _make_minimal_transformer_config() -> TransformerConfig:
    """Construct a TransformerConfig whose derived fields are not finalized."""
    return TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2)


class TestFinalizeTransformerConfigsInSpecs:
    def test_finalises_language_model_spec_config(self):
        cfg = _make_minimal_transformer_config()
        assert cfg.init_method is None, "precondition: deferred-finalize keeps init_method None"

        language_spec = ModuleSpec(module=Mock, params={"config": cfg})
        _finalize_transformer_configs_in_specs(language_spec, modality_specs={})

        assert cfg.init_method is not None, "finalise should populate init_method"

    def test_finalises_modality_encoder_transformer_config(self):
        cfg = _make_minimal_transformer_config()
        assert cfg.init_method is None

        encoder_spec = ModuleSpec(module=Mock, params={"transformer_config": cfg})
        modality_spec = ModuleSpec(
            module=Mock,
            params={},
            submodules={"encoders": {"vit": encoder_spec}},
        )
        _finalize_transformer_configs_in_specs(
            ModuleSpec(module=Mock, params={}),
            modality_specs={"images": modality_spec},
        )

        assert cfg.init_method is not None

    def test_finalises_input_projection_transformer_config(self):
        cfg = _make_minimal_transformer_config()
        assert cfg.init_method is None

        proj_spec = ModuleSpec(module=Mock, params={"config": cfg})
        modality_spec = ModuleSpec(
            module=Mock,
            params={},
            submodules={"encoders": {}, "input_projections": [proj_spec]},
        )
        _finalize_transformer_configs_in_specs(
            ModuleSpec(module=Mock, params={}),
            modality_specs={"images": modality_spec},
        )

        assert cfg.init_method is not None

    def test_idempotent(self):
        """Calling twice on the same provider must not raise. finalize() is
        documented as safe to call multiple times.
        """
        cfg = _make_minimal_transformer_config()
        language_spec = ModuleSpec(module=Mock, params={"config": cfg})

        _finalize_transformer_configs_in_specs(language_spec, modality_specs={})
        first_init_method = cfg.init_method
        _finalize_transformer_configs_in_specs(language_spec, modality_specs={})
        # init_method stays callable after the second pass.
        assert cfg.init_method is not None
        assert callable(cfg.init_method)
        assert callable(first_init_method)
