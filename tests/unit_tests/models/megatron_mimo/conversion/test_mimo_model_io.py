# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compatibility tests for MegatronMIMO checkpoint model metadata."""

import inspect
from types import SimpleNamespace

import pytest

from megatron.bridge.models.megatron_mimo.conversion.mimo_model_io import (
    _checkpoint_serializable_model_config,
    save_megatron_mimo_model,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider


@pytest.mark.unit
def test_save_api_preserves_provider_keyword_name() -> None:
    """Keep released keyword calls compatible while accepting pure configs."""
    signature = inspect.signature(save_megatron_mimo_model)

    assert "provider" in signature.parameters
    assert "model_config" not in signature.parameters


@pytest.mark.unit
def test_legacy_provider_runtime_fields_are_restored_after_serialization() -> None:
    """Legacy providers omit derived specs and grids only inside the save scope."""
    language_spec = object()
    modality_specs = {"images": object()}
    special_token_ids = {"images": 7}
    grids = {"language": object()}
    provider = MegatronMIMOProvider(
        language_model_spec=language_spec,
        modality_submodules_spec=modality_specs,
        special_token_ids=special_token_ids,
    )
    provider._grids = grids

    with _checkpoint_serializable_model_config(provider):
        assert provider.language_model_spec is None
        assert provider.modality_submodules_spec == {}
        assert provider.special_token_ids == {}
        assert provider._grids is None

    assert provider.language_model_spec is language_spec
    assert provider.modality_submodules_spec is modality_specs
    assert provider.special_token_ids is special_token_ids
    assert provider._grids is grids


@pytest.mark.unit
def test_pure_model_config_is_not_mutated_for_checkpoint_serialization() -> None:
    """Pure configs already contain only serialized inputs."""
    model_config = SimpleNamespace(marker=object())

    with _checkpoint_serializable_model_config(model_config):
        assert model_config.marker is not None

    assert model_config.marker is not None
