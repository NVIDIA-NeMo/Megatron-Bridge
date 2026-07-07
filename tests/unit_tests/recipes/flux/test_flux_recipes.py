# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Contract tests for builder-backed FLUX H100 recipes."""

from types import SimpleNamespace

import pytest

from megatron.bridge.diffusion.models.flux.model_config import FluxModelConfig
from megatron.bridge.recipes.flux.h100 import flux as flux_recipe


pytestmark = pytest.mark.unit


class _FakePreTrainedFlux:
    """Config-only FLUX wrapper that avoids Hub access."""

    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.config = SimpleNamespace(
            num_attention_heads=24,
            attention_head_dim=128,
            in_channels=64,
            patch_size=1,
            num_layers=19,
            num_single_layers=38,
            joint_attention_dim=4096,
            pooled_projection_dim=768,
            guidance_embeds=True,
            axes_dims_rope=[16, 56, 56],
            ffn_dim=12288,
        )


def test_flux_h100_recipe_uses_standalone_model_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The official FLUX recipe must not fall back to its compatibility provider."""
    monkeypatch.setattr(flux_recipe, "PreTrainedFlux", _FakePreTrainedFlux)

    config = flux_recipe.flux_12b_pretrain_2gpu_h100_bf16_config()

    assert isinstance(config.model, FluxModelConfig)
    assert config.model.get_builder_cls().__name__ == "FluxModelBuilder"
    assert config.model.tensor_model_parallel_size == 2
    assert config.model.pipeline_model_parallel_size == 1
    assert config.model.num_joint_layers == 19
    assert config.dataset.context_dim == 4096
