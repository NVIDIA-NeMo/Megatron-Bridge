# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
from megatron.core.transformer import TransformerConfig

from megatron.bridge.diffusion.models.nemotron_labs_diffusion.model_config import (
    NemotronLabsDiffusionModelConfig,
)
from megatron.bridge.diffusion.recipes.nemotron_labs_diffusion import ar_to_dlm


pytestmark = pytest.mark.unit


def _model_config() -> NemotronLabsDiffusionModelConfig:
    return NemotronLabsDiffusionModelConfig(
        transformer=TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=2),
        vocab_size=32,
        block_size=23,
        hf_config={"model_type": "test-diffusion"},
    )


@pytest.mark.parametrize(
    ("recipe", "tensor_parallel_size"),
    [
        (ar_to_dlm.nemotron_labs_diffusion_3b_pretrain_config, 1),
        (ar_to_dlm.nemotron_labs_diffusion_8b_pretrain_config, 4),
        (ar_to_dlm.nemotron_labs_diffusion_14b_pretrain_config, 8),
    ],
)
def test_size_recipe_uses_copied_model_config_without_loading_hf(monkeypatch, recipe, tensor_parallel_size):
    source = _model_config()
    monkeypatch.setattr(
        ar_to_dlm.PreTrainedCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("custom model_config must not load Hugging Face"),
    )

    config = recipe(model_config=source)

    assert config.model is not source
    assert isinstance(config.model, NemotronLabsDiffusionModelConfig)
    assert config.model.block_size == 23
    assert config.model.hf_config == {"model_type": "test-diffusion"}
    assert config.model.tensor_model_parallel_size == tensor_parallel_size
    assert source.tensor_model_parallel_size == 1
    assert source.seq_length == 1024


@pytest.mark.parametrize(
    "recipe",
    [
        ar_to_dlm.nemotron_labs_diffusion_3b_pretrain_config,
        ar_to_dlm.nemotron_labs_diffusion_8b_pretrain_config,
        ar_to_dlm.nemotron_labs_diffusion_14b_pretrain_config,
    ],
)
def test_size_recipe_rejects_explicit_hf_path_with_model_config(recipe):
    with pytest.raises(ValueError, match="mutually exclusive"):
        recipe(model_config=_model_config(), hf_path="org/model")
