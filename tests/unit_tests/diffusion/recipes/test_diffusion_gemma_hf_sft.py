from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.diffusion.models.diffusion_gemma.provider import DiffusionGemmaModelProvider
from megatron.bridge.diffusion.recipes.diffusion_gemma.sft import diffusion_gemma_sft_config


pytestmark = pytest.mark.unit


def test_hf_source_reuses_autobridge_loading_hook():
    bridge = MagicMock()
    bridge.to_megatron_provider.return_value = DiffusionGemmaModelProvider()
    with patch("megatron.bridge.diffusion.recipes.diffusion_gemma.sft.AutoBridge", return_value=bridge) as auto:
        auto.from_hf_pretrained.return_value = bridge
        cfg = diffusion_gemma_sft_config(
            dataset_root="data",
            tokenizer_model="tokenizer",
            pretrained_checkpoint=None,
            experiment_dir="output",
            hf_model="checkpoint",
        )
    auto.from_hf_pretrained.assert_called_once_with("checkpoint")
    bridge.to_megatron_provider.assert_called_once_with(load_weights=True)
    assert cfg.model is bridge.to_megatron_provider.return_value
    assert cfg.checkpoint.pretrained_checkpoint is None
    assert cfg.checkpoint.load == "output/checkpoints"


@pytest.mark.parametrize(
    "options",
    [
        {"pretrained_checkpoint": "native"},
        {"model": DiffusionGemmaModelProvider()},
        {"allow_random_init": True},
    ],
)
def test_hf_source_rejects_conflicting_initialization(options):
    arguments = dict(
        dataset_root="data", tokenizer_model="tok", pretrained_checkpoint=None, experiment_dir="out", hf_model="hf"
    )
    arguments.update(options)
    with pytest.raises(ValueError, match="HF|hf_model"):
        diffusion_gemma_sft_config(**arguments)
