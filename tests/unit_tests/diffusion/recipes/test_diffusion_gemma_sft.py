import pytest

from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.diffusion.recipes.diffusion_gemma.sft import diffusion_gemma_sft_config


pytestmark = pytest.mark.unit


def test_offline_recipe_uses_native_checkpoint_and_unpacked_sft():
    cfg = diffusion_gemma_sft_config(
        dataset_root="dataset",
        tokenizer_model="tokenizer",
        pretrained_checkpoint="native",
        experiment_dir="output",
    )
    assert isinstance(cfg.dataset, GPTSFTDatasetConfig)
    assert cfg.dataset.dataset_root == "dataset"
    assert not cfg.dataset.enable_offline_packing
    assert not cfg.dataset.enable_in_batch_packing
    assert not cfg.dataset.enable_global_batch_packing
    assert cfg.checkpoint.pretrained_checkpoint == "native"
    assert cfg.checkpoint.save_rng_state_per_dp_rank
    assert not cfg.model.calculate_per_token_loss
    assert cfg.model.freeze_router
    assert cfg.peft is None
    assert cfg.tokenizer.tokenizer_model == "tokenizer"
    assert cfg.checkpoint.save == "output/checkpoints"


def test_random_init_requires_explicit_fixture_opt_in():
    with pytest.raises(ValueError, match="native.*checkpoint"):
        diffusion_gemma_sft_config(
            dataset_root="dataset",
            tokenizer_model="tokenizer",
            pretrained_checkpoint=None,
            experiment_dir="output",
        )
    cfg = diffusion_gemma_sft_config(
        dataset_root="dataset",
        tokenizer_model="tokenizer",
        pretrained_checkpoint=None,
        experiment_dir="output",
        allow_random_init=True,
    )
    assert cfg.checkpoint.pretrained_checkpoint is None
