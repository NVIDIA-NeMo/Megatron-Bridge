# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functional smoke tests for Nemotron Nano v2 finetuning recipe configurations."""

import pytest

from megatron.bridge.recipes.nemotronh import (
    nemotron_nano_9b_v2_finetune_config,
)


def _finetune_wrapper_lora(**kwargs):
    """Wrapper to adapt Nemotron Nano v2 finetune_config to the test runner signature (with LoRA).

    The runner will pass (dir, name) among others; we forward
    everything to finetune_config and inject a dummy pretrained_checkpoint.
    """
    kwargs.setdefault("pretrained_checkpoint", "/tmp/fake_nemotron_nano_v2_ckpt")
    kwargs.setdefault("peft", "lora")  # Explicitly use LoRA
    # Set mock=True to use MockGPTDataset instead of HFDataset for faster testing
    # The finetune config doesn't support 'mock' directly, but we'll override dataset
    return nemotron_nano_9b_v2_finetune_config(**kwargs)


def _finetune_wrapper_full(**kwargs):
    """Wrapper to adapt Nemotron Nano v2 finetune_config to the test runner signature (full SFT, no LoRA).

    The runner will pass (dir, name) among others; we forward
    everything to finetune_config and inject a dummy pretrained_checkpoint.
    """
    kwargs.setdefault("pretrained_checkpoint", "/tmp/fake_nemotron_nano_v2_ckpt")
    kwargs.setdefault("peft", None)  # No PEFT for full finetuning
    return nemotron_nano_9b_v2_finetune_config(**kwargs)


NEMOTRON_NANO_V2_FINETUNE_RECIPES = [
    # Test LoRA finetuning
    (
        _finetune_wrapper_lora,
        "nemotron_nano_9b_v2_lora",
        {
            "num_layers": 3,
            "hybrid_override_pattern": "M*-",
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "sequence_parallel": False,
        },
    ),
    # Test full finetuning (no LoRA)
    (
        _finetune_wrapper_full,
        "nemotron_nano_9b_v2_full",
        {
            "num_layers": 3,
            "hybrid_override_pattern": "M*-",
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "sequence_parallel": False,
        },
    ),
]


class TestNemotronNanoV2FinetuneRecipes:
    """Test class for Nemotron Nano v2 finetune recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,model_overrides", NEMOTRON_NANO_V2_FINETUNE_RECIPES)
    def test_nemotron_nano_v2_finetune_recipes(self, config_func, recipe_name, model_overrides, tmp_path):
        """Functional test for Nemotron Nano v2 finetuning recipes with LoRA and full SFT."""
        # Create the config
        config = config_func(dir=str(tmp_path), name=recipe_name)

        # Override the dataset to use MockGPTDataset for faster testing
        from megatron.bridge.training.config import MockGPTDatasetConfig

        seq_length = 512
        config.dataset = MockGPTDatasetConfig(
            random_seed=5678,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            sequence_length=seq_length,
            num_dataset_builder_threads=1,
            data_sharding=True,
            dataloader_type="single",
            num_workers=0,
        )

        # Apply model overrides
        for attribute_name, attribute_value in model_overrides.items():
            setattr(config.model, attribute_name, attribute_value)

        # Override to use smaller model for faster testing
        config.model.seq_length = seq_length
        config.train.train_iters = 10
        config.train.eval_interval = 5
        config.train.eval_iters = 2
        config.train.micro_batch_size = 1
        config.train.global_batch_size = 8
        config.scheduler.lr_warmup_iters = 2

        # Calculate proper dataset splits
        train_samples_needed = config.train.train_iters * config.train.global_batch_size
        eval_samples_needed = config.train.eval_iters * config.train.global_batch_size
        test_samples_needed = 100
        total_samples = train_samples_needed + eval_samples_needed + test_samples_needed

        train_split = train_samples_needed / total_samples
        valid_split = eval_samples_needed / total_samples
        test_split = test_samples_needed / total_samples
        config.dataset.split = [train_split, valid_split, test_split]

        # Run the test using the actual finetuning function
        from megatron.bridge.training.finetune import finetune
        from megatron.bridge.training.gpt_step import forward_step
        from tests.functional_tests.utils import (
            clear_directories,
            initialize_distributed,
            verify_checkpoint_files,
        )

        initialize_distributed()
        try:
            # Run finetuning
            finetune(config, forward_step)

            # Verify checkpoints were saved
            verify_checkpoint_files(config.checkpoint.save, config.train.train_iters)

        finally:
            clear_directories(tmp_path)
