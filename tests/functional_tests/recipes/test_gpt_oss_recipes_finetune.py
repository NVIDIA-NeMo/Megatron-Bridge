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

"""Functional smoke tests for GPT-OSS finetuning recipe configurations."""

import pytest
import torch
from transformers import AutoTokenizer, GptOssConfig, GptOssForCausalLM

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.recipes.gpt_oss.gpt_oss import (
    gpt_oss_20b_finetune_config,
)


# Overrides for toy model - create a tiny GPT-OSS MoE model for testing
# Keep architecture consistent but minimize size for fast CI
HF_GPT_OSS_TOY_MODEL_OVERRIDES = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "num_local_experts": 4,  # Minimal number of experts
    "num_experts_per_tok": 2,  # Top-k routing
    "max_position_embeddings": 2048,
    "vocab_size": 32000,
    "rope_theta": 150000,
}


class TestGPTOSSFinetuneRecipes:
    """Test class for GPT-OSS finetune recipe functional tests."""

    @pytest.fixture(scope="class")
    def gpt_oss_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace GPT-OSS toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("gpt_oss_toy_model")
        model_dir = temp_dir / "gpt_oss_toy"

        # Create GPT-OSS toy model config
        config = GptOssConfig(**HF_GPT_OSS_TOY_MODEL_OVERRIDES)

        # Create model with random weights and convert to bfloat16
        model = GptOssForCausalLM(config)
        model = model.bfloat16() if hasattr(model, "bfloat16") else model

        # Use a tokenizer from a similar model architecture
        # GPT-OSS typically uses a similar tokenizer to Llama
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
        except Exception:
            # Fallback to a basic tokenizer if Llama tokenizer not available
            from transformers import GPT2Tokenizer

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.save_pretrained(model_dir)

        # Save model and config to directory
        model.save_pretrained(model_dir, safe_serialization=True)

        return str(model_dir)

    @pytest.fixture(scope="class")
    def gpt_oss_megatron_checkpoint(self, gpt_oss_toy_model_path, tmp_path_factory):
        """
        Convert the toy HuggingFace model to Megatron checkpoint format.

        Args:
            gpt_oss_toy_model_path: Path to the toy HuggingFace model (from fixture)
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the Megatron checkpoint directory
        """
        from megatron.core import parallel_state
        from megatron.core.rerun_state_machine import destroy_rerun_state_machine

        # Create a temporary directory for the Megatron checkpoint
        temp_dir = tmp_path_factory.mktemp("gpt_oss_megatron_ckpt")
        megatron_checkpoint_dir = temp_dir / "megatron_checkpoint"

        # Import the HF model to Megatron format
        AutoBridge.import_ckpt(
            hf_model_id=gpt_oss_toy_model_path,
            megatron_path=str(megatron_checkpoint_dir),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Clean up global state after import_ckpt
        # This is necessary to avoid conflicts when the actual test initializes Megatron
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        destroy_rerun_state_machine()

        return str(megatron_checkpoint_dir)

    def _finetune_wrapper_lora(self, checkpoint_dir, **kwargs):
        """Wrapper to adapt GPT-OSS finetune_config to the test runner signature (with LoRA).

        The runner will pass (dir, name) among others; we forward
        everything to finetune_config and inject the toy model checkpoint.
        """
        kwargs.setdefault("pretrained_checkpoint", checkpoint_dir)
        kwargs.setdefault("peft", "lora")  # Explicitly use LoRA
        return gpt_oss_20b_finetune_config(**kwargs)

    def _finetune_wrapper_full(self, checkpoint_dir, **kwargs):
        """Wrapper to adapt GPT-OSS finetune_config to the test runner signature (full SFT, no LoRA).

        The runner will pass (dir, name) among others; we forward
        everything to finetune_config and inject the toy model checkpoint.
        """
        kwargs.setdefault("pretrained_checkpoint", checkpoint_dir)
        kwargs.setdefault("peft", None)  # No PEFT for full finetuning
        return gpt_oss_20b_finetune_config(**kwargs)

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "recipe_name,model_overrides,use_lora",
        [
            (
                "gpt_oss_20b_lora",
                {
                    "num_layers": 2,  # Match toy model
                    "hidden_size": 256,  # Match toy model
                    "ffn_hidden_size": 512,  # Match toy model intermediate_size
                    "num_attention_heads": 8,  # Match toy model
                    "kv_channels": 32,  # hidden_size / num_attention_heads
                    "num_query_groups": 2,  # Match toy model num_key_value_heads
                    "num_moe_experts": 4,  # Match toy model
                    "moe_router_topk": 2,  # Match toy model num_experts_per_tok
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "expert_model_parallel_size": 1,
                    "sequence_parallel": False,
                },
                True,
            ),
            (
                "gpt_oss_20b_full",
                {
                    "num_layers": 2,  # Match toy model
                    "hidden_size": 256,  # Match toy model
                    "ffn_hidden_size": 512,  # Match toy model intermediate_size
                    "num_attention_heads": 8,  # Match toy model
                    "kv_channels": 32,  # hidden_size / num_attention_heads
                    "num_query_groups": 2,  # Match toy model num_key_value_heads
                    "num_moe_experts": 4,  # Match toy model
                    "moe_router_topk": 2,  # Match toy model num_experts_per_tok
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "expert_model_parallel_size": 1,
                    "sequence_parallel": False,
                },
                False,
            ),
        ],
    )
    def test_gpt_oss_finetune_recipes(
        self, gpt_oss_megatron_checkpoint, recipe_name, model_overrides, use_lora, tmp_path
    ):
        """Functional test for GPT-OSS finetuning recipes with LoRA and full SFT."""
        # Create the config using the appropriate wrapper
        if use_lora:
            config = self._finetune_wrapper_lora(
                checkpoint_dir=gpt_oss_megatron_checkpoint, dir=str(tmp_path), name=recipe_name
            )
        else:
            config = self._finetune_wrapper_full(
                checkpoint_dir=gpt_oss_megatron_checkpoint, dir=str(tmp_path), name=recipe_name
            )

        # Override the dataset to use MockGPTDataset for faster testing
        from megatron.bridge.training.config import MockGPTDatasetConfig

        seq_length = 512
        config.dataset = MockGPTDatasetConfig(
            random_seed=5678,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=seq_length,
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
