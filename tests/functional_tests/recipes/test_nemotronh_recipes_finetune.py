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

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer, dynamic_module_utils

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.recipes.nemotronh import (
    nemotron_3_nano_peft_config,
    nemotron_3_nano_sft_config,
    nemotron_nano_9b_v2_peft_config,
    nemotron_nano_9b_v2_sft_config,
)
from megatron.bridge.training.config import ConfigContainer


# Overrides for toy model - create a tiny version that matches Nemotron Nano v2 architecture
# Key: keep the same mamba_num_heads and mamba_head_dim ratio as the full model for shape compatibility
HF_NEMOTRONH_TOY_MODEL_OVERRIDES = {
    "attention_head_dim": 80,  # Match mamba_head_dim
    "head_dim": 80,  # Match mamba_head_dim
    "chunk_size": 48,
    "expand": 2,
    "hidden_size": 640,  # 8 heads * 80 head_dim
    "hybrid_override_pattern": "M*M-",  # Minimal 4-layer pattern
    "initializer_range": 0.02,
    "intermediate_size": 2240,  # ~3.5x hidden_size
    "layer_norm_epsilon": 1e-05,
    "mamba_head_dim": 80,  # Match Nemotron Nano v2
    "mamba_hidden_act": "silu",
    "mamba_num_heads": 128,  # Match Nemotron Nano v2 for shape compatibility
    "max_position_embeddings": 8192,
    "n_groups": 8,
    "num_attention_heads": 8,  # hidden_size / attention_head_dim = 640 / 80
    "num_hidden_layers": 4,
    "num_key_value_heads": 4,  # Half of num_attention_heads (GQA)
    "ssm_state_size": 128,  # Match Nemotron Nano v2
    "vocab_size": 131072,
}


class TestNemotronNanoV2FinetuneRecipes:
    """Test class for Nemotron Nano v2 finetune recipe functional tests."""

    @pytest.fixture(scope="class")
    def nemotronh_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace NemotronH toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("nemotronh_toy_model")
        model_dir = temp_dir / "nemotronh_toy"

        repo_id = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"

        # Create NemotronH toy model config by starting with 8B and applying overrides
        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        for k, v in HF_NEMOTRONH_TOY_MODEL_OVERRIDES.items():
            setattr(config, k, v)

        # Create model with random weights and convert to bfloat16
        model_class_ref = config.auto_map["AutoModelForCausalLM"]
        model_class = dynamic_module_utils.get_class_from_dynamic_module(
            class_reference=model_class_ref,
            pretrained_model_name_or_path=repo_id,
            cache_dir=None,
            force_download=False,
            resume_download=True,
            proxies=None,
            use_auth_token=None,
            revision=None,
            local_files_only=False,
            repo_id=repo_id,
        )
        model = model_class(config)
        model = model.bfloat16() if hasattr(model, "bfloat16") else model

        # Download and save tokenizer from a reference NemotronH model
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        tokenizer.save_pretrained(model_dir)

        # Save model, config, and modeling code to directory
        model.save_pretrained(model_dir, safe_serialization=True)
        modeling_filepath = os.path.abspath(sys.modules[model_class.__module__].__file__)
        shutil.copy(modeling_filepath, model_dir)

        # Ensure config.json exists with expected keys
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

        return str(model_dir)

    @pytest.fixture(scope="class")
    def nemotronh_megatron_checkpoint(self, nemotronh_toy_model_path, tmp_path_factory):
        """
        Convert the toy HuggingFace model to Megatron checkpoint format.

        Args:
            nemotronh_toy_model_path: Path to the toy HuggingFace model (from fixture)
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the Megatron checkpoint directory
        """
        from megatron.core import parallel_state
        from megatron.core.rerun_state_machine import destroy_rerun_state_machine

        # Create a temporary directory for the Megatron checkpoint
        temp_dir = tmp_path_factory.mktemp("nemotronh_megatron_ckpt")
        megatron_checkpoint_dir = temp_dir / "megatron_checkpoint"

        # Import the HF model to Megatron format
        AutoBridge.import_ckpt(
            hf_model_id=nemotronh_toy_model_path,
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
        """Wrapper to adapt Nemotron Nano v2 peft_config to the test runner signature (with LoRA).

        The runner will pass (dir, name) among others; we create the config
        with the new parameterless API and inject the toy model checkpoint.
        """
        config = nemotron_nano_9b_v2_peft_config(peft_scheme="lora")
        config.checkpoint.pretrained_checkpoint = checkpoint_dir
        config.checkpoint.load = None  # Don't try to resume from default path and load from pretrained checkpoint
        return config

    def _finetune_wrapper_full(self, checkpoint_dir, **kwargs):
        """Wrapper to adapt Nemotron Nano v2 sft_config to the test runner signature (full SFT, no LoRA).

        The runner will pass (dir, name) among others; we create the config
        with the new parameterless API and inject the toy model checkpoint.
        """
        config = nemotron_nano_9b_v2_sft_config()
        config.checkpoint.pretrained_checkpoint = checkpoint_dir
        config.checkpoint.load = None  # Don't try to resume from default path and load from pretrained checkpoint
        return config

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "recipe_name,model_overrides,use_lora",
        [
            (
                "nemotron_nano_9b_v2_lora",
                {
                    "num_layers": 4,  # Match toy model
                    "hybrid_override_pattern": "M*M-",  # Match toy model
                    "hidden_size": 640,  # Match toy model
                    "ffn_hidden_size": 2240,  # Match toy model
                    "num_attention_heads": 8,  # Match toy model
                    "kv_channels": 80,  # Match toy model attention_head_dim
                    "num_query_groups": 4,  # Match toy model
                    "mamba_num_heads": 128,  # Match toy model
                    "mamba_head_dim": 80,  # Match toy model
                    "mamba_state_dim": 128,  # Match toy model
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "sequence_parallel": False,
                },
                True,
            ),
            (
                "nemotron_nano_9b_v2_full",
                {
                    "num_layers": 4,  # Match toy model
                    "hybrid_override_pattern": "M*M-",  # Match toy model
                    "hidden_size": 640,  # Match toy model
                    "ffn_hidden_size": 2240,  # Match toy model
                    "num_attention_heads": 8,  # Match toy model
                    "kv_channels": 80,  # Match toy model attention_head_dim
                    "num_query_groups": 4,  # Match toy model
                    "mamba_num_heads": 128,  # Match toy model
                    "mamba_head_dim": 80,  # Match toy model
                    "mamba_state_dim": 128,  # Match toy model
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "sequence_parallel": False,
                },
                False,
            ),
        ],
    )
    def test_nemotron_nano_v2_finetune_recipes(
        self, nemotronh_megatron_checkpoint, recipe_name, model_overrides, use_lora, tmp_path
    ):
        """Functional test for Nemotron Nano v2 finetuning recipes with LoRA and full SFT."""
        # Create the config using the appropriate wrapper
        if use_lora:
            config = self._finetune_wrapper_lora(
                checkpoint_dir=nemotronh_megatron_checkpoint, dir=str(tmp_path), name=recipe_name
            )
        else:
            config = self._finetune_wrapper_full(
                checkpoint_dir=nemotronh_megatron_checkpoint, dir=str(tmp_path), name=recipe_name
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


HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES = {
    "num_hidden_layers": 3,
    "hybrid_override_pattern": "M*E",
    "hidden_size": 672,
    "n_routed_experts": 16,
}

MEGATRON_NEMOTRON_3_NANO_OVERRIDES = {
    "num_layers": HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES["num_hidden_layers"],
    "hybrid_override_pattern": HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES["hybrid_override_pattern"],
    "hidden_size": HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES["hidden_size"],
    "num_moe_experts": HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES["n_routed_experts"],
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "expert_tensor_parallel_size": 1,
    "expert_model_parallel_size": 1,
    "sequence_parallel": False,
    "moe_token_dispatcher_type": "alltoall",
    "moe_shared_expert_overlap": True,
}


class TestNemotron3NanoFinetuneRecipes:
    """Test class for Nemotron 3 Nano finetune recipe functional tests."""

    _TOY_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    @pytest.fixture(scope="class")
    def temp_hf_modules(self, tmp_path_factory):
        """Change transformers.dynamic_module_utils.HF_MODULES_CACHE to a temp path"""
        temp_hf_modules_cache = tmp_path_factory.mktemp("hf_modules_cache")

        # Store original value
        original_cache = dynamic_module_utils.HF_MODULES_CACHE

        # Patch
        dynamic_module_utils.HF_MODULES_CACHE = temp_hf_modules_cache

        yield temp_hf_modules_cache

        # Restore
        dynamic_module_utils.HF_MODULES_CACHE = original_cache

    @pytest.fixture(scope="class")
    def nemotron_3_nano_toy_model_path(self, tmp_path_factory: pytest.TempPathFactory) -> str:
        """
        Create and save a HuggingFace Nemotron 3 Nano toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("nemotron_3_nano_toy_model")
        model_dir = temp_dir / "nemotron_3_nano_toy"

        repo_id = self._TOY_MODEL_ID

        # Create Nemotron 3 Nano toy model config by starting with 30B and applying overrides
        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        for k, v in HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES.items():
            setattr(config, k, v)

        # Create model with random weights and convert to bfloat16
        model_class_ref = config.auto_map["AutoModelForCausalLM"]
        model_class = dynamic_module_utils.get_class_from_dynamic_module(
            class_reference=model_class_ref,
            pretrained_model_name_or_path=repo_id,
            cache_dir=None,
            force_download=False,
            resume_download=True,
            proxies=None,
            use_auth_token=None,
            revision=None,
            local_files_only=False,
            repo_id=repo_id,
        )
        model = model_class(config)
        model = model.bfloat16() if hasattr(model, "bfloat16") else model

        # There is a bug in Nemotron Nano HF implementation that
        # TopKRouter weights are not initialized correctly, which leads to NaN values.
        # Reinitialize weights of all NemotronHTopkRouter modules if present
        for module in model.modules():
            if module.__class__.__name__ == "NemotronHTopkRouter":
                torch.nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
                torch.nn.init.zeros_(module.e_score_correction_bias)

        # Initialize e_score_correction_bias to float32 if present
        for k, v in model.named_buffers():
            if "e_score_correction_bias" in k:
                v.data = v.data.to(torch.float32)

        # Download and save tokenizer from a reference Nemotron 3 Nano model
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        tokenizer.save_pretrained(model_dir)

        # Save model, config, and modeling code to directory
        model.save_pretrained(model_dir, safe_serialization=True)
        modeling_filepath = os.path.abspath(sys.modules[model_class.__module__].__file__)
        shutil.copy(modeling_filepath, model_dir)

        # Ensure config.json exists with expected keys
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

        return str(model_dir)

    @pytest.fixture(scope="class")
    def nemotron_3_nano_megatron_checkpoint(
        self,
        nemotron_3_nano_toy_model_path: str,
        tmp_path_factory: pytest.TempPathFactory,
        temp_hf_modules: Path,
    ) -> str:
        """
        Convert the toy HuggingFace model to Megatron checkpoint format.

        Args:
            nemotron_3_nano_toy_model_path: Path to the toy HuggingFace model (from fixture)
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures
            temp_hf_modules: Temporary HF modules cache path (from fixture)

        Returns:
            str: Path to the Megatron checkpoint directory
        """
        from megatron.core import parallel_state
        from megatron.core.rerun_state_machine import destroy_rerun_state_machine

        # Create a temporary directory for the Megatron checkpoint
        temp_dir = tmp_path_factory.mktemp("nemotron_3_nano_megatron_ckpt")
        megatron_checkpoint_dir = temp_dir / "megatron_checkpoint"

        # Import the HF model to Megatron format
        AutoBridge.import_ckpt(
            hf_model_id=nemotron_3_nano_toy_model_path,
            megatron_path=str(megatron_checkpoint_dir),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Clean up global state after import_ckpt
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        destroy_rerun_state_machine()

        return str(megatron_checkpoint_dir)

    def _get_finetune_config(self, checkpoint_dir: str, peft: Optional[str] = None, **kwargs) -> ConfigContainer:
        """
        Wrapper to adapt Nemotron 3 Nano sft/peft config to the test runner signature.

        Args:
            checkpoint_dir: Path to the pretrained Megatron checkpoint.
            peft: PEFT method to use (e.g. "lora"). If None, full finetuning is used.
            **kwargs: Additional arguments (ignored, kept for compatibility).

        Returns:
            ConfigContainer: The generated finetuning configuration.
        """
        if peft is not None:
            config = nemotron_3_nano_peft_config(peft_scheme=peft)
        else:
            config = nemotron_3_nano_sft_config()
        config.checkpoint.pretrained_checkpoint = checkpoint_dir
        config.checkpoint.load = None  # Don't try to resume from default path and load from pretrained checkpoint
        return config

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "recipe_name,model_overrides,use_lora",
        [
            (
                "nemotron_3_nano_lora",
                MEGATRON_NEMOTRON_3_NANO_OVERRIDES,
                True,
            ),
            (
                "nemotron_3_nano_full",
                MEGATRON_NEMOTRON_3_NANO_OVERRIDES,
                False,
            ),
        ],
    )
    def test_nemotron_3_nano_finetune_recipes(
        self,
        nemotron_3_nano_megatron_checkpoint: str,
        recipe_name: str,
        model_overrides: Dict[str, Any],
        use_lora: bool,
        tmp_path: Any,
    ):
        """
        Functional test for Nemotron 3 Nano finetuning recipes with LoRA and full SFT.

        Args:
            nemotron_3_nano_megatron_checkpoint: Path to the pretrained checkpoint.
            recipe_name: Name of the recipe test case.
            model_overrides: Dictionary of model configuration overrides.
            use_lora: Whether to use LoRA (True) or full fine-tuning (False).
            tmp_path: Temporary directory fixture.
        """
        # Create the config using the consolidated wrapper
        peft_strategy = "lora" if use_lora else None
        config = self._get_finetune_config(
            checkpoint_dir=nemotron_3_nano_megatron_checkpoint, dir=str(tmp_path), name=recipe_name, peft=peft_strategy
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
