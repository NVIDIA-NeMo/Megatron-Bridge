#!/usr/bin/env python3
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

"""
NeMo Llama3-8B LoRA Fine-tuning Script on SQuAD with Weights & Biases Logging

This script fine-tunes Llama3-8B using LoRA (Low-Rank Adaptation) on the SQuAD dataset
using the official finetune_recipe with hardcoded configuration.

Usage:
    python nemo_llama3_8b_lora_finetune.py

Environment Variables Required:
    WANDB_API_KEY: Your Weights & Biases API key
"""

import os
from pathlib import Path


# Set NEMO_HOME to a writable directory in your workspace
os.environ["NEMO_HOME"] = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/nemo_cache"
os.environ["NEMO_MODELS_CACHE"] = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/nemo_cache/models"

import nemo_run as run
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.collections.llm.gpt.model.llama import Llama3Config8B, LlamaModel
from nemo.collections.llm.recipes.llama3_8b import finetune_recipe
from nemo.collections.llm.recipes.log.default import wandb_logger


# HuggingFace model source
HF_MODEL_URI = "meta-llama/Meta-Llama-3-8B"

# Set this to True if checkpoint is already imported
SKIP_IMPORT = True


def create_hf_tokenizer():
    """Create HuggingFace tokenizer configuration."""
    return run.Config(
        AutoTokenizer,
        pretrained_model_name=HF_MODEL_URI,
        use_fast=True,
    )


def import_checkpoint_directly():
    """
    Import checkpoint directly in the main process to avoid subprocess environment issues.
    """
    print("Importing checkpoint directly...")

    # Import checkpoint using proper model instance
    output_path = Path("/lustre/fsw/coreai_dlalgo_genai/ansubramania/nemo_cache/models/meta-llama/Meta-Llama-3-8B")

    result = llm.import_ckpt(
        model=LlamaModel(Llama3Config8B()),
        source=f"hf://{HF_MODEL_URI}",
        output_path=output_path,
        overwrite=True,
    )
    print(f"Checkpoint import completed successfully: {result}")
    return result


def main():
    """Main execution function."""
    experiment_name = "llama3_8b_lora_finetune"

    # Configuration parameters
    save_dir = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/checkpoints/megatron_hub_peft/nemo_peft_baseline"
    wandb_exp_name = "nemo_squad_llama3_8b_lora_gbs_128_seq_length_2048_lr_1e-4"

    # Import checkpoint directly if needed
    if not SKIP_IMPORT:
        print("Starting checkpoint import...")
        import_checkpoint_directly()
        print("Checkpoint import completed!")

    # Create fine-tuning recipe
    recipe = finetune_recipe(
        dir=save_dir,
        name=wandb_exp_name,
        num_nodes=1,
        num_gpus_per_node=8,
        peft_scheme="lora",
        seq_length=2048,
        packed_sequence=False,
        performance_mode=False,
    )

    # Set explicit tokenizer
    recipe.data.tokenizer = create_hf_tokenizer()

    # Set W&B logger directly in the recipe
    recipe.log.wandb = wandb_logger(project="megatron-hub-custom-loop-peft", name=wandb_exp_name, entity="nvidia")

    # Create main executor with environment variables
    executor = run.LocalExecutor(
        ntasks_per_node=8,
        launcher="torchrun",
        env_vars={
            "NEMO_HOME": "/lustre/fsw/coreai_dlalgo_genai/ansubramania/nemo_cache",
            "NEMO_MODELS_CACHE": "/lustre/fsw/coreai_dlalgo_genai/ansubramania/nemo_cache/models",
            "TRANSFORMERS_OFFLINE": "0",  # Allow online downloads
        },
    )

    # Create experiment for fine-tuning only
    with run.Experiment(experiment_name) as exp:
        print("Adding fine-tuning task...")
        exp.add(
            recipe,
            executor=executor,
            name=experiment_name,
        )

        # Run fine-tuning
        print("Starting fine-tuning...")
        exp.run(sequential=True, detach=True)

    print("Fine-tuning completed!")


if __name__ == "__main__":
    main()
