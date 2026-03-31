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
Getting Started — Step 2: LoRA Fine-tuning

Fine-tune a model with LoRA using SFT recipes. LoRA freezes the base model
and only trains low-rank adapter weights, dramatically reducing GPU memory.

Usage (single GPU):
    torchrun --nproc-per-node=1 examples/getting_started/02_finetune_lora.py

Usage (multi-GPU):
    torchrun --nproc-per-node=8 examples/getting_started/02_finetune_lora.py
"""

from megatron.bridge.recipes.llama import llama32_1b_peft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step


def main():
    """Fine-tune Llama 3.2 1B with LoRA using an SFT recipe."""
    # Step 1: Load a pre-configured LoRA SFT recipe
    cfg = llama32_1b_peft_config(seq_length=1024)

    # Step 2: Override training parameters
    cfg.train.train_iters = 10

    # Step 3: Launch fine-tuning
    finetune(cfg, forward_step)


if __name__ == "__main__":
    main()
