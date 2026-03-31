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
Getting Started — Step 1: Pretraining with Recipes

Launch a quick pretraining run using a pre-configured recipe.
Recipes set model architecture, optimizer, data, and parallelism defaults.

Usage (single GPU):
    torchrun --nproc-per-node=1 examples/getting_started/01_pretrain.py

Usage (multi-GPU):
    torchrun --nproc-per-node=8 examples/getting_started/01_pretrain.py
"""

from megatron.bridge.recipes.llama import llama32_1b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


def main():
    """Launch a quick pretraining run with a Llama 3.2 1B recipe."""
    # Step 1: Load a pre-configured recipe
    cfg = llama32_1b_pretrain_config(seq_length=1024)

    # Step 2: Override training parameters for a quick test run
    cfg.train.train_iters = 10
    cfg.scheduler.lr_decay_iters = 10000
    cfg.model.vocab_size = 8192
    cfg.tokenizer.vocab_size = cfg.model.vocab_size

    # Step 3: Launch pretraining
    pretrain(cfg, forward_step)


if __name__ == "__main__":
    main()
