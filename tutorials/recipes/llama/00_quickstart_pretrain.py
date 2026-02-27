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
Quickstart: Pretrain Llama 3.2 1B with Megatron Bridge

Usage:
    Single GPU:
        torchrun --nproc_per_node=1 00_quickstart_pretrain.py

    Multiple GPUs (automatic data parallelism):
        torchrun --nproc_per_node=8 00_quickstart_pretrain.py

The script uses sensible defaults and mock data for quick testing.
For custom configurations through YAML and Hydra-style overrides, see 02_pretrain_with_yaml.py
For multi-node training, see launch_with_sbatch.sh or 04_launch_slurm_with_nemo_run.py
"""

from megatron.bridge.recipes.llama import llama32_1b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


def main() -> None:
    """Run Llama 3.2 1B pretraining with default configuration."""

    # Load the base recipe configuration
    # Llama 3.2 1B works on a single GPU (TP=1, PP=1, CP=1)
    config = llama32_1b_pretrain_config()

    # OPTIONAL: Customize key settings here
    # Uncomment and modify as needed:

    # For a quick test run:
    config.train.train_iters = 10
    config.scheduler.lr_warmup_iters = 2

    # Use your own data:
    # config.data.data_path = "/path/to/your/dataset"

    # Adjust batch sizes for your GPU memory:
    # config.train.global_batch_size = 256
    # config.train.micro_batch_size = 2

    # Change checkpoint save frequency:
    # config.train.save_interval = 500

    # Start pretraining
    pretrain(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
