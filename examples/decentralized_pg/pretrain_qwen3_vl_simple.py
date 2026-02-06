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
==============================================================================
Example: Qwen3_VL Pretraining with Decentralized Process Groups (Simple)
==============================================================================

This example demonstrates the simplest way to enable decentralized process groups:
just use an existing recipe and set `cfg.dist.use_decentralized_pg = True`.

The setup() function inside pretrain() will automatically create the
ProcessGroupCollection using HyperCommGrid based on the parallelism settings.

How to Run
----------
# 8 GPUs: EP8
uv run python -m torch.distributed.run --nproc_per_node=8 examples/decentralized_pg/pretrain_qwen3_vl_simple.py
"""

import torch

from megatron.bridge.recipes.qwen_vl.qwen3_vl import qwen3_vl_30b_a3b_pretrain_config
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.vlm_step import forward_step


def main() -> None:
    """Run Qwen3 pretraining with decentralized process groups enabled."""
    # Get the standard Qwen3 4B pretrain config with overrides
    cfg = qwen3_vl_30b_a3b_pretrain_config(
        # Use mock data for demo
        mock=True,
        # Parallelism
        expert_model_parallel_size=8,
        # Training settings (small for demo)
        train_iters=100,
        seq_length=1024,
        global_batch_size=32,
        micro_batch_size=1,
        # LR schedule (must fit within train_iters)
        lr_warmup_iters=10,
        lr_decay_iters=100,
    )
    # known issue with share_embeddings_and_output_weights
    cfg.model.share_embeddings_and_output_weights = False

    # =========================================================================
    # KEY: Enable decentralized process groups
    # =========================================================================
    cfg.dist.use_decentralized_pg = True
    cfg.dist.use_gloo_process_groups = False  # Gloo not supported with decentralized PG

    pretrain(config=cfg, forward_step_func=forward_step)

    # Cleanup
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
