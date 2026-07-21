# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""GB200 library recipes for Nemotron 3.5 Nano."""

from megatron.bridge.recipes.nemotronh.h100.nemotron_3_5_nano import (
    nemotron_3_5_nano_pretrain_16gpu_h100_bf16_config,
)
from megatron.bridge.training.config import ConfigContainer


def nemotron_3_5_nano_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Return the bounded pre-training config for eight GB200 GPUs.

    The convergence contract matches the H100 recipe: 100 steps at sequence
    length 4096 and global batch size 1024 with natural MoE routing. GB200 uses
    micro batch size 2, the broader Transformer Engine CUDA graph scope, and no
    activation recompute.

    Returns:
        ConfigContainer: Pre-training configuration for Nemotron 3.5 Nano.
    """
    cfg = nemotron_3_5_nano_pretrain_16gpu_h100_bf16_config()

    cfg.train.micro_batch_size = 2

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None

    cfg.optimizer.optimizer_cpu_offload = False
    cfg.optimizer.optimizer_offload_fraction = 0.0
    cfg.optimizer.overlap_cpu_optimizer_d2h_h2d = False

    cfg.env_vars = {
        **cfg.env_vars,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg
