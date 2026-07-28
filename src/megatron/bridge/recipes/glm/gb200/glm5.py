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
"""Functional GB200 recipes for GLM-5.2 verification."""

from __future__ import annotations

from megatron.bridge.recipes.glm.h100.glm5 import glm52_sft_long_context_608gpu_h100_bf16_config
from megatron.bridge.training.config import ConfigContainer


def glm52_sft_long_context_192gpu_gb200_bf16_config() -> ConfigContainer:
    """GLM-5.2 131K packed SFT with context parallelism on 192 GB200 GPUs."""
    cfg = glm52_sft_long_context_608gpu_h100_bf16_config()

    cfg.model.seq_length = 131072
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.num_layers_in_first_pipeline_stage = 14
    cfg.model.num_layers_in_last_pipeline_stage = 16
    cfg.model.microbatch_group_size_per_vp_stage = None
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.apply_rope_fusion = False

    cfg.dataset.seq_length = 131072
    cfg.dataset.dataset_root = "work/data/glm5-2/synthetic-long-context-gb200"
    cfg.dataset.offline_packing_specs.packed_sequence_size = 131072
    cfg.dataset.offline_packing_specs.tokenizer_model_name = "glm5"
    cfg.dataset.offline_packing_specs.pad_seq_to_mult = 64
    cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.train_iters = 20
    cfg.env_vars.update(
        {
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
            "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
            "NVLINK_DOMAIN_SIZE": 72,
            "USE_MNNVL": 1,
        }
    )
    return cfg


__all__ = [
    "glm52_sft_long_context_192gpu_gb200_bf16_config",
]
