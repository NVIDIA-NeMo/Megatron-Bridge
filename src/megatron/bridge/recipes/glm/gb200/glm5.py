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
"""GB200 recipes for GLM-5.2."""

from __future__ import annotations

from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.glm.h100.glm5 import (
    glm52_peft_208gpu_h100_bf16_config,
    glm52_pretrain_416gpu_h100_bf16_config,
    glm52_sft_416gpu_h100_bf16_config,
    glm52_sft_608gpu_h100_bf16_200k_config,
)
from megatron.bridge.training.config import ConfigContainer


def _configure_gb200_model(
    cfg: ConfigContainer,
    *,
    context_parallel_size: int,
    microbatch_group_size: int | None,
    dispatcher_type: str,
    flex_dispatcher_backend: str | None,
) -> None:
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.context_parallel_size = context_parallel_size
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.num_layers_in_first_pipeline_stage = 14
    cfg.model.num_layers_in_last_pipeline_stage = 16
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.microbatch_group_size_per_vp_stage = microbatch_group_size
    cfg.model.moe_token_dispatcher_type = dispatcher_type
    cfg.model.moe_flex_dispatcher_backend = flex_dispatcher_backend
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.apply_rope_fusion = False


def glm52_pretrain_192gpu_gb200_bf16_config() -> ConfigContainer:
    """GLM-5.2 bounded pretraining on 192 GB200 GPUs."""
    cfg = glm52_pretrain_416gpu_h100_bf16_config()
    _configure_gb200_model(
        cfg,
        context_parallel_size=1,
        microbatch_group_size=6,
        dispatcher_type="alltoall",
        flex_dispatcher_backend=None,
    )
    return cfg


def glm52_sft_192gpu_gb200_bf16_config() -> ConfigContainer:
    """GLM-5.2 bounded full SFT on 192 GB200 GPUs."""
    cfg = glm52_sft_416gpu_h100_bf16_config()
    _configure_gb200_model(
        cfg,
        context_parallel_size=1,
        microbatch_group_size=6,
        dispatcher_type="alltoall",
        flex_dispatcher_backend=None,
    )
    cfg.dataset.hf_output_root = "work/data/glm5-2/tulu3-full-sft-gb200"
    cfg.dataset.hf_rewrite = False
    return cfg


def glm52_sft_192gpu_gb200_bf16_128k_config() -> ConfigContainer:
    """GLM-5.2 128K packed SFT with context parallelism on 192 GB200 GPUs."""
    cfg = glm52_sft_608gpu_h100_bf16_200k_config()

    cfg.model.seq_length = 131072
    _configure_gb200_model(
        cfg,
        context_parallel_size=32,
        microbatch_group_size=None,
        dispatcher_type="flex",
        flex_dispatcher_backend="hybridep",
    )

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


def glm52_peft_192gpu_gb200_bf16_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """GLM-5.2 bounded PEFT on 192 GB200 GPUs."""
    cfg = glm52_peft_208gpu_h100_bf16_config(peft_scheme)
    _configure_gb200_model(
        cfg,
        context_parallel_size=1,
        microbatch_group_size=6,
        dispatcher_type="alltoall",
        flex_dispatcher_backend=None,
    )
    cfg.dataset.hf_output_root = "work/data/glm5-2/tulu3-peft-gb200"
    cfg.dataset.hf_rewrite = False
    return cfg


__all__ = [
    "glm52_peft_192gpu_gb200_bf16_config",
    "glm52_pretrain_192gpu_gb200_bf16_config",
    "glm52_sft_192gpu_gb200_bf16_128k_config",
    "glm52_sft_192gpu_gb200_bf16_config",
]
