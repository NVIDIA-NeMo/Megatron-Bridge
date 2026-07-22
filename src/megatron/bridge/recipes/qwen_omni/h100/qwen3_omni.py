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

"""Qwen3-Omni thinker training recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import ChatSFTPreprocessingConfig, DirectHFSFTDatasetConfig, HFDatasetSourceConfig
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


_QWEN3_OMNI_HF_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def qwen3_omni_30b_a3b_sft_8gpu_h100_bf16_config() -> "ConfigContainer":
    """Return an 8-GPU thinker-only SFT config for Qwen3-Omni 30B-A3B.

    Recommended parallelism: TP=1, PP=1, EP=8.
    """

    cfg = _sft_common_vlm()

    cfg.model = AutoBridge.from_hf_pretrained(_QWEN3_OMNI_HF_PATH).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_audio_model = False
    cfg.model.vit_gradient_checkpointing = False
    cfg.model.multimodal_attn_impl = "auto"

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.attention_backend = "auto"

    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=50,
        lr_decay_iters=1000,
        max_lr=5e-6,
        min_lr=5e-7,
        adam_beta2=0.98,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = _QWEN3_OMNI_HF_PATH
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.skip_getting_attention_mask_from_dataset = False

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen3_omni_30b_a3b_sft_8gpu_h100_bf16_hf_json_config() -> "ConfigContainer":
    """Return an 8-GPU SFT config for JSONL loaded through Hugging Face datasets."""

    cfg = qwen3_omni_30b_a3b_sft_8gpu_h100_bf16_config()
    cfg.dataset = DirectHFSFTDatasetConfig(
        seq_length=cfg.model.seq_length,
        preprocessing=ChatSFTPreprocessingConfig(),
        hf_processor_path=_QWEN3_OMNI_HF_PATH,
        source=HFDatasetSourceConfig(
            path_or_dataset="json",
            split="train",
            load_kwargs={"data_files": {"train": None}},
        ),
        validation_source=HFDatasetSourceConfig(
            path_or_dataset="json",
            split="validation",
            load_kwargs={"data_files": {"validation": None}},
        ),
        test_source=HFDatasetSourceConfig(
            path_or_dataset="json",
            split="test",
            load_kwargs={"data_files": {"test": None}},
        ),
        dataloader_type="single",
        num_workers=2,
        enable_in_batch_packing=False,
        skip_getting_attention_mask_from_dataset=False,
    )
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


__all__ = [
    "qwen3_omni_30b_a3b_sft_8gpu_h100_bf16_config",
    "qwen3_omni_30b_a3b_sft_8gpu_h100_bf16_hf_json_config",
]
