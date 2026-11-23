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
"""HF-defined native Kimi K3 pretraining recipe."""

import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


def kimi_k3_pretrain_config() -> ConfigContainer:
    """Build the 512-GPU BF16 setup using the canonical HF-defined Hybrid model.

    TP4/PP4/EP128 and full recomputation follow the early K3 training setup.
    Complete source blocks are balanced over PP stages (24/23/23/23 for 93 blocks).
    This recipe has no full-model throughput or convergence claim. Set KIMI_K3_HF_PATH
    for a local snapshot, provide a tokenized dataset, and select a checkpoint before use.
    KDA context parallelism and virtual pipeline parallelism remain unsupported.
    """
    return _kimi_k3_pretrain_common()


def _kimi_k3_pretrain_common() -> ConfigContainer:
    """Shared early Kimi K3 BF16 configuration for NVL72 platforms."""
    hf_path = os.environ.get("KIMI_K3_HF_PATH", "moonshotai/Kimi-K3")
    cfg = _pretrain_common()
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).get_model_config()
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 128
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.variable_seq_lengths = True
    cfg.model.overlap_moe_expert_parallel_comm = False
    cfg.model.seq_length = 8192
    cfg.model.params_dtype = torch.bfloat16
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.gradient_accumulation_fusion = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.train.train_iters = 10000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        max_lr=1e-5,
        min_lr=0.0,
        lr_warmup_iters=100,
        lr_decay_iters=10000,
        weight_decay=0.1,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
        clip_grad=1.0,
    )
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.comm_overlap = None

    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.tokenizer.vocab_file = (
        str(Path(hf_path) / "tiktoken.model") if Path(hf_path).is_dir() else hf_hub_download(hf_path, "tiktoken.model")
    )
    cfg.tokenizer.trust_remote_code = True
    cfg.tokenizer.use_tokenizer_vocab_size = False
    cfg.tokenizer.vocab_size = cfg.model.vocab_size
    cfg.dataset.seq_length = 8192
    cfg.dataset.num_workers = 2
    cfg.dataset.skip_getting_attention_mask_from_dataset = True
    cfg.dataset.reset_attention_mask = False
    cfg.dataset.reset_position_ids = False
    cfg.dataset.eod_mask_loss = False
    cfg.checkpoint.hf_trust_remote_code = True
    cfg.logger.log_interval = 1
    cfg.logger.log_memory_interval = 1
    cfg.env_vars = {**COMMON_RECIPE_ENV_VARS, "NCCL_PXN_DISABLE": "1"}
    return cfg
