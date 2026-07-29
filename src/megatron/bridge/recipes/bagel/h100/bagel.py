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

"""BAGEL-7B-MoT training recipes."""

from megatron.bridge.models.bagel.data.dataset import BagelDatasetConfig
from megatron.bridge.models.bagel.provider import BagelModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


def bagel_7b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return an 8-GPU Megatron-FSDP BAGEL pretraining configuration."""
    cfg = _pretrain_common()
    cfg.model = BagelModelProvider()
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_vit = True
    cfg.dataset = BagelDatasetConfig()
    cfg.train.train_iters = 500_000
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = None
    cfg.validation.eval_iters = 0
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=2000,
        lr_decay_iters=500_000,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-15,
        weight_decay=0.0,
        max_lr=1e-4,
        min_lr=1e-4,
        start_weight_decay=0.0,
        end_weight_decay=0.0,
        lr_decay_style="constant",
    )
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.fsdp_double_buffer = True
    cfg.logger.log_interval = 10
    cfg.checkpoint.save_interval = 2000
    cfg.rng.seed = 42
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = 152064
    cfg.mixed_precision = "bf16_mixed"
    return cfg


def bagel_7b_finetune_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the official BAGEL fine-tuning overrides for the 8-GPU recipe."""
    cfg = bagel_7b_pretrain_8gpu_h100_bf16_config()
    cfg.dataset.expected_num_tokens = 10240
    cfg.dataset.max_num_tokens = 11520
    cfg.dataset.max_num_tokens_per_sample = 10240
    cfg.optimizer.lr = 2e-5
    cfg.optimizer.min_lr = 2e-5
    cfg.logger.log_interval = 1
    return cfg


__all__ = [
    "bagel_7b_finetune_8gpu_h100_bf16_config",
    "bagel_7b_pretrain_8gpu_h100_bf16_config",
]
