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
Nemotron Nano v2 recipes using the new flattened layout with _pretrain_common.

This file demonstrates the new pattern where recipes:
1. Call _pretrain_common() to get base config
2. Override model-specific settings directly on the returned config

Nemotron Nano v2 models:
- 9B: Uses NemotronNanoModelProvider9Bv2, TP=2, PP=1, SP=True, BF16 mixed precision
- 12B: Uses NemotronNanoModelProvider12Bv2, TP=4, PP=1, SP=True, FP8 precision

Key differences from _pretrain_common:
- train_iters=1_168_251 (vs 300000)
- global_batch_size=768 (vs 32)
- micro_batch_size=1 (vs 2)
- seq_length=8192 (vs 4096)
- lr_warmup_iters=2000 (vs 500)
- eval_interval=10 (vs 500)
- save_interval=10 (vs 500)
- DDP: overlap_param_gather=False (vs True)
- Uses NullTokenizer by default
"""

import torch

from megatron.bridge.models.nemotronh import (
    NemotronNanoModelProvider9Bv2,
    NemotronNanoModelProvider12Bv2,
)
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


def nemotron_nano_9b_v2_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Nemotron Nano 9B v2.

    This recipe is designed for single-node training (1 node).
    Default parallelism: TP=2, PP=1, SP=True.
    """
    cfg = _pretrain_common()

    # Model config - uses NemotronNanoModelProvider9Bv2
    cfg.model = NemotronNanoModelProvider9Bv2(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
    )

    # Parallel settings (already set in model provider above)
    cfg.model.pipeline_model_parallel_layout = None

    # Tokenizer - uses NullTokenizer by default
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data
    cfg.dataset.seq_length = 8192  # Must match model.seq_length
    cfg.dataset.num_workers = 8

    # Training config (DIFFERENT from _pretrain_common)
    cfg.train.train_iters = 1_168_251
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1
    cfg.train.eval_interval = 10
    # Old recipe doesn't set manual_gc, uses TrainingConfig defaults (not _pretrain_common)
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0
    cfg.train.manual_gc_eval = True

    # Optimizer - override only what differs from _pretrain_common
    # _pretrain_common defaults: lr_warmup_iters=500
    cfg.scheduler.lr_warmup_iters = 2000

    # Logger - old recipe doesn't set log_timers_to_tensorboard
    cfg.logger.log_timers_to_tensorboard = False

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None  # None means auto selection
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"  # NemotronH uses native

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - bf16_mixed
    cfg.mixed_precision = "bf16_mixed"
    # FP8 settings (commented - enable if using FP8)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.use_precision_aware_optimizer = False  # default in mcore's OptimizerConfig
    cfg.optimizer.main_grads_dtype = torch.float32  # default in mcore's OptimizerConfig
    cfg.optimizer.main_params_dtype = torch.float32  # default in mcore's OptimizerConfig
    cfg.optimizer.exp_avg_dtype = torch.float32  # default in mcore's OptimizerConfig
    cfg.optimizer.exp_avg_sq_dtype = torch.float32  # default in mcore's OptimizerConfig

    # Communication overlap - enabled by default for 9B
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=True,
    )

    # Checkpoint config (DIFFERENT from _pretrain_common: save_interval=10 vs 500)
    cfg.checkpoint.save_interval = 10
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config (DIFFERENT from _pretrain_common)
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False  # Different from _pretrain_common
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"  # Different from _pretrain_common

    return cfg


def nemotron_nano_12b_v2_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Nemotron Nano 12B v2.

    This recipe is designed for single-node training (1 node).
    Default parallelism: TP=4, PP=1, SP=True.

    Note: Uses FP8 precision by default. Communication overlap is disabled by default.
    """
    cfg = _pretrain_common()

    # Model config - uses NemotronNanoModelProvider12Bv2
    cfg.model = NemotronNanoModelProvider12Bv2(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
    )

    # Parallel settings (already set in model provider above)
    cfg.model.pipeline_model_parallel_layout = None

    # Tokenizer - uses NullTokenizer by default
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data
    cfg.dataset.seq_length = 8192  # Must match model.seq_length
    cfg.dataset.num_workers = 8

    # Training config (DIFFERENT from _pretrain_common)
    cfg.train.train_iters = 1_168_251
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1
    cfg.train.eval_interval = 10
    # Old recipe doesn't set manual_gc
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0
    cfg.train.manual_gc_eval = True

    # Optimizer - override only what differs from _pretrain_common
    # _pretrain_common defaults: lr_warmup_iters=500
    cfg.scheduler.lr_warmup_iters = 2000

    # Logger - old recipe doesn't set log_timers_to_tensorboard
    cfg.logger.log_timers_to_tensorboard = False

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None  # None means auto selection
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"  # NemotronH uses native

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - FP8 with current scaling for 12B
    cfg.mixed_precision = "nanov2_bf16_with_fp8_current_scaling_mixed"
    # FP8 settings (commented - already enabled via precision string above)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.use_precision_aware_optimizer = False  # default in mcore's OptimizerConfig
    cfg.optimizer.main_grads_dtype = torch.float32  # default in mcore's OptimizerConfig
    cfg.optimizer.main_params_dtype = torch.float32  # default in mcore's OptimizerConfig
    cfg.optimizer.exp_avg_dtype = torch.float32  # default in mcore's OptimizerConfig
    cfg.optimizer.exp_avg_sq_dtype = torch.float32  # default in mcore's OptimizerConfig

    # Communication overlap - disabled by default for 12B (FP8 compatibility)
    cfg.comm_overlap = None

    # Checkpoint config (DIFFERENT from _pretrain_common: save_interval=10 vs 500)
    cfg.checkpoint.save_interval = 10
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config (DIFFERENT from _pretrain_common)
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False  # Different from _pretrain_common
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"  # Different from _pretrain_common

    return cfg
