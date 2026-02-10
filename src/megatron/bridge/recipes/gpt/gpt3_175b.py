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

import torch

from megatron.bridge.models.gpt_provider import GPTProvider175B
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig, userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import get_mixed_precision_config


def gpt3_175b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for GPT3-175B.

    The default configuration is expected to run on 64 nodes with 8 GPUs each.
    Default parallelism: TP=4, PP=8, VP=6, SP=True.
    """
    cfg = _pretrain_common()

    # Model config - uses GPTProvider175B
    cfg.model = GPTProvider175B(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=8,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=6,
        context_parallel_size=1,
        sequence_parallel=True,
    )

    # Parallel settings
    cfg.model.pipeline_model_parallel_layout = None

    # Tokenizer - uses NullTokenizer
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.seq_length = 2048
    cfg.dataset.num_workers = 8

    # Training config
    cfg.train.train_iters = 1_168_251
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 2
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Optimizer
    cfg.scheduler.lr_warmup_iters = 2000
    cfg.optimizer.lr = 0.9e-4
    cfg.optimizer.min_lr = 0.9e-5

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"  # GPT uses native

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - bf16_mixed with grad_reduce_in_fp32=False
    cfg.mixed_precision = get_mixed_precision_config("bf16_mixed")
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    # FP8 settings (commented - enable if using FP8)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap - enabled with userbuffers config
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        tp_comm_overlap_cfg=userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048,
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=50,
        overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to issue with async checkpointing
    )

    # Checkpoint config
    cfg.checkpoint.save_interval = 2000
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg
