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

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.training.config import ConfigContainer


_HF_PATH = "MiniMaxAI/MiniMax-2.5"


def minimax_m2_5_sft_config() -> ConfigContainer:
    """Return a full SFT config for MiniMax-2.5 (456B sparse MoE).

    MiniMax-2.5 is a 456B sparse MoE model with 256 experts, top-8 sigmoid
    routing, and FP8 block-wise quantized weights. Conversion from HuggingFace
    dequantizes FP8 weights automatically via ``MiniMaxM2Bridge``.

    Default parallelism: TP=1, PP=1, EP=32 (4 nodes / 32 GPUs).

    Note:
        MTP (Multi-Token Prediction) heads are not mapped by the bridge and
        are disabled in this recipe (``mtp_num_layers=0``). Enable them only
        when training from a Megatron checkpoint that already includes MTP weights.

    Returns:
        ConfigContainer: Pre-configured SFT config for MiniMax-2.5.
    """
    cfg = _sft_common()

    # Model — dispatches to the existing MiniMaxM2Bridge (model_type="minimax_m2")
    cfg.model = AutoBridge.from_hf_pretrained(_HF_PATH).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = _HF_PATH

    # Sequence length
    seq_length = 2048
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length

    # Parallelism
    # EP must divide 256 (number of experts). EP=32 → 8 experts per rank.
    # Minimum hardware: 4 nodes × 8 GPUs (TP=1, PP=1, EP=32).
    # TP does NOT reduce expert memory — increase EP instead.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False

    # MTP is not supported by MiniMaxM2Bridge; disable it
    cfg.model.mtp_num_layers = 0

    # MoE token dispatcher
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training
    cfg.train.train_iters = 1000
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 32
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    # Logger
    cfg.logger.log_interval = 1

    # Optimizer
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Scheduler
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 5e-6

    # Transformer Engine
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Recompute / offloading
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/megatron/checkpoint"

    # DDP — disable overlap to avoid OOM on 456B MoE during full fine-tuning
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.use_megatron_fsdp = False

    # RNG
    cfg.rng.seed = 5678

    return cfg


def minimax_m2_5_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT (LoRA/DoRA) config for MiniMax-2.5 (456B sparse MoE).

    Only adapter weights are trained, so fewer GPUs are required compared to
    full SFT.

    Default parallelism: TP=1, PP=1, EP=16 (2 nodes / 16 GPUs).

    Note:
        MTP heads are disabled (same limitation as SFT).

    Args:
        peft_scheme: PEFT method — ``"lora"``, ``"dora"``, or a custom
            :class:`~megatron.bridge.peft.base.PEFT` instance. Defaults to
            ``"lora"``.

    Returns:
        ConfigContainer: Pre-configured PEFT config for MiniMax-2.5.
    """
    cfg = _peft_common()

    # Model — dispatches to the existing MiniMaxM2Bridge
    cfg.model = AutoBridge.from_hf_pretrained(_HF_PATH).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = _HF_PATH

    # Sequence length
    seq_length = 2048
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length

    # Parallelism
    # EP=16 → 16 experts per rank. Minimum: 2 nodes × 8 GPUs (TP=1, PP=1, EP=16).
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.sequence_parallel = False

    # MTP not supported by MiniMaxM2Bridge
    cfg.model.mtp_num_layers = 0

    # MoE token dispatcher
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training
    cfg.train.train_iters = 1000
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 32
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    # Logger
    cfg.logger.log_interval = 1

    # Optimizer
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Scheduler
    cfg.scheduler.lr_warmup_iters = 50

    # Transformer Engine
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Recompute / offloading
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/megatron/checkpoint"

    # DDP
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.use_megatron_fsdp = False

    # PEFT
    cfg.peft = default_peft_config(peft_scheme)

    return cfg
