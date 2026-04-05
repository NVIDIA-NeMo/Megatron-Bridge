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

"""Training recipes for MiMo-V2-Flash.

MiMo-V2-Flash is a ~15B active parameter model with:
- 48 layers, hidden_size=4096
- 256 routed experts (moe_intermediate_size=2048), top-8 routing
- Hybrid attention: full + sliding-window alternating
- Total parameters ~70B+ (due to MoE)
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import ConfigContainer, TokenizerConfig
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


_HF_PATH = "XiaomiMiMo/MiMo-V2-Flash"


def _mimo_v2_flash_model(hf_path: str = _HF_PATH):
    """Load MiMo-V2-Flash architecture from HF (no weights)."""
    return AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(load_weights=False)


def mimo_v2_flash_pretrain_config(hf_path: str = _HF_PATH) -> ConfigContainer:
    """Return a pre-training config for MiMo-V2-Flash.

    Recommended parallelism: TP=2, PP=4, EP=32.

    The model has 256 fine-grained experts per MoE layer (47 of 48 layers),
    requiring significant expert parallelism.
    """
    cfg = _pretrain_common()

    cfg.model = _mimo_v2_flash_model(hf_path)

    # Tokenizer — NullTokenizer by default
    cfg.tokenizer = TokenizerConfig(
        tokenizer_type="NullTokenizer",
        tokenizer_model=None,
        vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE,
    )

    # Dataset — mock data by default
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Parallelism (MoE: EP dominates)
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096

    # Training
    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5

    # Scheduler
    cfg.scheduler.lr_warmup_iters = 2000

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"

    # Kernels
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory — selective recompute
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None

    # Mixed precision
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
    cfg.model.moe_router_padding_for_fp8 = False

    # Optimizer
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    # Checkpoint
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False

    # DDP
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def mimo_v2_flash_sft_config(hf_path: str = _HF_PATH) -> ConfigContainer:
    """Return a supervised fine-tuning config for MiMo-V2-Flash.

    Recommended parallelism: TP=2, PP=4, EP=32.
    """
    cfg = _sft_common()

    cfg.model = _mimo_v2_flash_model(hf_path)

    # Tokenizer
    cfg.tokenizer = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=hf_path,
    )

    # Parallelism (MoE: EP dominates)
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096

    # Training
    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    # Memory — selective recompute
    cfg.model.recompute_granularity = "selective"

    # Mixed precision
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )

    # DDP
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False

    return cfg


def mimo_v2_flash_peft_config(hf_path: str = _HF_PATH) -> ConfigContainer:
    """Return a LoRA fine-tuning config for MiMo-V2-Flash.

    PEFT uses smaller parallelism since only adapters are trained.
    Recommended parallelism: TP=1, PP=1, EP=32.
    """
    cfg = _peft_common()

    cfg.model = _mimo_v2_flash_model(hf_path)

    # Tokenizer
    cfg.tokenizer = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=hf_path,
    )

    # Parallelism — smaller TP/PP for PEFT, EP still needed for MoE
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096

    # Training
    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    # Mixed precision
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )

    return cfg
