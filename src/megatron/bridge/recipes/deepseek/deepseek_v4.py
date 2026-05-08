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

"""Pre-training recipe for DeepSeek-V4.

DSv4 currently requires TP=1 (no MLA tensor parallelism with the hybrid attention
path); scale via expert and pipeline parallelism. The same recipe drives all four
published variants — Flash, Flash-Base, Pro, Pro-Base — by overriding
``hf_model_id`` and the EP/PP defaults appropriate for the model size.
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


_DEFAULT_HF_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-Base"


def deepseek_v4_pretrain_config(
    *,
    hf_model_id: str = _DEFAULT_HF_MODEL_ID,
    pipeline_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 8,
    seq_length: int = 4096,
    train_iters: int = 1_000_000,
    global_batch_size: int = 1024,
    mtp_num_layers: int = 1,
) -> ConfigContainer:
    """Return a pre-training config for the DeepSeek-V4 family.

    Args:
        hf_model_id: HF repo id of the source variant. Use ``deepseek-ai/DeepSeek-V4-Flash``,
            ``DeepSeek-V4-Flash-Base``, ``DeepSeek-V4-Pro``, or ``DeepSeek-V4-Pro-Base``.
        pipeline_model_parallel_size: Pipeline-parallel size. Recommended values:
            1 for Flash on a single 4×B200 node, 4 for Flash multi-node, 8 for Pro.
        expert_model_parallel_size: Expert-parallel size. Recommended values:
            8 for Flash (256 experts → 32 experts/rank) on 8 GPUs, 16 for Pro.
        seq_length: Sequence length. DSv4's YaRN config supports up to 1M tokens.
        train_iters: Total pre-training iterations.
        global_batch_size: Global batch size in samples.
        mtp_num_layers: Number of Multi-Token Prediction layers. Set to 0 to disable MTP.
    """
    cfg = _pretrain_common()

    # Model: derived from HF config; bridge supplies all DSv4-specific knobs
    cfg.model = AutoBridge.from_hf_pretrained(hf_model_id).to_megatron_provider(load_weights=False)

    # Tokenizer — NullTokenizer for mock/pretrain blends
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset — mock by default; pass blend tuples to use real data
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Parallelism — DSv4 requires TP=1
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = pipeline_model_parallel_size
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False  # TP=1, so sequence_parallel has no effect
    cfg.model.seq_length = seq_length

    # MTP
    cfg.model.mtp_num_layers = mtp_num_layers if mtp_num_layers > 0 else None
    cfg.model.mtp_loss_scaling_factor = 0.1

    # MoE token dispatcher
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None  # Options: None, "deepep", "hybridep"
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_router_fusion = False

    # Training
    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5
    cfg.validation.eval_interval = 2000
    cfg.scheduler.lr_warmup_iters = 2000

    # TE / kernel selections
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None  # auto
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # CUDA Graph (off by default; enable for inference-leaning workloads)
    cfg.model.cuda_graph_impl = "none"

    # Recompute / offloading — off by default
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision — bf16 master + bf16 grads (precision-aware optimizer below)
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
    cfg.model.moe_router_padding_for_fp8 = False

    # Optimizer — precision-aware with bf16 moments
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    # Communication overlap — TP overlap is irrelevant at TP=1
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = True

    # Checkpointing
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

    # MoE force load balancing — off by default
    cfg.model.moe_router_force_load_balancing = False

    if cfg.model.apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True  # MLA rope fusion is experimental

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    return cfg
