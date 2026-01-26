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
OLMoE-7B recipe using the new flattened layout with _pretrain_common.

This file demonstrates the new pattern where recipes:
1. Call _pretrain_common() to get base config
2. Override model-specific settings directly on the returned config

OLMoE-7B is a 7B total (~1B active) MoE model with unique characteristics:
- Uses OlMoEModelProvider
- Uses precision-aware optimizer with bf16 gradients/moments
- Custom MixedPrecisionConfig (not "bf16_mixed" string) with grad_reduce_in_fp32=False
- adam_eps=1e-8 (different from default 1e-5)
- manual_gc_interval=5 (different from default 100)

Key differences from _pretrain_common:
- train_iters=500_000 (vs 300000)
- global_batch_size=2048 (vs 32)
- micro_batch_size=1 (vs 2)
- lr_warmup_iters=2000 (vs 500)
- eval_interval=2000 (vs 500)
- save_interval=2000 (vs 500)
- DDP: grad_reduce_in_fp32=False (vs True)
"""

import torch

from megatron.bridge.models.olmoe import OlMoEModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


def _get_olmoe_pipeline_layout(pp_size: int, vp_size: int):
    """Get pipeline layout for OLMoE-7B based on PP and VP size."""
    # OLMoE has 16 layers
    map_pp_vp_to_layout = {
        (1, 1): None,
        (2, 1): [["embedding"] + ["decoder"] * 8, ["decoder"] * 8 + ["loss"]],
        (4, 1): [["embedding"] + ["decoder"] * 4, ["decoder"] * 4, ["decoder"] * 4, ["decoder"] * 4 + ["loss"]],
        (2, 2): [["embedding"] + ["decoder"] * 4, ["decoder"] * 4, ["decoder"] * 4, ["decoder"] * 4 + ["loss"]],
    }
    if (pp_size, vp_size) not in map_pp_vp_to_layout:
        raise ValueError(
            f"Invalid PP and VP size: {pp_size} and {vp_size} to infer PP layout "
            f"for OLMoE (7B). Known PP and VP combinations: {map_pp_vp_to_layout.keys()}"
        )
    layout = map_pp_vp_to_layout[(pp_size, vp_size)]
    if layout is not None:
        layout = list([list(x) for x in layout])
    return layout


def olmoe_7b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for OLMoE-7B (7B total, ~1B active).

    Recommended parallelism: TP=1, PP=1, EP=8
    Uses precision-aware optimizer with bf16 gradients/moments.
    """
    cfg = _pretrain_common()

    # Model config - uses OlMoEModelProvider
    cfg.model = OlMoEModelProvider(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        sequence_parallel=False,
        recompute_granularity="selective",
        recompute_modules=None,
        recompute_method=None,
        recompute_num_layers=None,
    )

    # Pipeline layout
    cfg.model.pipeline_model_parallel_layout = _get_olmoe_pipeline_layout(1, 1)

    # Performance optimization knobs
    cfg.model.moe_permute_fusion = True

    # Tokenizer - uses NullTokenizer with model vocab_size
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = cfg.model.vocab_size

    # Dataset config
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data
    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"  # Default
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # Options: None, deepep, hybridep
    cfg.model.moe_hybridep_num_sms = 16  # Number of SMs for hybridep backend

    # Training config (DIFFERENT from _pretrain_common)
    cfg.train.train_iters = 500_000
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.train.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5

    # Optimizer - override what differs from _pretrain_common
    cfg.scheduler.lr_warmup_iters = 2000
    cfg.scheduler.lr_decay_iters = cfg.train.train_iters
    cfg.optimizer.adam_eps = 1e-8  # Different from default 1e-5

    # Precision-aware optimizer settings (DIFFERENT: uses bf16 moments)
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections (includes MoE-specific kernels)
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    # cfg.model.moe_permute_fusion = True  # Already set above
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"  # OLMoE uses native

    # Memory saving (recompute & offloading) - already set in OlMoEModelProvider
    # cfg.model.recompute_granularity = "selective"
    # cfg.model.recompute_modules = None
    cfg.model.apply_rope_fusion = False  # Set to True for RoPE fusion (requires experimental flag)
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - OLMoE uses custom MixedPrecisionConfig (NOT "bf16_mixed" string)
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
    # FP8 settings (commented - enable if using FP8)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Communication overlap
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = False  # OLMoE doesn't have shared experts

    # Checkpoint config
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config (DIFFERENT: grad_reduce_in_fp32=False)
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    if cfg.model.apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True  # for rope fusion

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    return cfg
