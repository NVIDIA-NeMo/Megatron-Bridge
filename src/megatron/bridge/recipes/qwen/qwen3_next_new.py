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
Qwen3-Next recipe using the new flattened layout with _pretrain_common.

This file demonstrates the new pattern where recipes:
1. Call _pretrain_common() to get base config
2. Override model-specific settings directly on the returned config

Qwen3-Next is an MoE model with Multi-Token Prediction (MTP) support.
MoE models require additional settings beyond dense models:
- Expert parallelism (expert_model_parallel_size)
- Token dispatcher settings (moe_token_dispatcher_type, moe_flex_dispatcher_backend)
- MoE kernel selections (moe_router_fusion, moe_permute_fusion, moe_grouped_gemm)
- MoE overlap settings (moe_shared_expert_overlap, overlap_moe_expert_parallel_comm)
- MTP support (mtp_num_layers, mtp_loss_scaling_factor)
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.config import ConfigContainer


def qwen3_next_80b_a3b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3-Next 80B-A3B.

    Recommended parallelism: TP=1, PP=4, EP=8.
    Note: Qwen3-Next supports Multi-Token Prediction (MTP) with mtp_num_layers and mtp_loss_scaling_factor.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct").to_megatron_provider(load_weights=False)

    # Tokenizer (--tokenizer-model)
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.mmap_bin_files = False  # Qwen3-Next specific setting

    # Parallelism settings (MoE-specific: includes expert_model_parallel_size)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None  # Custom pipeline layout, None uses default
    cfg.model.pipeline_dtype = torch.bfloat16  # Required for PP > 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8  # MoE-specific: Expert parallelism
    cfg.model.expert_tensor_parallel_size = 1  # MoE-specific: Expert tensor parallelism (default from qwen3_next.py)
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # Multi-Token Prediction (MTP) settings - Qwen3-Next specific
    cfg.model.mtp_num_layers = 1  # Number of MTP layers (0 to disable)
    cfg.model.mtp_loss_scaling_factor = 0.1  # Loss scaling factor for MTP

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"  # Options: alltoall, allgather, flex
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # Options: None, deepep, hybridep (default from TransformerConfig)
    cfg.model.moe_hybridep_num_sms = 16  # Number of SMs for hybridep backend (default from TransformerConfig)

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # Scheduler config - Qwen3-Next specific
    cfg.scheduler.no_weight_decay_cond_type = "qwen3_next"

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections (includes MoE-specific kernels)
    cfg.model.attention_backend = None  # None means auto selection
    cfg.model.moe_router_fusion = False  # MoE-specific: Fuse router computation
    cfg.model.moe_permute_fusion = True  # MoE-specific: Fuse permute operations (set in qwen3_next.py)
    cfg.model.moe_grouped_gemm = True  # MoE-specific: Use grouped GEMM for experts (set in qwen3_next.py)
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"  # Qwen3-Next uses native (model default)

    # Memory saving (recompute & offloading) - ENABLED for 80B with selective recompute
    cfg.model.recompute_granularity = "selective"  # Qwen3-Next uses selective recompute
    cfg.model.recompute_modules = ["layernorm", "moe", "moe_act"]  # Qwen3-Next specific modules
    cfg.model.recompute_method = None  # Not used for selective recompute
    cfg.model.recompute_num_layers = None  # Not used for selective recompute
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False  # Pad router for FP8 alignment, MoE FP8 setting

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap (default None, can pass CommOverlapConfig for advanced overlap)
    # cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)  # Uncomment to enable
    # cfg.comm_overlap.delay_wgrad_compute = False  # Delay wgrad compute for overlap
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False  # MoE-specific: Overlap EP communication
    cfg.model.moe_shared_expert_overlap = False  # Overlap shared expert computation

    # Checkpoint config (paths set in _pretrain_common)
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False  # Force load balancing in router

    return cfg
