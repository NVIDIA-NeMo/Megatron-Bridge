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
"""GB200 performance recipes for DeepSeek V4."""

from megatron.bridge.models.deepseek.deepseek_v4_bridge import set_deepseek_v4_pipeline_model_parallel_layout
from megatron.bridge.perf_recipes.deepseek.common import (
    ConfigContainer,
    _benchmark_common,
    _enable_deepseek_precision_aware_optimizer,
    _perf_precision,
)
from megatron.bridge.recipes.deepseek.deepseek_v4 import deepseek_v4_flash_pretrain_config


def deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_paged_stash_config() -> ConfigContainer:
    """DeepSeek V4 Flash pretrain: 64× GB200, MXFP8 and full-iteration CUDA graphs."""
    cfg = deepseek_v4_flash_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _benchmark_common(cfg, cross_entropy_impl="native")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    cfg.model.make_vocab_size_divisible_by = 3232
    cfg.model.rotary_scaling_factor = 4
    cfg.model.experimental_attention_variant = "dsv4_hybrid"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_load_balancing_type = "seq_aux_loss"
    cfg.model.moe_aux_loss_coeff = 1e-4
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.moe_router_fusion = True
    cfg.model.dsa_indexer_loss_coeff = 0.01
    cfg.model.dsa_indexer_use_sparse_loss = True
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_paged_stash = True
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.moe_paged_stash_page_size = 64
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 0.0
    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = "full_iteration"
    cfg.model.use_te_rng_tracker = True
    cfg.model.offload_modules = []

    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 2048
    cfg.train.manual_gc_interval = 10

    cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
    cfg.ddp.check_for_nan_in_grad = False
    cfg.rng.te_rng_tracker = True

    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)
    _enable_deepseek_precision_aware_optimizer(cfg)
    return cfg
