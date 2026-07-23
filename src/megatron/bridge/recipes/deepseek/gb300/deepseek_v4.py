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
"""GB300 recipes for DeepSeek V4 Pro."""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    deepseek_v4_supports_blackwell_fused_kernels,
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.deepseek.h100.deepseek_v4 import _deepseek_v4_mxfp8_quant_recipe
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed


DEEPSEEK_V4_PRO_HF_PATH = "deepseek-ai/DeepSeek-V4-Pro"


def deepseek_v4_pro_pretrain_32gpu_gb300_bf16_config() -> ConfigContainer:
    """Return the DeepSeek-V4-Pro GB300 pre-training base config.

    DeepSeek-V4 still requires a compatible Megatron-Core development commit;
    the Megatron-Core commit pinned by Megatron Bridge ``main`` is not a
    supported runtime for this recipe.
    """
    cfg = _pretrain_common()
    cfg.model = AutoBridge.from_hf_pretrained(
        DEEPSEEK_V4_PRO_HF_PATH, trust_remote_code=True
    ).to_megatron_provider(load_weights=False)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.params_dtype = torch.bfloat16

    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None
    cfg.model.apply_dsa_kernel_fusion = False
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_mhc = deepseek_v4_supports_blackwell_fused_kernels()
    cfg.model.dsa_indexer_loss_coeff = 0.0
    cfg.model.dsa_indexer_use_sparse_loss = False

    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act", "mhc"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = cfg.model.vocab_size
    cfg.tokenizer.make_vocab_size_divisible_by = cfg.model.make_vocab_size_divisible_by
    cfg.tokenizer.tensor_model_parallel_size = cfg.model.tensor_model_parallel_size
    cfg.tokenizer.rank = 0

    cfg.dataset.blend = None
    cfg.dataset.blend_per_split = None
    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8
    cfg.dataset.skip_getting_attention_mask_from_dataset = True
    cfg.dataset.dataloader_type = "single"

    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5
    cfg.validation.eval_interval = 2000
    cfg.validation.eval_iters = 32

    cfg.logger.log_interval = 10
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False
    cfg.dist.enable_megatron_core_experimental = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def deepseek_v4_pro_pretrain_32gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Return the DeepSeek-V4-Pro Adam + MXFP8 pre-training config."""
    cfg = deepseek_v4_pro_pretrain_32gpu_gb300_bf16_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.model.apply_dsa_kernel_fusion = False
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_mhc = deepseek_v4_supports_blackwell_fused_kernels()
    cfg.model.dsa_indexer_loss_coeff = 0.0
    cfg.model.dsa_indexer_use_sparse_loss = False
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act", "mhc"]
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=2000,
        lr_decay_iters=cfg.train.train_iters,
        max_lr=2.7e-4,
        min_lr=2.7e-5,
        weight_decay=0.1,
        clip_grad=1.0,
    )
    opt_cfg.use_precision_aware_optimizer = True
    opt_cfg.main_grads_dtype = torch.float32
    opt_cfg.main_params_dtype = torch.float32
    opt_cfg.exp_avg_dtype = torch.bfloat16
    opt_cfg.exp_avg_sq_dtype = torch.bfloat16
    opt_cfg.adam_beta1 = 0.9
    opt_cfg.adam_beta2 = 0.95
    opt_cfg.adam_eps = 1e-20

    scheduler_cfg.start_weight_decay = 0.1
    scheduler_cfg.end_weight_decay = 0.1
    scheduler_cfg.weight_decay_incr_style = "constant"

    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = True
    cfg.model.mtp_eval_in_bf16 = True
    cfg.model.quant_recipe = _deepseek_v4_mxfp8_quant_recipe()
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg
