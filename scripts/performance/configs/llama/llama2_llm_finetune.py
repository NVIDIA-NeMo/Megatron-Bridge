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

import logging
import torch

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.llama import (
    llama2_70b_peft_config,
)
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
)
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


# Llama2 70B Finetune configs (MLPerf) --------------------------------------------------

def set_llama2_70b_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Llama 2 70B configs."""
    cfg.optimizer.bf16 = True
    cfg.optimizer.clip_grad = 0.3
    cfg.optimizer.fp8_recipe = "delayed"
    cfg.optimizer.lr = 0.0005
    cfg.optimizer.min_lr = 0
    cfg.optimizer.overlap_params = True
    cfg.optimizer.params_dtype = torch.bfloat16
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.weight_decay = 0.0001
    cfg.ddp.bucket_size = 45000000
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.hidden_size = 8192
    cfg.model.ffn_hidden_size = 28672
    cfg.model.num_layers = 80
    cfg.model.num_attention_heads = 64
    cfg.model.num_query_groups = 8
    cfg.model.kv_channels = 128
    cfg.model.normalization = "RMSNorm"
    cfg.model.gated_linear_unit = True
    cfg.model.add_bias_linear = False
    cfg.model.bf16 = True
    cfg.model.autocast_dtype = torch.bfloat16
    cfg.model.params_dtype = torch.bfloat16
    cfg.model.fp8_amax_compute_algo = "max"
    cfg.model.fp8_amax_history_len = 4
    cfg.model.fp8_dot_product_attention = True
    cfg.model.num_layers_at_start_in_bf16 = 0
    cfg.model.num_layers_at_end_in_bf16 = 0
    cfg.model.apply_rope_fusion = True
    cfg.model.fused_single_qkv_rope = 1
    cfg.model.bias_activation_fusion = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.masked_softmax_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.persist_layer_norm = True
    cfg.model.recompute_modules = ['core_attn']
    cfg.model.use_transformer_engine_op_fuser = 1
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True
    cfg.model.cp_comm_type = "a2a"
    cfg.model.cpu_offloading = False
    cfg.model.cuda_graph_modules = []
    cfg.model.cuda_graph_warmup_steps = 5
    cfg.model.deallocate_pipeline_outputs = True
    cfg.model.disable_parameter_transpose_cache = True
    cfg.model.attention_dropout = 0.0
    cfg.model.hidden_dropout = 0.0
    cfg.model.attention_softmax_in_fp32 = False
    cfg.model.embedding_init_method_std = 0.02
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.microbatch_group_size_per_vp_stage = 1
    cfg.model.pipeline_dtype = None
    cfg.mixed_precision.fp8_amax_compute_algo = "max"
    cfg.mixed_precision.fp8_amax_history_len = 4
    cfg.mixed_precision.fp8_dot_product_attention = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.mixed_precision.pipeline_dtype = None
    cfg.train.manual_gc = True
    cfg.train.manual_gc_eval = False
    cfg.train.manual_gc_interval = 10000
    cfg.train.skip_sync_grad_norm_across_mp = True
    cfg.train.check_optimizer_step_success = False
    cfg.train.decrease_batch_size_if_needed = False
    cfg.train.empty_unused_memory_level = 0
    cfg.validation.eval_micro_batch_size = 1
    cfg.validation.full_validation = False
    cfg.validation.multiple_validation_sets = False
    cfg.train.skip_train = False
    cfg.train.test_mode = False
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.scheduler.lr_warmup_fraction = 0.0
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_warmup_steps = 0.0
    cfg.scheduler.start_weight_decay = 0.0001
    cfg.scheduler.end_weight_decay = 0.0001
    cfg.scheduler.weight_decay_incr_style = "constant"
    cfg.scheduler.lr_wsd_decay_style = "exponential"
    cfg.scheduler.override_opt_param_scheduler = False
    cfg.scheduler.use_checkpoint_opt_param_scheduler = False
    cfg.dataset.seq_length = 8192
    cfg.dataset.enable_offline_packing = True
    cfg.dataset.create_attention_mask = False
    cfg.dataset.dataloader_type = "batch"
    cfg.dataset.data_sharding = True
    cfg.dataset.drop_last = True
    cfg.dataset.persistent_workers = True
    cfg.dataset.pin_memory = True
    cfg.dataset.memmap_workers = 1
    cfg.dataset.do_validation = True
    cfg.dataset.do_test = False
    cfg.checkpoint.finetune = True
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.load_optim = False
    cfg.checkpoint.load_rng = False
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tensor_model_parallel_size = 1

def get_llama2_70b_precision_config(compute_dtype: str):
    """Get the precision configs for the given compute dtype and FP8 recipe."""
    precision_config = get_precision_config(compute_dtype)
    precision_config.fp4_param = False
    precision_config.fp4_param_gather = False
    precision_config.fp8_param = False
    precision_config.fp8_param_gather = False
    precision_config.reuse_grad_buf_for_mxfp8_param_ag = False
    precision_config.num_layers_at_start_in_bf16 = 0
    precision_config.num_layers_at_end_in_bf16 = 0
    precision_config.first_last_layers_bf16 = False
    if compute_dtype == "fp8_ds":
        precision_config.fp8_param = True
        precision_config.fp8_param_gather = True
    return precision_config

def llama2_70b_lora_config_gb200(precision: str = "fp8_ds", config_variant: str = "v1") -> ConfigContainer:
    """GB200, MLPerf config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama2_70b",
        task="lora",
        gpu="gb200",
        compute_dtype=precision.upper(),
        config_variant=config_variant,
    )
    cfg = llama2_70b_peft_config(peft_scheme="lora")
    set_workload_base_configs(cfg, base_cfg)
    precision_config = get_llama2_70b_precision_config(precision)
    cfg.mixed_precision = precision_config
    set_llama2_70b_common_configs(cfg)

    # 4 GPUs
    if precision == "fp8_ds" and config_variant == "v1":
        cfg.model.cpu_offloading = True
        cfg.validation.eval_global_batch_size = 4
        cfg.validation.eval_interval = 48
        cfg.validation.eval_iters = 44
        cfg.validation.start_at_eval_iter = 192
        cfg.scheduler.lr_decay_iters = 800
        cfg.scheduler.lr_decay_steps = 6400
        cfg.scheduler.wd_incr_steps = 6400
        cfg.dataset.max_train_samples = 6432
        cfg.dataset.num_workers = 4
        cfg.dataset.seed = 30339
        cfg.rng.seed = 30339
    # 8 GPUs
    elif precision == "nvfp4" and config_variant == "v1":
        cfg.optimizer.lr = 0.0006
        cfg.validation.eval_global_batch_size = 4
        cfg.validation.eval_interval = 48
        cfg.validation.eval_iters = 44
        cfg.validation.start_at_eval_iter = 192
        cfg.scheduler.lr_decay_iters = 700
        cfg.scheduler.lr_decay_steps = 5600
        cfg.scheduler.wd_incr_steps = 5600
        cfg.dataset.max_train_samples = 5628
        cfg.dataset.num_workers = 4
        cfg.dataset.seed = 23829
        cfg.rng.seed = 23829
    elif precision == "fp8_ds" and config_variant == "v2":
        cfg.model.cpu_offloading = True
        cfg.validation.eval_global_batch_size = 8
        cfg.validation.eval_interval = 48
        cfg.validation.eval_iters = 22
        cfg.validation.start_at_eval_iter = 192
        cfg.scheduler.lr_decay_iters = 800
        cfg.scheduler.lr_decay_steps = 6400
        cfg.scheduler.wd_incr_steps = 6400
        cfg.dataset.max_train_samples = 6432
        cfg.dataset.num_workers = 4
        cfg.dataset.seed = 27208
        cfg.rng.seed = 27208
    # 72 GPUs
    elif precision == "fp8_ds" and config_variant == "v3":
        cfg.validation.eval_global_batch_size = 36
        cfg.validation.eval_interval = 43
        cfg.validation.eval_iters = 5
        cfg.validation.start_at_eval_iter = 172
        cfg.scheduler.lr_decay_iters = 800
        cfg.scheduler.lr_decay_steps = 7200
        cfg.scheduler.wd_incr_steps = 7200
        cfg.dataset.max_train_samples = 7236
        cfg.dataset.num_workers = 2
        cfg.dataset.seed = 16584
        cfg.rng.seed = 16584
    # 512 GPUs
    elif precision == "fp8_ds" and config_variant == "v4":
        cfg.optimizer.lr = 0.0006
        cfg.validation.eval_global_batch_size = 64
        cfg.validation.eval_interval = 6
        cfg.validation.eval_iters = 3
        cfg.validation.start_at_eval_iter = 66
        cfg.scheduler.lr_decay_iters = 600
        cfg.scheduler.lr_decay_steps = 38400
        cfg.scheduler.wd_incr_steps = 38400
        cfg.dataset.max_train_samples = 38592
        cfg.dataset.num_workers = 2
        cfg.dataset.seed = 22205
        cfg.rng.seed = 22205
    return cfg

def llama2_70b_lora_config_gb300(precision: str = "fp8_ds", config_variant: str = "v1") -> ConfigContainer:
    """GB300, MLPerf config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama2_70b",
        task="lora",
        gpu="gb300",
        compute_dtype=precision.upper(),
        config_variant=config_variant,
    )
    cfg = llama2_70b_peft_config(peft_scheme="lora")
    set_workload_base_configs(cfg, base_cfg)
    precision_config = get_llama2_70b_precision_config(precision)
    cfg.mixed_precision = precision_config
    set_llama2_70b_common_configs(cfg)

    # 4 GPUs
    if config_variant == "v1":
        cfg.model.cuda_graph_warmup_steps = 1
        cfg.validation.eval_global_batch_size = 4
        cfg.validation.eval_interval = 48
        cfg.validation.eval_iters = 44
        cfg.validation.start_at_eval_iter = 192
        cfg.scheduler.lr_decay_iters = 800
        cfg.scheduler.lr_decay_steps = 6400
        cfg.scheduler.wd_incr_steps = 6400
        cfg.dataset.max_train_samples = 6432
        cfg.dataset.num_workers = 4
        cfg.dataset.seed = 10710
        cfg.rng.seed = 10710
    # 8 GPUs
    elif config_variant == "v2":
        cfg.validation.eval_global_batch_size = 8
        cfg.validation.eval_interval = 48
        cfg.validation.eval_iters = 22
        cfg.validation.start_at_eval_iter = 192
        cfg.scheduler.lr_decay_iters = 800
        cfg.scheduler.lr_decay_steps = 6400
        cfg.scheduler.wd_incr_steps = 6400
        cfg.dataset.max_train_samples = 6432
        cfg.dataset.num_workers = 4
        cfg.dataset.seed = 22699
        cfg.rng.seed = 22699
    # 72 GPUs
    elif config_variant == "v3":
        cfg.validation.eval_global_batch_size = 36
        cfg.validation.eval_interval = 43
        cfg.validation.eval_iters = 5
        cfg.validation.start_at_eval_iter = 172
        cfg.scheduler.lr_decay_iters = 800
        cfg.scheduler.lr_decay_steps = 7200
        cfg.scheduler.wd_incr_steps = 7200
        cfg.dataset.max_train_samples = 7236
        cfg.dataset.num_workers = 2
        cfg.dataset.seed = 14954
        cfg.rng.seed = 14954
    # 512 GPUs
    elif config_variant == "v4":
        cfg.optimizer.lr = 0.0006
        cfg.validation.eval_global_batch_size = 64
        cfg.validation.eval_interval = 6
        cfg.validation.eval_iters = 3
        cfg.validation.start_at_eval_iter = 66
        cfg.scheduler.lr_decay_iters = 600
        cfg.scheduler.lr_decay_steps = 38400
        cfg.scheduler.wd_incr_steps = 38400
        cfg.dataset.max_train_samples = 38592
        cfg.dataset.num_workers = 2
        cfg.dataset.seed = 8353
        cfg.rng.seed = 8353
    return cfg

