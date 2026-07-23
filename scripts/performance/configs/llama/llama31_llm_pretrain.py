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

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.llama import (
    llama31_405b_pretrain_config,
    llama31_8b_pretrain_config,
)
from megatron.bridge.training.comm_overlap import (
    userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def set_llama31_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Llama3.1 configs."""
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False


def disable_param_gather_overlap(cfg: ConfigContainer) -> None:
    """
    Disable parameter-gather overlap to reduce training peak memory and avoid OOM.
    Note: This is a workaround and should be removed once the issue is fixed.
    See: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/3714
    """
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather = False
    cfg.comm_overlap.overlap_param_gather = False
    cfg.comm_overlap.align_param_gather = False


def llama31_405b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_405b",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    if precision == "bf16":
        comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
    else:
        comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = precision_config
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    return cfg


def llama31_405b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_405b",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    if precision == "bf16":
        comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
    else:
        comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = precision_config
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors
        cfg.ddp.num_distributed_optimizer_instances = 2

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap
    if precision == "nvfp4" and config_variant.lower() == "v2":
        disable_param_gather_overlap(cfg)

    return cfg


def llama31_405b_pretrain_config_vr200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """VR200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_405b",
        gpu="vr200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    if precision == "bf16":
        comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
    else:
        comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = precision_config
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors
        cfg.ddp.num_distributed_optimizer_instances = 2

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    return cfg


def llama31_405b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_405b",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    if precision == "bf16":
        comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
    else:
        comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = precision_config
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    return cfg


def llama31_405b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_405b",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    if precision == "bf16":
        comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
    else:
        comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = precision_config
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg
    cfg.comm_overlap.tp_comm_overlap = False if precision == "nvfp4" else cfg.comm_overlap.tp_comm_overlap

    return cfg


def llama31_405b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_405b",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    if precision == "bf16":
        comm_overlap_cfg = userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192
    else:
        comm_overlap_cfg = userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192

    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = precision_config
    set_llama31_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.tp_comm_overlap_cfg = comm_overlap_cfg

    return cfg

# MLPerf Llama3.1 8B configs ---------------------------------------------------------

def set_llama31_8b_common_configs(cfg: ConfigContainer) -> None:
    """Set common configurations for all Llama3.1 8B configs."""
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-05
    cfg.optimizer.bf16 = True
    cfg.optimizer.fp8_recipe = 'tensorwise'
    cfg.optimizer.overlap_param_gather = True
    cfg.optimizer.params_dtype = torch.bfloat16
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.bucket_size = 768000000
    cfg.ddp.data_parallel_sharding_strategy = 'optim_grads_params'
    cfg.ddp.delay_wgrad_compute = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.model.add_bias_linear = False
    cfg.model.apply_rope_fusion = True
    cfg.model.attention_dropout = 0.0
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.hidden_dropout = 0.0
    cfg.model.hidden_size = 4096
    cfg.model.kv_channels = 128
    cfg.model.masked_softmax_fusion = True
    cfg.model.microbatch_group_size_per_vp_stage = 1
    cfg.model.normalization = 'RMSNorm'
    cfg.model.num_attention_heads = 32
    cfg.model.num_layers = 32
    cfg.model.num_layers_at_end_in_bf16 = 0
    cfg.model.num_layers_at_start_in_bf16 = 0
    cfg.model.num_query_groups = 8
    cfg.model.overlap_p2p_comm = True
    cfg.model.params_dtype = torch.bfloat16
    cfg.model.persist_layer_norm = True
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.recompute_modules = ['core_attn']
    cfg.model.tp_only_amax_red = True
    cfg.model.use_te_rng_tracker = True
    cfg.model.wgrad_deferral_limit = 50
    cfg.train.check_optimizer_step_success = False
    cfg.train.decrease_batch_size_if_needed = False
    cfg.train.empty_unused_memory_level = 0
    cfg.train.exit_signal_handler = False
    cfg.train.exit_signal_handler_for_dataloader = False
    cfg.train.exit_signal_handler_for_training = False
    cfg.train.iterations_to_skip = []
    cfg.train.manual_gc = True
    cfg.train.manual_gc_eval = False
    cfg.train.manual_gc_interval = 500
    cfg.train.skip_sync_grad_norm_across_mp = True
    cfg.eval.eval_iters = 64
    cfg.eval.full_validation = False
    cfg.eval.multiple_validation_sets = False
    cfg.eval.skip_train = False
    cfg.eval.test_mode = False
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = 'cosine'
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.lr_warmup_samples = 0
    cfg.scheduler.lr_wsd_decay_style = 'exponential'
    cfg.scheduler.override_opt_param_scheduler = False
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.use_checkpoint_opt_param_scheduler = False
    cfg.scheduler.weight_decay_incr_style = 'constant'
    cfg.dataset.add_extra_token_to_sequence = True
    cfg.dataset.allow_ambiguous_pad_tokens = False
    cfg.dataset.create_attention_mask = False
    cfg.dataset.data_parallel_size = 1
    cfg.dataset.data_sharding = True
    cfg.dataset.dataloader_type = 'single'
    cfg.dataset.defer_npy_index_mmap = True
    cfg.dataset.drop_last = True
    cfg.dataset.drop_last_partial_validation_sequence = True
    cfg.dataset.eod_mask_loss = False
    cfg.dataset.fast_cache_load = True
    cfg.dataset.hybrid_context_parallel = False
    cfg.dataset.inter_document_masking = False
    cfg.dataset.mid_level_dataset_surplus = 0.005
    cfg.dataset.mmap_bin_files = True
    cfg.dataset.mock = False
    cfg.dataset.num_dataset_builder_threads = 1
    cfg.dataset.num_workers = 8
    cfg.dataset.persistent_workers = True
    cfg.dataset.pin_memory = True
    cfg.dataset.reset_attention_mask = False
    cfg.dataset.reset_position_ids = False
    cfg.dataset.seq_length = 8192
    cfg.dataset.sequence_length = 8192
    cfg.dataset.sequence_parallel_size = 0
    also_save_hf_checkpoint: False
    cfg.checkpoint.async_ckpt_cpu_priority = 10
    cfg.checkpoint.async_ckpt_io_priority = 3
    cfg.checkpoint.async_ckpt_use_cpu_shm = False
    cfg.checkpoint.async_save = False
    cfg.checkpoint.async_strategy = 'mcore'
    cfg.checkpoint.async_write_results_mp_mode = 'fork'
    cfg.checkpoint.auto_detect_ckpt_format = False
    cfg.checkpoint.ckpt_assume_constant_structure = False
    cfg.checkpoint.ckpt_convert_update_legacy_dist_opt_format = False
    cfg.checkpoint.ckpt_format = 'torch_dist'
    cfg.checkpoint.ckpt_fully_parallel_load_exchange_algo = 'broadcast'
    cfg.checkpoint.ckpt_fully_parallel_load_process_group = 'dp'
    cfg.checkpoint.ckpt_fully_parallel_save_process_group = 'dp'
    cfg.checkpoint.ckpt_load_validate_sharding_integrity = True
    cfg.checkpoint.dist_ckpt_optim_fully_reshardable = False
    cfg.checkpoint.dist_ckpt_save_pre_mcore_014 = False
    cfg.checkpoint.dist_ckpt_strictness = 'log_all'
    cfg.checkpoint.dist_ckpt_workers = 1
    cfg.checkpoint.distrib_optim_fully_reshardable_mem_efficient = False
    cfg.checkpoint.exit_on_missing_checkpoint = False
    cfg.checkpoint.finetune = False
    cfg.checkpoint.fully_parallel_load = True
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.hf_distributed_save = False
    cfg.checkpoint.hf_save_every_n_ranks = 1
    cfg.checkpoint.hf_trust_remote_code = False
    cfg.checkpoint.load_optim = False
    cfg.checkpoint.load_rng = False
    cfg.checkpoint.most_recent_k = -1
    cfg.checkpoint.non_persistent_local_ckpt_algo = 'fully_parallel'
    cfg.checkpoint.replication = False
    cfg.checkpoint.replication_factor = 2
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.save_rng = True
    cfg.checkpoint.save_tokenizer_assets = True
    cfg.checkpoint.storage_writers_per_rank = 1
    cfg.checkpoint.strict_fsdp_dtensor_load = False
    cfg.checkpoint.use_checkpoint_args = False
    cfg.checkpoint.use_mp_args_from_checkpoint_args = False
    cfg.checkpoint.use_persistent_ckpt_worker = True
    cfg.checkpoint.use_tokenizer_model_from_checkpoint_args = True
    cfg.checkpoint.verify_integrity = False
    cfg.logger.barrier_with_L1_time = True
    cfg.logger.filter_warnings = True
    cfg.logger.log_device_memory_used = False
    cfg.logger.log_energy = False
    cfg.logger.log_interval = 1200001
    cfg.logger.log_l2_norm_grad_to_tensorboard = False
    cfg.logger.log_loss_scale_to_tensorboard = True
    cfg.logger.log_max_attention_logit = False
    cfg.logger.log_memory_to_tensorboard = False
    cfg.logger.log_num_zeros_in_grad = False
    cfg.logger.log_params_norm = False
    cfg.logger.log_progress = False
    cfg.logger.log_runtime_to_tensorboard = False
    cfg.logger.log_throughput = False
    cfg.logger.log_throughput_to_tensorboard = False
    cfg.logger.log_timers_to_tensorboard = False
    cfg.logger.log_validation_ppl_to_tensorboard = False
    cfg.logger.log_world_size_to_tensorboard = False
    cfg.logger.logging_level = 20
    cfg.logger.mlflow_log_artifacts = True
    cfg.logger.moe_routing_trace_capture_hidden_states = False
    cfg.logger.moe_routing_trace_capture_logits = False
    cfg.logger.moe_routing_trace_dump_weights = False
    cfg.logger.runtime_time_unit = 'hours'
    cfg.logger.set_level_for_all_loggers = False
    cfg.logger.skip_train_metrics_log = True
    cfg.logger.tensorboard_log_interval = 1
    cfg.logger.tensorboard_queue_size = 1000
    cfg.logger.throughput_window_size = 100
    cfg.logger.timing_log_level = -1
    cfg.logger.timing_log_option = 'minmax'
    cfg.tokenizer.force_system_message = False
    cfg.tokenizer.hf_tokenizer_kwargs = {'use_fast': True}
    cfg.tokenizer.make_vocab_size_divisible_by = 1
    cfg.tokenizer.null_tokenizer_pad_id = -1
    cfg.tokenizer.pad_vocab_size = False
    cfg.tokenizer.rank = 0
    cfg.tokenizer.sp_tokenizer_kwargs = {}
    cfg.tokenizer.tensor_model_parallel_size = 1
    cfg.tokenizer.tiktoken_num_special_tokens = 1000
    cfg.tokenizer.tokenizer_hf_no_include_special_tokens = False
    cfg.tokenizer.tokenizer_hf_no_use_fast = False
    cfg.tokenizer.tokenizer_sentencepiece_ignore_extra_whitespaces = True
    cfg.tokenizer.tokenizer_sentencepiece_legacy = False
    cfg.tokenizer.tokenizer_type = 'HuggingFaceTokenizer'
    cfg.tokenizer.trust_remote_code = False
    cfg.tokenizer.vocab_extra_ids = 0
    cfg.rng.data_parallel_random_init: False
    cfg.rng.inference_rng_tracker = False
    cfg.rng.te_rng_tracker = True


def llama31_8b_pretrain_config_gb300(
    precision: str = "fp8_cs", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_8b",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = llama31_8b_pretrain_config()
    cfg.mixed_precision = precision_config
    cfg.mixed_precision.num_layers_at_start_in_bf16 = 0
    cfg.mixed_precision.num_layers_at_end_in_bf16 = 0
    cfg.mixed_precision.first_last_layers_bf16 = False
    cfg.mixed_precision.fp4_param_gather = False
    set_workload_base_configs(cfg, base_cfg)
    set_llama31_8b_common_configs(cfg)

    if config_variant.lower() == "v1" and precision.lower() == "nvfp4":
        cfg.optimizer.lr = 0.0004
        cfg.optimizer.min_lr = 4e-05
        cfg.model.tp_comm_bootstrap_backend = 'mpi'
        cfg.model.fp8_dot_product_attention = True
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.mixed_precision.fp8_dot_product_attention = True
        cfg.validation.eval_interval = 768
        cfg.validation.eval_iters = 64
        cfg.scheduler.lr_decay_iters = 1199984
        cfg.scheduler.lr_decay_steps = 19199744
        cfg.scheduler.lr_warmup_iters = 16
        cfg.scheduler.lr_warmup_steps = 256
        cfg.scheduler.wd_incr_steps = 19200000
        cfg.load_main_params_from_ckpt = False
    elif config_variant.lower() == "v2" and precision.lower() == "nvfp4":
        cfg.optimizer.lr = 0.0008
        cfg.optimizer.min_lr = 8e-05
        cfg.model.tp_comm_bootstrap_backend = 'mpi'
        cfg.model.fp8_dot_product_attention = True
        cfg.mixed_precision.fp8_dot_product_attention = True
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.validation.eval_interval = 171
        cfg.validation.eval_iters = 15
        cfg.scheduler.lr_decay_iters = 1199936
        cfg.scheduler.lr_decay_steps = 86395392
        cfg.scheduler.lr_warmup_iters = 64
        cfg.scheduler.lr_warmup_steps = 4608
        cfg.scheduler.wd_incr_steps = 86400000
        cfg.load_main_params_from_ckpt = False
    elif config_variant.lower() == "v1" and precision.lower() == "fp8_cs":
        cfg.optimizer.lr = 0.0008
        cfg.optimizer.min_lr = 8e-05
        cfg.ddp.fp8_param_gather = True
        cfg.model.tp_comm_bootstrap_backend = 'gloo'
        cfg.model.tp_comm_overlap = True
        cfg.validation.eval_interval = 192
        cfg.validation.eval_iters = 16
        cfg.scheduler.lr_decay_iters = 1199936
        cfg.scheduler.lr_decay_steps = 76795904
        cfg.scheduler.lr_warmup_iters = 64
        cfg.scheduler.lr_warmup_steps = 4096
        cfg.scheduler.wd_incr_steps = 76800000
        cfg.load_main_params_from_ckpt = True

    return cfg


def llama31_8b_pretrain_config_gb200(
    precision: str = "fp8_cs", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama31_8b",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = llama31_8b_pretrain_config()
    cfg.mixed_precision = precision_config
    set_workload_base_configs(cfg, base_cfg)
    set_llama31_8b_common_configs(cfg)

    if config_variant.lower() == "v1" and precision.lower() == "nvfp4":
        cfg.optimizer.lr = 0.0004
        cfg.optimizer.min_lr = 4e-05
        cfg.model.tp_comm_bootstrap_backend = 'mpi'
        cfg.model.fp8_dot_product_attention = True
        cfg.mixed_precision.fp8_dot_product_attention = True
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.validation.eval_interval = 768
        cfg.validation.eval_iters = 64
        cfg.scheduler.lr_decay_iters = 1199984
        cfg.scheduler.lr_decay_steps = 19199744
        cfg.scheduler.lr_warmup_iters = 16
        cfg.scheduler.lr_warmup_steps = 256
        cfg.scheduler.wd_incr_steps = 19200000
        cfg.load_main_params_from_ckpt = False
    elif config_variant.lower() == "v2" and precision.lower() == "nvfp4":
        cfg.optimizer.lr = 0.0008
        cfg.optimizer.min_lr = 8e-05
        cfg.model.tp_comm_bootstrap_backend = 'mpi'
        cfg.model.fp8_dot_product_attention = True
        cfg.mixed_precision.fp8_dot_product_attention = True
        cfg.model.use_transformer_engine_op_fuser = True
        cfg.validation.eval_interval = 171
        cfg.validation.eval_iters = 15
        cfg.scheduler.lr_decay_iters = 1199936
        cfg.scheduler.lr_decay_steps = 86395392
        cfg.scheduler.lr_warmup_iters = 64
        cfg.scheduler.lr_warmup_steps = 4608
        cfg.scheduler.wd_incr_steps = 86400000
        cfg.load_main_params_from_ckpt = False
    elif config_variant.lower() == "v1" and precision.lower() == "fp8_cs":
        cfg.optimizer.lr = 0.0008
        cfg.optimizer.min_lr = 8e-05
        cfg.ddp.fp8_param_gather = True
        cfg.model.tp_comm_bootstrap_backend = 'gloo'
        cfg.model.tp_comm_overlap = True
        cfg.validation.eval_interval = 192
        cfg.validation.eval_iters = 16
        cfg.scheduler.lr_decay_iters = 1199936
        cfg.scheduler.lr_decay_steps = 76795904
        cfg.scheduler.lr_warmup_iters = 64
        cfg.scheduler.lr_warmup_steps = 4096
        cfg.scheduler.wd_incr_steps = 76800000
        cfg.load_main_params_from_ckpt = True

    return cfg
