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

"""llama benchmark overrides applied to library recipes.

The literal names preserve existing CLI selectors. Common and model overrides
retain their ordering because graph, RNG and precision policies are dependent.
"""

from functools import partial

import torch

from megatron.bridge.recipes.llama.llama2 import llama2_70b_peft_config
from megatron.bridge.recipes.llama.llama3 import (
    llama3_8b_pretrain_config,
    llama3_8b_sft_config,
    llama3_70b_peft_config,
    llama3_70b_pretrain_config,
    llama3_70b_sft_config,
    llama31_8b_pretrain_config,
    llama31_405b_pretrain_config,
)
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
    userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.perf_benchmark_overrides._common import _benchmark_common, _perf_precision
from megatron.bridge.training.perf_benchmark_overrides.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.utils.cuda_graph import clear_cuda_graph_modules


BENCHMARK_ENVIRONMENTS = {
    "llama3_70b_peft_8gpu_b200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "llama3_70b_pretrain_64gpu_b200_fp8cs": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "llama3_70b_pretrain_64gpu_b200_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
    },
    "llama3_70b_pretrain_64gpu_b200_nvfp4": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
        "NVTE_USE_FAST_MATH": 1,
    },
    "llama3_8b_pretrain_64gpu_b200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "llama3_8b_pretrain_8gpu_b200_nvfp4": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_USE_FAST_MATH": 1,
    },
    "llama3_70b_peft_8gpu_b300_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "llama3_70b_pretrain_64gpu_b300_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "llama3_70b_pretrain_64gpu_b300_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "llama3_70b_pretrain_64gpu_b300_nvfp4": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
        "NVTE_USE_FAST_MATH": 1,
    },
    "llama3_8b_pretrain_8gpu_b300_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "llama2_70b_peft_4gpu_gb200_fp8ds": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "llama3_70b_pretrain_32gpu_gb200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    },
    "llama3_70b_sft_32gpu_gb200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
    },
    "llama3_8b_sft_8gpu_gb200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    },
    "llama31_8b_pretrain_512gpu_gb200_fp8cs": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
        "NVTE_USE_FAST_MATH": 0,
    },
    "llama3_70b_pretrain_32gpu_gb300_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 1,
        "NCCL_CTA_POLICY": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    },
    "llama3_8b_pretrain_32gpu_gb300_nvfp4": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_USE_FAST_MATH": 1,
    },
    "llama3_70b_pretrain_64gpu_h100_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
    },
    "llama3_70b_sft_32gpu_h100_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
    },
}


def _with_global_batch_size(cfg: ConfigContainer, global_batch_size: int) -> ConfigContainer:
    cfg.train.global_batch_size = global_batch_size
    return cfg


def _llama_benchmark_common(cfg: ConfigContainer) -> None:
    """Apply legacy Llama benchmark defaults shared by the flat recipes."""
    te_rng_disabled = not cfg.rng.te_rng_tracker and not cfg.model.use_te_rng_tracker
    cuda_graph_impl = getattr(cfg.model, "cuda_graph_impl", None)

    _benchmark_common(cfg)

    cfg.model.moe_token_dispatcher_type = "alltoall"
    if cuda_graph_impl == "none" and te_rng_disabled:
        cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False


def _llama2_70b_precision_config(compute_dtype: str):
    """Get the precision configs for the given compute dtype and FP8 recipe."""
    precision_config = _perf_precision(compute_dtype)
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


def _llama2_benchmark_common(cfg: ConfigContainer) -> None:
    """Apply legacy Llama 2 benchmark defaults shared by the flat recipes."""
    cfg.optimizer.bf16 = True
    cfg.optimizer.clip_grad = 0.3
    cfg.optimizer.fp8_recipe = "delayed"
    cfg.optimizer.lr = 0.0005
    cfg.optimizer.min_lr = 0
    cfg.optimizer.overlap_param_gather = True
    cfg.optimizer.params_dtype = torch.bfloat16
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.weight_decay = 0.0001
    cfg.ddp.bucket_size = 45000000
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.check_for_nan_in_grad = False
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
    cfg.model.fused_single_qkv_rope = True
    cfg.model.bias_activation_fusion = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.masked_softmax_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.persist_layer_norm = True
    cfg.model.recompute_modules = ["core_attn"]
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True
    cfg.model.cp_comm_type = "a2a"
    cfg.model.cpu_offloading = False
    cfg.model.cuda_graph_modules = []
    cfg.model.cuda_graph_warmup_steps = 1
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
    cfg.tokenizer.use_tokenizer_vocab_size = False


def _llama31_8b_common(cfg: ConfigContainer) -> None:
    """Set common configurations for all Llama3.1 8B configs."""
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-05
    cfg.optimizer.bf16 = True
    cfg.optimizer.fp8_recipe = "tensorwise"
    cfg.optimizer.overlap_param_gather = True
    cfg.optimizer.params_dtype = torch.bfloat16
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.bucket_size = 768000000
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.delay_wgrad_compute = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.bucket_size = 768000000
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = False
    cfg.model.add_bias_linear = False
    cfg.model.apply_rope_fusion = True
    cfg.model.attention_dropout = 0.0
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.hidden_dropout = 0.0
    cfg.model.hidden_size = 4096
    cfg.model.kv_channels = 128
    cfg.model.masked_softmax_fusion = True
    cfg.model.microbatch_group_size_per_vp_stage = 1
    cfg.model.normalization = "RMSNorm"
    cfg.model.num_attention_heads = 32
    cfg.model.num_layers = 32
    cfg.model.num_layers_at_end_in_bf16 = 0
    cfg.model.num_layers_at_start_in_bf16 = 0
    cfg.model.num_query_groups = 8
    cfg.model.overlap_p2p_comm = True
    cfg.model.params_dtype = torch.bfloat16
    cfg.model.persist_layer_norm = True
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.recompute_modules = ["core_attn"]
    cfg.model.tp_only_amax_red = True
    cfg.model.use_te_rng_tracker = True
    cfg.model.wgrad_deferral_limit = 50
    cfg.model.batch_p2p_comm = False
    cfg.model.fused_single_qkv_rope = True
    cfg.model.cuda_graph_warmup_steps = 1
    cfg.model.vocab_size = 128256
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
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.lr_warmup_samples = 0
    cfg.scheduler.lr_wsd_decay_style = "exponential"
    cfg.scheduler.override_opt_param_scheduler = False
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.use_checkpoint_opt_param_scheduler = False
    cfg.scheduler.weight_decay_incr_style = "constant"
    cfg.dataset.add_extra_token_to_sequence = True
    cfg.dataset.allow_ambiguous_pad_tokens = False
    cfg.dataset.create_attention_mask = False
    cfg.dataset.data_parallel_size = 1
    cfg.dataset.data_sharding = True
    cfg.dataset.dataloader_type = "single"
    cfg.dataset.defer_npy_index_mmap = True
    cfg.dataset.drop_last = True
    cfg.dataset.drop_last_partial_validation_sequence = True
    cfg.dataset.dynamic_context_parallel = False
    cfg.dataset.eod_mask_loss = False
    cfg.dataset.fast_cache_load = True
    cfg.dataset.inter_document_masking = False
    cfg.dataset.mid_level_dataset_surplus = 0.005
    cfg.dataset.mmap_bin_files = True
    cfg.dataset.num_dataset_builder_threads = 1
    cfg.dataset.num_workers = 8
    cfg.dataset.persistent_workers = True
    cfg.dataset.pin_memory = True
    cfg.dataset.reset_attention_mask = False
    cfg.dataset.reset_position_ids = False
    cfg.dataset.seq_length = 8192
    cfg.dataset.sequence_length = 8192
    cfg.dataset.sequence_parallel_size = 0
    cfg.checkpoint.also_save_hf_checkpoint = False
    cfg.checkpoint.async_ckpt_cpu_priority = 10
    cfg.checkpoint.async_ckpt_io_priority = 3
    cfg.checkpoint.async_ckpt_use_cpu_shm = False
    cfg.checkpoint.async_save = False
    cfg.checkpoint.async_strategy = "mcore"
    cfg.checkpoint.async_write_results_mp_mode = "fork"
    cfg.checkpoint.auto_detect_ckpt_format = False
    cfg.checkpoint.ckpt_assume_constant_structure = False
    cfg.checkpoint.ckpt_convert_update_legacy_dist_opt_format = False
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.ckpt_fully_parallel_load_exchange_algo = "broadcast"
    cfg.checkpoint.ckpt_fully_parallel_load_process_group = "dp"
    cfg.checkpoint.ckpt_fully_parallel_save_process_group = "dp"
    cfg.checkpoint.ckpt_load_validate_sharding_integrity = True
    cfg.checkpoint.dist_ckpt_optim_fully_reshardable = False
    cfg.checkpoint.dist_ckpt_save_pre_mcore_014 = False
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
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
    cfg.checkpoint.non_persistent_local_ckpt_algo = "fully_parallel"
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
    cfg.logger.log_interval = 1
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
    cfg.logger.runtime_time_unit = "hours"
    cfg.logger.set_level_for_all_loggers = False
    cfg.logger.skip_train_metrics_log = False
    cfg.logger.tensorboard_log_interval = 1
    cfg.logger.tensorboard_queue_size = 1000
    cfg.logger.throughput_window_size = 100
    cfg.logger.timing_log_level = 0
    cfg.logger.timing_log_option = "minmax"
    cfg.tokenizer.force_system_message = False
    cfg.tokenizer.hf_tokenizer_kwargs = {"use_fast": True}
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
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.trust_remote_code = False
    cfg.tokenizer.use_tokenizer_vocab_size = False
    cfg.tokenizer.vocab_extra_ids = 0
    cfg.rng.data_parallel_random_init = False
    cfg.rng.inference_rng_tracker = False
    cfg.rng.te_rng_tracker = True
    cfg.ddp.align_param_gather = False
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    cfg.validation.full_validation = False
    cfg.validation.multiple_validation_sets = False
    cfg.validation.skip_train = False
    cfg.validation.test_mode = False


LIBRARY_RECIPES = {
    "llama3_70b_peft_8gpu_b200_bf16_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_peft_8gpu_b200_fp8cs_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_peft_8gpu_b200_fp8mx_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_pretrain_64gpu_b200_bf16_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_b200_fp8cs_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_b200_fp8mx_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_b200_nvfp4_config": llama3_70b_pretrain_config,
    "llama3_8b_pretrain_64gpu_b200_bf16_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_64gpu_b200_fp8cs_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_b200_bf16_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_b200_fp8cs_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_b200_fp8mx_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_b200_nvfp4_config": llama3_8b_pretrain_config,
    "llama31_405b_pretrain_128gpu_b200_bf16_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_128gpu_b200_fp8cs_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_128gpu_b200_fp8mx_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_128gpu_b200_nvfp4_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_b200_nvfp4_config": llama31_405b_pretrain_config,
    "llama3_70b_peft_8gpu_b300_bf16_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_peft_8gpu_b300_fp8cs_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_peft_8gpu_b300_fp8mx_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_pretrain_64gpu_b300_bf16_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_b300_fp8cs_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_b300_fp8mx_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_b300_nvfp4_config": llama3_70b_pretrain_config,
    "llama3_8b_pretrain_8gpu_b300_bf16_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_b300_fp8cs_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_b300_fp8mx_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_b300_nvfp4_config": llama3_8b_pretrain_config,
    "llama31_405b_pretrain_128gpu_b300_bf16_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_128gpu_b300_fp8cs_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_128gpu_b300_fp8mx_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_128gpu_b300_nvfp4_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_b300_bf16_config": llama31_405b_pretrain_config,
    "llama2_70b_peft_4gpu_gb200_fp8ds_config": partial(llama2_70b_peft_config, peft_scheme="lora"),
    "llama2_70b_peft_512gpu_gb200_fp8ds_config": partial(llama2_70b_peft_config, peft_scheme="lora"),
    "llama2_70b_peft_72gpu_gb200_fp8ds_config": partial(llama2_70b_peft_config, peft_scheme="lora"),
    "llama2_70b_peft_8gpu_gb200_fp8ds_config": partial(llama2_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_peft_8gpu_gb200_bf16_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_peft_8gpu_gb200_fp8cs_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_peft_8gpu_gb200_fp8mx_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_pretrain_32gpu_gb200_bf16_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_32gpu_gb200_fp8cs_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_gb200_bf16_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_gb200_fp8cs_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_gb200_fp8mx_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_gb200_nvfp4_config": llama3_70b_pretrain_config,
    "llama3_70b_sft_32gpu_gb200_bf16_config": llama3_70b_sft_config,
    "llama3_70b_sft_32gpu_gb200_fp8cs_config": llama3_70b_sft_config,
    "llama3_70b_sft_32gpu_gb200_fp8mx_config": llama3_70b_sft_config,
    "llama3_8b_pretrain_32gpu_gb200_bf16_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_32gpu_gb200_fp8cs_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_gb200_bf16_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_gb200_fp8cs_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_gb200_fp8mx_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_gb200_nvfp4_config": llama3_8b_pretrain_config,
    "llama3_8b_sft_8gpu_gb200_bf16_config": llama3_8b_sft_config,
    "llama3_8b_sft_8gpu_gb200_fp8cs_config": llama3_8b_sft_config,
    "llama3_8b_sft_8gpu_gb200_fp8mx_config": llama3_8b_sft_config,
    "llama31_405b_pretrain_128gpu_gb200_bf16_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_128gpu_gb200_fp8cs_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_128gpu_gb200_fp8mx_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_128gpu_gb200_nvfp4_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_gb200_bf16_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_gb200_fp8cs_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_gb200_fp8mx_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_gb200_nvfp4_config": llama31_405b_pretrain_config,
    "llama31_8b_pretrain_512gpu_gb200_fp8cs_config": llama31_8b_pretrain_config,
    "llama31_8b_pretrain_72gpu_gb200_nvfp4_config": llama31_8b_pretrain_config,
    "llama31_8b_pretrain_8gpu_gb200_nvfp4_config": llama31_8b_pretrain_config,
    "llama2_70b_peft_4gpu_gb300_fp8ds_config": partial(llama2_70b_peft_config, peft_scheme="lora"),
    "llama2_70b_peft_512gpu_gb300_fp8ds_config": partial(llama2_70b_peft_config, peft_scheme="lora"),
    "llama2_70b_peft_72gpu_gb300_fp8ds_config": partial(llama2_70b_peft_config, peft_scheme="lora"),
    "llama2_70b_peft_8gpu_gb300_fp8ds_config": partial(llama2_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_peft_8gpu_gb300_bf16_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_peft_8gpu_gb300_fp8cs_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_peft_8gpu_gb300_fp8mx_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_pretrain_32gpu_gb300_bf16_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_32gpu_gb300_fp8cs_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_gb300_bf16_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_gb300_fp8cs_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_gb300_fp8mx_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_gb300_nvfp4_config": llama3_70b_pretrain_config,
    "llama3_70b_sft_32gpu_gb300_bf16_config": llama3_70b_sft_config,
    "llama3_70b_sft_32gpu_gb300_fp8cs_config": llama3_70b_sft_config,
    "llama3_70b_sft_32gpu_gb300_fp8mx_config": llama3_70b_sft_config,
    "llama3_8b_pretrain_32gpu_gb300_bf16_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_32gpu_gb300_fp8cs_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_32gpu_gb300_fp8mx_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_32gpu_gb300_nvfp4_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_gb300_bf16_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_gb300_fp8cs_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_gb300_fp8mx_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_gb300_nvfp4_config": llama3_8b_pretrain_config,
    "llama31_405b_pretrain_128gpu_gb300_bf16_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_128gpu_gb300_fp8cs_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_128gpu_gb300_fp8mx_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_128gpu_gb300_nvfp4_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_gb300_bf16_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_gb300_fp8cs_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_gb300_fp8mx_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_gb300_nvfp4_config": llama31_405b_pretrain_config,
    "llama31_8b_pretrain_512gpu_gb300_fp8cs_config": llama31_8b_pretrain_config,
    "llama31_8b_pretrain_72gpu_gb300_nvfp4_config": llama31_8b_pretrain_config,
    "llama31_8b_pretrain_8gpu_gb300_nvfp4_config": llama31_8b_pretrain_config,
    "llama3_70b_peft_8gpu_h100_bf16_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_peft_8gpu_h100_fp8cs_config": partial(llama3_70b_peft_config, peft_scheme="lora"),
    "llama3_70b_pretrain_64gpu_h100_bf16_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_h100_fp8cs_config": llama3_70b_pretrain_config,
    "llama3_70b_sft_32gpu_h100_bf16_config": llama3_70b_sft_config,
    "llama3_70b_sft_32gpu_h100_fp8cs_config": llama3_70b_sft_config,
    "llama3_8b_pretrain_64gpu_h100_bf16_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_64gpu_h100_fp8cs_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_h100_bf16_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_h100_fp8cs_config": llama3_8b_pretrain_config,
    "llama3_8b_sft_8gpu_h100_bf16_config": llama3_8b_sft_config,
    "llama3_8b_sft_8gpu_h100_fp8cs_config": llama3_8b_sft_config,
    "llama31_405b_pretrain_1024gpu_h100_bf16_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_1024gpu_h100_fp8cs_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_512gpu_h100_bf16_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_512gpu_h100_fp8cs_config": llama31_405b_pretrain_config,
    "llama3_8b_pretrain_8gpu_r100_bf16_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_r100_fp8cs_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_r100_fp8mx_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_r100_nvfp4_config": llama3_8b_pretrain_config,
    "llama3_70b_pretrain_64gpu_vrnvl72_bf16_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_vrnvl72_fp8mx_config": llama3_70b_pretrain_config,
    "llama3_70b_pretrain_64gpu_vrnvl72_nvfp4_config": llama3_70b_pretrain_config,
    "llama3_8b_pretrain_8gpu_vrnvl72_bf16_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_vrnvl72_fp8cs_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_vrnvl72_fp8mx_config": llama3_8b_pretrain_config,
    "llama3_8b_pretrain_8gpu_vrnvl72_nvfp4_config": llama3_8b_pretrain_config,
    "llama31_405b_pretrain_256gpu_vrnvl72_bf16_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_vrnvl72_fp8mx_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_vrnvl72_nvfp4_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_b200_bf16_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_b200_fp8cs_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_b200_fp8mx_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_b300_fp8cs_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_b300_fp8mx_config": llama31_405b_pretrain_config,
    "llama31_405b_pretrain_256gpu_b300_nvfp4_config": llama31_405b_pretrain_config,
}
ALLOCATION_AWARE = ()


def apply_overrides(cfg: ConfigContainer, name: str) -> ConfigContainer:
    """Apply model-specific benchmark overrides to an existing library config."""
    match name:
        case "llama3_70b_peft_8gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.peft.target_modules = ["linear_qkv"]
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b200_bf16"].copy()
            return cfg
        case (
            "llama3_70b_peft_8gpu_b200_fp8cs_config"
            | "llama3_70b_peft_8gpu_gb200_fp8cs_config"
            | "llama3_70b_peft_8gpu_gb300_fp8cs_config"
        ):
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.peft.target_modules = ["linear_qkv"]
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 20
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b200_bf16"].copy()
            return cfg
        case "llama3_70b_peft_8gpu_b200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.peft.target_modules = ["linear_qkv"]
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b200_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 5
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
                "NCCL_NVLS_ENABLE": 0,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NCCL_P2P_NET_CHUNKSIZE": 2097152,
            }
            return cfg
        case "llama3_70b_pretrain_64gpu_b200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.average_in_collective = False
            cfg.ddp.fsdp_double_buffer = True
            cfg.ddp.suggested_communication_unit_size = 800000000
            cfg.model.init_model_with_meta_device = True
            cfg.model.gradient_accumulation_fusion = False
            cfg.checkpoint.load = None
            cfg.model.cpu_offloading = True
            cfg.model.cpu_offloading_weights = False
            cfg.model.cpu_offloading_num_layers = 5
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8cs"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_b200_fp8mx_config" | "llama3_70b_pretrain_64gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 5
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_b200_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 5
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_nvfp4"].copy()
            return cfg
        case "llama3_8b_pretrain_64gpu_b200_bf16_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_8b_pretrain_8gpu_b200_bf16_config"), 1024)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_64gpu_b200_fp8cs_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_8b_pretrain_8gpu_b200_fp8cs_config"), 1024)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_b200_bf16_config" | "llama3_8b_pretrain_8gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 2
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_b200_fp8cs_config" | "llama3_8b_pretrain_8gpu_gb200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 2
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_b200_fp8mx_config" | "llama3_8b_pretrain_8gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 2
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_b200_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 4
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_8gpu_b200_nvfp4"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_b200_bf16_config" | "llama31_405b_pretrain_128gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_b200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_b200_fp8mx_config" | "llama31_405b_pretrain_128gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_b200_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            cfg.comm_overlap.tp_comm_overlap = False
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_nvfp4"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_b200_nvfp4_config":
            apply_overrides(cfg, "llama31_405b_pretrain_256gpu_gb200_nvfp4_config")
            cfg.ddp.overlap_param_gather = True
            cfg.optimizer.overlap_param_gather = False
            cfg.comm_overlap.overlap_param_gather = None
            cfg.comm_overlap.align_param_gather = None
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_nvfp4"].copy()
            return cfg
        case "llama3_70b_peft_8gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.peft.target_modules = ["linear_qkv"]
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b300_bf16"].copy()
            return cfg
        case "llama3_70b_peft_8gpu_b300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.peft.target_modules = ["linear_qkv"]
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b300_bf16"].copy()
            return cfg
        case "llama3_70b_peft_8gpu_b300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.peft.target_modules = ["linear_qkv"]
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b300_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.average_in_collective = False
            cfg.ddp.fsdp_double_buffer = True
            cfg.ddp.suggested_communication_unit_size = 800000000
            cfg.model.init_model_with_meta_device = True
            cfg.model.gradient_accumulation_fusion = False
            cfg.checkpoint.load = None
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b300_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_b300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.average_in_collective = False
            cfg.ddp.fsdp_double_buffer = True
            cfg.ddp.suggested_communication_unit_size = 800000000
            cfg.model.init_model_with_meta_device = True
            cfg.model.gradient_accumulation_fusion = False
            cfg.checkpoint.load = None
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b300_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_b300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 5
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b300_fp8mx"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_b300_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 5
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b300_nvfp4"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 4
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_8gpu_b300_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_b300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 4
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_8gpu_b300_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_b300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 4
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_8gpu_b300_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_b300_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 4
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            _llama_benchmark_common(cfg)
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NCCL_IGNORE_CPU_AFFINITY": 1,
                "NVTE_USE_FAST_MATH": 1,
            }
            return cfg
        case "llama31_405b_pretrain_128gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.average_in_collective = False
            cfg.model.init_model_with_meta_device = True
            cfg.model.gradient_accumulation_fusion = True
            cfg.checkpoint.load = None
            cfg.model.cpu_offloading = True
            cfg.model.cpu_offloading_weights = False
            cfg.model.cpu_offloading_num_layers = 40
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b300_bf16"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_b300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b300_fp8mx"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_b300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b300_fp8mx"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_b300_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            cfg.comm_overlap.tp_comm_overlap = False
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b300_nvfp4"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_b300_bf16_config":
            apply_overrides(cfg, "llama31_405b_pretrain_256gpu_gb300_bf16_config")
            cfg.ddp.fsdp_double_buffer = False
            cfg.model.gradient_accumulation_fusion = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b300_bf16"].copy()
            return cfg
        case "llama2_70b_peft_4gpu_gb200_fp8ds_config":
            cfg.mixed_precision = _llama2_70b_precision_config("fp8_ds")
            _llama2_benchmark_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.train.global_batch_size = 8
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = None
            cfg.model.cpu_offloading = True
            cfg.model.cpu_offloading_num_layers = 20
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama2_70b_peft_4gpu_gb200_fp8ds"].copy()
            return cfg
        case "llama2_70b_peft_512gpu_gb200_fp8ds_config":
            cfg.mixed_precision = _llama2_70b_precision_config("fp8_ds")
            _llama2_benchmark_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 8
            cfg.train.global_batch_size = 64
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama2_70b_peft_4gpu_gb200_fp8ds"].copy()
            return cfg
        case "llama2_70b_peft_72gpu_gb200_fp8ds_config":
            cfg.mixed_precision = _llama2_70b_precision_config("fp8_ds")
            _llama2_benchmark_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 8
            cfg.train.global_batch_size = 9
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.validation.eval_global_batch_size = 8
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama2_70b_peft_4gpu_gb200_fp8ds"].copy()
            return cfg
        case "llama2_70b_peft_8gpu_gb200_fp8ds_config":
            cfg.mixed_precision = _llama2_70b_precision_config("fp8_ds")
            _llama2_benchmark_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.train.global_batch_size = 8
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.model.cpu_offloading = True
            cfg.model.cpu_offloading_num_layers = 11
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama2_70b_peft_4gpu_gb200_fp8ds"].copy()
            return cfg
        case "llama3_70b_peft_8gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.peft.target_modules = ["linear_qkv"]
            cfg.model.seq_length = 2048
            cfg.dataset.seq_length = 2048
            cfg.dataset.offline_packing_specs.packed_sequence_size = 2048
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 20
            cfg.train.global_batch_size = 64
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b200_bf16"].copy()
            return cfg
        case "llama3_70b_peft_8gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.peft.target_modules = ["linear_qkv"]
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 20
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b200_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_32gpu_gb200_bf16_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_70b_pretrain_64gpu_gb200_bf16_config"), 128)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_32gpu_gb200_fp8cs_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_70b_pretrain_64gpu_gb200_fp8cs_config"), 128)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.average_in_collective = False
            cfg.ddp.fsdp_double_buffer = True
            cfg.ddp.suggested_communication_unit_size = 800000000
            cfg.model.init_model_with_meta_device = True
            cfg.model.gradient_accumulation_fusion = False
            cfg.checkpoint.load = None
            cfg.model.cpu_offloading = True
            cfg.model.cpu_offloading_weights = False
            cfg.model.cpu_offloading_num_layers = 20
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_gb200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 2
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.average_in_collective = False
            cfg.ddp.fsdp_double_buffer = True
            cfg.ddp.suggested_communication_unit_size = 800000000
            cfg.model.init_model_with_meta_device = True
            cfg.model.gradient_accumulation_fusion = False
            cfg.checkpoint.load = None
            cfg.model.cpu_offloading = True
            cfg.model.cpu_offloading_weights = False
            cfg.model.cpu_offloading_num_layers = 40
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_gb200_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 5
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            clear_cuda_graph_modules(cfg.model)
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_nvfp4"].copy()
            return cfg
        case "llama3_70b_sft_32gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.virtual_pipeline_model_parallel_size = 10
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.comm_overlap.defer_embedding_wgrad_compute = True
            cfg.comm_overlap.wgrad_deferral_limit = 22
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_sft_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_70b_sft_32gpu_gb200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.virtual_pipeline_model_parallel_size = 10
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.comm_overlap.defer_embedding_wgrad_compute = True
            cfg.comm_overlap.wgrad_deferral_limit = 22
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_sft_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_70b_sft_32gpu_gb200_fp8mx_config":
            apply_overrides(cfg, "llama3_70b_sft_32gpu_gb200_fp8cs_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_sft_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_32gpu_gb200_bf16_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_8b_pretrain_8gpu_gb200_bf16_config"), 512)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_32gpu_gb200_fp8cs_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_8b_pretrain_8gpu_gb200_fp8cs_config"), 512)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_gb200_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.mixed_precision.fp8_dot_product_attention = True
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 4
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = []
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            _llama_benchmark_common(cfg)
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_USE_FAST_MATH": 1,
                "NVTE_DPA_FP8_RECIPE": "Float8CurrentScaling",
            }
            return cfg
        case "llama3_8b_sft_8gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.model.seq_length = 16384
            cfg.dataset.seq_length = 16384
            cfg.dataset.offline_packing_specs.packed_sequence_size = 16384
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 8
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = ["mlp"]
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_sft_8gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_8b_sft_8gpu_gb200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.model.seq_length = 16384
            cfg.dataset.seq_length = 16384
            cfg.dataset.offline_packing_specs.packed_sequence_size = 16384
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 8
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = ["mlp"]
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_sft_8gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_8b_sft_8gpu_gb200_fp8mx_config":
            apply_overrides(cfg, "llama3_8b_sft_8gpu_gb200_fp8cs_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_sft_8gpu_gb200_bf16"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_gb200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_sft_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_gb200_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            cfg.comm_overlap.tp_comm_overlap = False
            _llama_benchmark_common(cfg)
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.comm_overlap.overlap_param_gather = False
            cfg.comm_overlap.align_param_gather = False
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_nvfp4"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.sequence_parallel = True
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.train.global_batch_size = 1536
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_gb200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.sequence_parallel = True
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.train.global_batch_size = 1536
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_sft_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 1536
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_gb200_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 16
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 1536
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            cfg.comm_overlap.tp_comm_overlap = False
            _llama_benchmark_common(cfg)
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.comm_overlap.overlap_param_gather = False
            cfg.comm_overlap.align_param_gather = False
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_nvfp4"].copy()
            return cfg
        case "llama31_8b_pretrain_512gpu_gb200_fp8cs_config" | "llama31_8b_pretrain_512gpu_gb300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.mixed_precision.num_layers_at_start_in_bf16 = 0
            cfg.mixed_precision.num_layers_at_end_in_bf16 = 0
            cfg.mixed_precision.first_last_layers_bf16 = False
            cfg.mixed_precision.fp4_param_gather = False
            cfg.mixed_precision.fp4_param = False
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 64
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            _llama31_8b_common(cfg)
            cfg.optimizer.lr = 0.0008
            cfg.optimizer.min_lr = 8e-05
            cfg.ddp.fp8_param_gather = True
            cfg.model.tp_comm_bootstrap_backend = "gloo"
            cfg.model.tp_comm_overlap = True
            cfg.validation.eval_global_batch_size = 64
            cfg.validation.eval_interval = 192
            cfg.validation.eval_iters = 16
            cfg.validation.eval_micro_batch_size = 1
            cfg.train.eval_iters = 16
            cfg.scheduler.lr_decay_iters = 1199936
            cfg.scheduler.lr_decay_steps = 76795904
            cfg.scheduler.lr_warmup_iters = 64
            cfg.scheduler.lr_warmup_steps = 4096
            cfg.scheduler.wd_incr_steps = 76800000
            cfg.load_main_params_from_ckpt = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama31_8b_pretrain_512gpu_gb200_fp8cs"].copy()
            return cfg
        case "llama31_8b_pretrain_72gpu_gb200_nvfp4_config" | "llama31_8b_pretrain_72gpu_gb300_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.mixed_precision.num_layers_at_start_in_bf16 = 0
            cfg.mixed_precision.num_layers_at_end_in_bf16 = 0
            cfg.mixed_precision.first_last_layers_bf16 = False
            cfg.mixed_precision.fp4_param_gather = False
            cfg.mixed_precision.fp4_param = False
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 72
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            _llama31_8b_common(cfg)
            cfg.optimizer.lr = 0.0008
            cfg.optimizer.min_lr = 8e-05
            cfg.model.tp_comm_bootstrap_backend = "mpi"
            cfg.model.fp8_dot_product_attention = True
            cfg.mixed_precision.fp8_dot_product_attention = True
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.validation.eval_global_batch_size = 72
            cfg.validation.eval_interval = 171
            cfg.validation.eval_iters = 15
            cfg.validation.eval_micro_batch_size = 1
            cfg.train.eval_iters = 15
            cfg.scheduler.lr_decay_iters = 1199936
            cfg.scheduler.lr_decay_steps = 86395392
            cfg.scheduler.lr_warmup_iters = 64
            cfg.scheduler.lr_warmup_steps = 4608
            cfg.scheduler.wd_incr_steps = 86400000
            cfg.load_main_params_from_ckpt = False
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama31_8b_pretrain_512gpu_gb200_fp8cs"].copy()
            return cfg
        case "llama31_8b_pretrain_8gpu_gb200_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.mixed_precision.num_layers_at_start_in_bf16 = 0
            cfg.mixed_precision.num_layers_at_end_in_bf16 = 0
            cfg.mixed_precision.first_last_layers_bf16 = False
            cfg.mixed_precision.fp4_param_gather = False
            cfg.mixed_precision.fp4_param = False
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 16
            cfg.train.micro_batch_size = 2
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            _llama31_8b_common(cfg)
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 4e-05
            cfg.model.tp_comm_bootstrap_backend = "mpi"
            cfg.model.fp8_dot_product_attention = True
            cfg.mixed_precision.fp8_dot_product_attention = True
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.validation.eval_global_batch_size = 16
            cfg.validation.eval_interval = 768
            cfg.validation.eval_iters = 64
            cfg.train.eval_iters = 64
            cfg.validation.eval_micro_batch_size = 2
            cfg.scheduler.lr_decay_iters = 1199984
            cfg.scheduler.lr_decay_steps = 19199744
            cfg.scheduler.lr_warmup_iters = 16
            cfg.scheduler.lr_warmup_steps = 256
            cfg.scheduler.wd_incr_steps = 19200000
            cfg.load_main_params_from_ckpt = False
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama31_8b_pretrain_512gpu_gb200_fp8cs"].copy()
            return cfg
        case "llama2_70b_peft_4gpu_gb300_fp8ds_config":
            cfg.mixed_precision = _llama2_70b_precision_config("fp8_ds")
            _llama2_benchmark_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.train.global_batch_size = 8
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = "full_iteration"
            cfg.model.cpu_offloading = False
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama2_70b_peft_4gpu_gb200_fp8ds"].copy()
            return cfg
        case "llama2_70b_peft_512gpu_gb300_fp8ds_config":
            cfg.mixed_precision = _llama2_70b_precision_config("fp8_ds")
            _llama2_benchmark_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 8
            cfg.train.global_batch_size = 64
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = "full_iteration"
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama2_70b_peft_4gpu_gb200_fp8ds"].copy()
            return cfg
        case "llama2_70b_peft_72gpu_gb300_fp8ds_config":
            cfg.mixed_precision = _llama2_70b_precision_config("fp8_ds")
            _llama2_benchmark_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 8
            cfg.train.global_batch_size = 9
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = "full_iteration"
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama2_70b_peft_4gpu_gb200_fp8ds"].copy()
            return cfg
        case "llama2_70b_peft_8gpu_gb300_fp8ds_config":
            cfg.mixed_precision = _llama2_70b_precision_config("fp8_ds")
            _llama2_benchmark_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.train.global_batch_size = 8
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = "full_iteration"
            cfg.model.cpu_offloading = False
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama2_70b_peft_4gpu_gb200_fp8ds"].copy()
            return cfg
        case "llama3_70b_peft_8gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.peft.target_modules = ["linear_qkv"]
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b200_bf16"].copy()
            return cfg
        case "llama3_70b_peft_8gpu_gb300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.peft.target_modules = ["linear_qkv"]
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 20
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b200_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_32gpu_gb300_bf16_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_70b_pretrain_64gpu_gb300_bf16_config"), 128)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_32gpu_gb300_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_32gpu_gb300_fp8cs_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_70b_pretrain_64gpu_gb300_fp8cs_config"), 128)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 2
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.average_in_collective = False
            cfg.ddp.fsdp_double_buffer = True
            cfg.ddp.suggested_communication_unit_size = 800000000
            cfg.model.init_model_with_meta_device = True
            cfg.model.gradient_accumulation_fusion = False
            cfg.checkpoint.load = None
            cfg.ddp.nccl_ub = True
            cfg.ddp.fsdp_manual_registration = True
            cfg.model.cpu_offloading = True
            cfg.model.cpu_offloading_weights = False
            cfg.model.cpu_offloading_num_layers = 30
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_32gpu_gb300_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_gb300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 2
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.average_in_collective = False
            cfg.ddp.fsdp_double_buffer = True
            cfg.ddp.suggested_communication_unit_size = 800000000
            cfg.model.init_model_with_meta_device = True
            cfg.model.gradient_accumulation_fusion = False
            cfg.checkpoint.load = None
            cfg.model.cpu_offloading = True
            cfg.model.cpu_offloading_weights = False
            cfg.model.cpu_offloading_num_layers = 20
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_gb300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 5
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_gb300_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 5
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_nvfp4"].copy()
            return cfg
        case "llama3_70b_sft_32gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 20
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.comm_overlap.defer_embedding_wgrad_compute = True
            cfg.comm_overlap.wgrad_deferral_limit = 22
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_sft_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_70b_sft_32gpu_gb300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 20
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.comm_overlap.defer_embedding_wgrad_compute = True
            cfg.comm_overlap.wgrad_deferral_limit = 22
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_sft_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_70b_sft_32gpu_gb300_fp8mx_config":
            apply_overrides(cfg, "llama3_70b_sft_32gpu_gb300_fp8cs_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_sft_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_32gpu_gb300_bf16_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_8b_pretrain_8gpu_gb300_bf16_config"), 512)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_32gpu_gb300_fp8cs_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_8b_pretrain_8gpu_gb300_fp8cs_config"), 512)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_32gpu_gb300_fp8mx_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_8b_pretrain_8gpu_gb300_fp8mx_config"), 512)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_32gpu_gb300_nvfp4_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_8b_pretrain_8gpu_gb300_nvfp4_config"), 512)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_32gpu_gb300_nvfp4"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 4
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_gb300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 4
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_gb300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 4
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_gb300_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 4
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_32gpu_gb300_nvfp4"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.average_in_collective = False
            cfg.ddp.fsdp_double_buffer = True
            cfg.model.init_model_with_meta_device = True
            cfg.model.gradient_accumulation_fusion = False
            cfg.checkpoint.load = None
            cfg.model.cpu_offloading = True
            cfg.model.cpu_offloading_weights = False
            cfg.model.cpu_offloading_num_layers = 40
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8cs"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_gb300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_sft_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_gb300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "llama31_405b_pretrain_128gpu_gb300_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 768
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = []
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            cfg.comm_overlap.tp_comm_overlap = False
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_nvfp4"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = True
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.train.global_batch_size = 1536
            cfg.train.micro_batch_size = 1
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.average_in_collective = False
            cfg.ddp.fsdp_double_buffer = True
            cfg.model.init_model_with_meta_device = True
            cfg.model.gradient_accumulation_fusion = False
            cfg.checkpoint.load = None
            cfg.model.cpu_offloading = True
            cfg.model.cpu_offloading_weights = False
            cfg.model.cpu_offloading_num_layers = 40
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8cs"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_gb300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.sequence_parallel = True
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.train.global_batch_size = 1536
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_sft_32gpu_gb200_bf16"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_gb300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 1536
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_gb300_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 1536
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
            cfg.comm_overlap.tp_comm_overlap = False
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_nvfp4"].copy()
            return cfg
        case "llama31_8b_pretrain_8gpu_gb300_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.mixed_precision.num_layers_at_start_in_bf16 = 0
            cfg.mixed_precision.num_layers_at_end_in_bf16 = 0
            cfg.mixed_precision.first_last_layers_bf16 = False
            cfg.mixed_precision.fp4_param_gather = False
            cfg.mixed_precision.fp4_param = False
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 16
            cfg.train.micro_batch_size = 2
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            _llama31_8b_common(cfg)
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 4e-05
            cfg.model.tp_comm_bootstrap_backend = "mpi"
            cfg.model.fp8_dot_product_attention = True
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.mixed_precision.fp8_dot_product_attention = True
            cfg.validation.eval_global_batch_size = 16
            cfg.validation.eval_interval = 768
            cfg.validation.eval_iters = 64
            cfg.validation.eval_micro_batch_size = 2
            cfg.train.eval_iters = 64
            cfg.scheduler.lr_decay_iters = 1199984
            cfg.scheduler.lr_decay_steps = 19199744
            cfg.scheduler.lr_warmup_iters = 16
            cfg.scheduler.lr_warmup_steps = 256
            cfg.scheduler.wd_incr_steps = 19200000
            cfg.load_main_params_from_ckpt = False
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama31_8b_pretrain_512gpu_gb200_fp8cs"].copy()
            return cfg
        case "llama3_70b_peft_8gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.peft.target_modules = ["linear_qkv"]
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 20
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_granularity = "full"
            cfg.model.recompute_method = "block"
            cfg.model.recompute_num_layers = 1
            cfg.comm_overlap.tp_comm_overlap = False
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = False
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b200_bf16"].copy()
            return cfg
        case "llama3_70b_peft_8gpu_h100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.peft.target_modules = ["linear_qkv"]
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 20
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = False
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 1,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            }
            return cfg
        case "llama3_70b_pretrain_64gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 5
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_h100_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_h100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 5
            cfg.model.pipeline_model_parallel_layout = "Et*2|(t*2|)*34,t*3|(t*2|)*3,tL"
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 256
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_h100_bf16"].copy()
            return cfg
        case "llama3_70b_sft_32gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 5
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap = True
            cfg.comm_overlap.defer_embedding_wgrad_compute = True
            cfg.comm_overlap.wgrad_deferral_limit = 22
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = False
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_sft_32gpu_h100_bf16"].copy()
            return cfg
        case "llama3_70b_sft_32gpu_h100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 4
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = 5
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mlp"]
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True, defer_embedding_wgrad_compute=True, wgrad_deferral_limit=22
            )
            cfg.dataset.offline_packing_specs.pad_cu_seqlens = False
            cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_sft_32gpu_h100_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_64gpu_h100_bf16_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_8b_pretrain_8gpu_h100_bf16_config"), 1024)
            return cfg
        case "llama3_8b_pretrain_64gpu_h100_fp8cs_config":
            _with_global_batch_size(apply_overrides(cfg, "llama3_8b_pretrain_8gpu_h100_fp8cs_config"), 1024)
            return cfg
        case "llama3_8b_pretrain_8gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = {**cfg.env_vars, "NVTE_BWD_LAYERNORM_SM_MARGIN": 20, "NVTE_FWD_LAYERNORM_SM_MARGIN": 20}
            return cfg
        case "llama3_8b_pretrain_8gpu_h100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.context_parallel_size = 1
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_granularity = "full"
            cfg.model.recompute_method = "block"
            cfg.model.recompute_num_layers = 5
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = {
                **cfg.env_vars,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_CTA_POLICY": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
            }
            return cfg
        case "llama3_8b_sft_8gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            _llama_benchmark_common(cfg)
            cfg.env_vars = {
                **cfg.env_vars,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
            }
            return cfg
        case "llama3_8b_sft_8gpu_h100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.mixed_precision.grad_reduce_in_fp32 = False
            cfg.ddp.grad_reduce_in_fp32 = False
            cfg.model.disable_parameter_transpose_cache = True
            cfg.ddp.use_distributed_optimizer = True
            cfg.optimizer.use_distributed_optimizer = True
            cfg.model.seq_length = 4096
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = ["mlp"]
            _llama_benchmark_common(cfg)
            cfg.env_vars = {
                **cfg.env_vars,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
            }
            return cfg
        case "llama31_405b_pretrain_1024gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 8
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.sequence_parallel = True
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.train.global_batch_size = 1536
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_h100_bf16"].copy()
            return cfg
        case "llama31_405b_pretrain_1024gpu_h100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.seq_length = 8192
            cfg.dataset.seq_length = 8192
            cfg.model.tensor_model_parallel_size = 8
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = 8
            cfg.model.sequence_parallel = True
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.train.global_batch_size = 1536
            cfg.train.micro_batch_size = 1
            cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_h100_bf16"].copy()
            return cfg
        case "llama31_405b_pretrain_512gpu_h100_bf16_config":
            _with_global_batch_size(apply_overrides(cfg, "llama31_405b_pretrain_1024gpu_h100_bf16_config"), 768)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_h100_bf16"].copy()
            return cfg
        case "llama31_405b_pretrain_512gpu_h100_fp8cs_config":
            _with_global_batch_size(apply_overrides(cfg, "llama31_405b_pretrain_1024gpu_h100_fp8cs_config"), 768)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_h100_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_r100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_r100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_r100_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_peft_8gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_r100_nvfp4_config":
            cfg.mixed_precision = _perf_precision("nvfp4")
            cfg.tokenizer.vocab_size = 128256
            cfg.model.should_pad_vocab = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 128
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "none"
            cfg.model.cuda_graph_scope = ["full_iteration"]
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.model.moe_token_dispatcher_type = "alltoall"
            _llama_benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_8gpu_b200_nvfp4"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_vrnvl72_bf16_config":
            apply_overrides(cfg, "llama3_70b_pretrain_64gpu_gb300_bf16_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_32gpu_gb300_bf16"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_vrnvl72_fp8mx_config":
            apply_overrides(cfg, "llama3_70b_pretrain_64gpu_gb300_fp8mx_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "llama3_70b_pretrain_64gpu_vrnvl72_nvfp4_config":
            apply_overrides(cfg, "llama3_70b_pretrain_64gpu_gb300_nvfp4_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_nvfp4"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_vrnvl72_bf16_config":
            apply_overrides(cfg, "llama3_8b_pretrain_8gpu_gb300_bf16_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_vrnvl72_fp8cs_config":
            apply_overrides(cfg, "llama3_8b_pretrain_8gpu_gb300_fp8cs_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_vrnvl72_fp8mx_config":
            apply_overrides(cfg, "llama3_8b_pretrain_8gpu_gb300_fp8mx_config")
            cfg.mixed_precision.fp8_dot_product_attention = True
            cfg.model.fused_single_qkv_rope = True
            cfg.model.use_te_activation_func = True
            cfg.model.bias_activation_fusion = False
            cfg.dataset.create_attention_mask = False
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_8b_pretrain_64gpu_b200_bf16"].copy()
            return cfg
        case "llama3_8b_pretrain_8gpu_vrnvl72_nvfp4_config":
            apply_overrides(cfg, "llama3_8b_pretrain_8gpu_gb300_nvfp4_config")
            cfg.mixed_precision.fp8_dot_product_attention = True
            cfg.model.fused_single_qkv_rope = True
            cfg.model.use_te_activation_func = True
            cfg.model.bias_activation_fusion = False
            cfg.dataset.create_attention_mask = False
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
                "NCCL_NVLS_ENABLE": 0,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_USE_FAST_MATH": 1,
                "NVTE_DPA_FP8_RECIPE": "MXFP8BlockScaling",
            }
            return cfg
        case "llama31_405b_pretrain_256gpu_vrnvl72_bf16_config":
            apply_overrides(cfg, "llama31_405b_pretrain_256gpu_gb300_bf16_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8cs"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_vrnvl72_fp8mx_config":
            apply_overrides(cfg, "llama31_405b_pretrain_256gpu_gb300_fp8mx_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_vrnvl72_nvfp4_config":
            apply_overrides(cfg, "llama31_405b_pretrain_256gpu_gb300_nvfp4_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["llama3_70b_pretrain_64gpu_b200_nvfp4"].copy()
            return cfg
        case "llama31_405b_pretrain_256gpu_b200_bf16_config":
            return apply_overrides(cfg, "llama31_405b_pretrain_256gpu_gb200_bf16_config")
        case "llama31_405b_pretrain_256gpu_b200_fp8cs_config":
            return apply_overrides(cfg, "llama31_405b_pretrain_256gpu_gb200_fp8cs_config")
        case "llama31_405b_pretrain_256gpu_b200_fp8mx_config":
            return apply_overrides(cfg, "llama31_405b_pretrain_256gpu_gb200_fp8mx_config")
        case "llama31_405b_pretrain_256gpu_b300_fp8cs_config":
            return apply_overrides(cfg, "llama31_405b_pretrain_256gpu_gb300_fp8cs_config")
        case "llama31_405b_pretrain_256gpu_b300_fp8mx_config":
            return apply_overrides(cfg, "llama31_405b_pretrain_256gpu_gb300_fp8mx_config")
        case "llama31_405b_pretrain_256gpu_b300_nvfp4_config":
            return apply_overrides(cfg, "llama31_405b_pretrain_256gpu_gb300_nvfp4_config")
        case _:
            raise ValueError(f"Unknown benchmark overrides: {name}")


BENCHMARK_ALIASES = {
    "llama31_405b_pretrain_256gpu_b200_bf16_config": "llama31_405b_pretrain_256gpu_gb200_bf16_config",
    "llama31_405b_pretrain_256gpu_b200_fp8cs_config": "llama31_405b_pretrain_256gpu_gb200_fp8cs_config",
    "llama31_405b_pretrain_256gpu_b200_fp8mx_config": "llama31_405b_pretrain_256gpu_gb200_fp8mx_config",
    "llama31_405b_pretrain_256gpu_b300_fp8cs_config": "llama31_405b_pretrain_256gpu_gb300_fp8cs_config",
    "llama31_405b_pretrain_256gpu_b300_fp8mx_config": "llama31_405b_pretrain_256gpu_gb300_fp8mx_config",
    "llama31_405b_pretrain_256gpu_b300_nvfp4_config": "llama31_405b_pretrain_256gpu_gb300_nvfp4_config",
}
