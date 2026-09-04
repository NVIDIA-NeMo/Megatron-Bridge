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
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_openmathinstruct2_config, default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


NEMOTRON_3_ULTRA_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
NEMOTRON_3_ULTRA_TOKENIZER_NAME = "nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH = 8192
NEMOTRON_3_ULTRA_OPENMATHINSTRUCT2_SEQ_LENGTH = 4096


def nemotron_3_ultra_pretrain_24gpu_h100_bf16_config() -> ConfigContainer:
    """Return a pre-training config for Nemotron 3 Ultra.

    Returns:
        Pre-training configuration for Nemotron 3 Ultra.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(NEMOTRON_3_ULTRA_HF_MODEL_ID).to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 3
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH
    cfg.model.apply_rope_fusion = False
    cfg.model.attention_backend = "fused"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.init_method_std = 0.014
    cfg.model.use_fused_weighted_squared_relu = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    cfg.model.mtp_num_layers = 2
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.use_te_rng_tracker = True

    cfg.tokenizer.tokenizer_model = NEMOTRON_3_ULTRA_HF_MODEL_ID
    cfg.dataset.seq_length = NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 1
    cfg.dataset.mmap_bin_files = False

    cfg.train.train_iters = 39735
    cfg.train.global_batch_size = 3072
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0
    cfg.validation.eval_interval = 1000

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.mixed_precision = "bf16_mixed"

    cfg.optimizer.lr = 2.5e-4
    cfg.optimizer.min_lr = 2.5e-4
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "constant"

    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.async_save = True
    cfg.checkpoint.async_strategy = "mcore"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        # HybridEP topology for this recipe.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


def _nemotron_3_ultra_large_scale_bf16_config() -> ConfigContainer:
    """Apply large-scale execution settings without benchmark-only policy."""
    cfg = nemotron_3_ultra_pretrain_24gpu_h100_bf16_config()

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.grad_reduce_in_fp32 = True

    # Keep the performance recipe's supported kernels and memory-management
    # settings, but retain the library recipe's optimizer, scheduler,
    # checkpointing, natural routing, and numerical validation contracts.
    cfg.model.apply_rope_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_quantization = False
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []

    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.dist.high_priority_stream_groups = ["ep"]
    cfg.dist.distributed_timeout_minutes = 30

    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.check_for_large_grads = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    return cfg


def nemotron_3_ultra_pretrain_256gpu_h100_bf16_fsdp_config() -> ConfigContainer:
    """Return a convergence-safe Nemotron 3 Ultra pretrain config for 256 H100 GPUs.

    The execution policy adopts the memory-feasible TP4/PP1/EP64 Megatron-FSDP
    shape derived from the performance configuration. It intentionally keeps
    BF16 compute, natural expert routing, numerical checks, and the library
    recipe's training objective.

    Returns:
        H100 BF16 Megatron-FSDP pretraining configuration.
    """
    cfg = _nemotron_3_ultra_large_scale_bf16_config()

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH
    cfg.dataset.seq_length = NEMOTRON_3_ULTRA_PRETRAIN_SEQ_LENGTH
    cfg.train.global_batch_size = 512

    # Stateful DeepEP/HybridEP dispatch cannot be replayed by the full-layer
    # activation checkpoint used to fit Ultra on H100. Use the supported
    # stateless all-to-all dispatcher for this hardware profile.
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.use_transformer_engine_op_fuser = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "block"
    # Recompute every model layer: the 64-layer performance setting leaves too
    # little headroom for naturally routed real-data batches on 80-GB H100s.
    cfg.model.recompute_num_layers = 108
    cfg.model.recompute_modules = None
    cfg.model.mlp_chunks_for_training = 64
    cfg.model.mamba_chunk_size = 256

    cfg.dist.use_megatron_fsdp = True
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.ddp.megatron_fsdp_enable_fine_grained_param_gather = True
    cfg.ddp.overlap_param_gather = True
    cfg.optimizer.overlap_param_gather = True
    cfg.ddp.average_in_collective = False
    cfg.model.init_model_with_meta_device = True

    # Use one distributed-optimizer instance so this memory-feasible profile
    # uses pure FSDP rather than an outer HSDP group.
    cfg.ddp.num_distributed_optimizer_instances = 1
    cfg.ddp.outer_dp_sharding_strategy = "no_shard"
    cfg.ddp.megatron_fsdp_grad_comm_dtype = torch.float32
    cfg.ddp.megatron_fsdp_main_params_dtype = torch.float32
    cfg.ddp.megatron_fsdp_main_grads_dtype = torch.float32
    cfg.ddp.megatron_fsdp_use_decoupled_grad = True
    cfg.ddp.suggested_communication_unit_size = 1

    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    cfg.model.gradient_accumulation_fusion = True
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 8,
        "NCCL_BUFFSIZE": 262144,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def nemotron_3_ultra_sft_192gpu_h100_bf16_openmathinstruct2_packed_config() -> ConfigContainer:
    """Return a packed OpenMathInstruct-2 full SFT config for Nemotron 3 Ultra.

    Returns:
        Full-parameter SFT configuration for OpenMathInstruct-2.
    """
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(NEMOTRON_3_ULTRA_HF_MODEL_ID).to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = NEMOTRON_3_ULTRA_OPENMATHINSTRUCT2_SEQ_LENGTH
    cfg.model.apply_rope_fusion = False
    cfg.model.attention_backend = "fused"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.init_method_std = 0.014
    cfg.model.use_fused_weighted_squared_relu = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    cfg.model.mtp_num_layers = 2
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.use_te_rng_tracker = True
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["moe", "layernorm", "core_attn", "moe_act"]

    cfg.tokenizer.tokenizer_model = NEMOTRON_3_ULTRA_HF_MODEL_ID
    cfg.dataset = default_openmathinstruct2_config(
        seq_length=NEMOTRON_3_ULTRA_OPENMATHINSTRUCT2_SEQ_LENGTH,
        enable_offline_packing=True,
    )
    if cfg.dataset.offline_packing_specs is not None:
        cfg.dataset.offline_packing_specs.packed_sequence_size = NEMOTRON_3_ULTRA_OPENMATHINSTRUCT2_SEQ_LENGTH
        cfg.dataset.offline_packing_specs.tokenizer_model_name = NEMOTRON_3_ULTRA_TOKENIZER_NAME

    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 32

    cfg.optimizer.lr = 5e-6
    cfg.optimizer.min_lr = 5e-7
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.scheduler.lr_warmup_iters = 250
    cfg.scheduler.lr_decay_iters = 1000

    cfg.checkpoint.save_interval = 250
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.async_save = True
    cfg.checkpoint.async_strategy = "mcore"

    cfg.logger.log_interval = 1
    cfg.rng.seed = 5678

    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        # HybridEP topology for this recipe.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


def nemotron_3_ultra_peft_32gpu_h100_bf16_openmathinstruct2_packed_config(
    peft_scheme: str | PEFT | None = "lora",
) -> ConfigContainer:
    """Return a packed OpenMathInstruct-2 PEFT config for Nemotron 3 Ultra.

    Args:
        peft_scheme: PEFT scheme, PEFT instance, or "none".

    Returns:
        PEFT configuration for OpenMathInstruct-2.
    """
    cfg = _peft_common()

    cfg.model = AutoBridge.from_hf_pretrained(NEMOTRON_3_ULTRA_HF_MODEL_ID).to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = NEMOTRON_3_ULTRA_OPENMATHINSTRUCT2_SEQ_LENGTH
    cfg.model.apply_rope_fusion = False
    cfg.model.attention_backend = "fused"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.init_method_std = 0.014
    cfg.model.use_fused_weighted_squared_relu = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    cfg.model.mtp_num_layers = 2
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.use_te_rng_tracker = True
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["moe", "layernorm", "core_attn", "moe_act", "mlp", "shared_experts"]

    target_modules = [
        "linear_qkv",
        "linear_proj",
        "linear_fc1",
        "linear_fc2",
        "in_proj",
        "out_proj",
    ]
    cfg.peft = default_peft_config(peft_scheme, target_modules=target_modules)

    cfg.tokenizer.tokenizer_model = NEMOTRON_3_ULTRA_HF_MODEL_ID
    cfg.dataset = default_openmathinstruct2_config(
        seq_length=NEMOTRON_3_ULTRA_OPENMATHINSTRUCT2_SEQ_LENGTH,
        enable_offline_packing=True,
    )
    if cfg.dataset.offline_packing_specs is not None:
        cfg.dataset.offline_packing_specs.packed_sequence_size = NEMOTRON_3_ULTRA_OPENMATHINSTRUCT2_SEQ_LENGTH
        cfg.dataset.offline_packing_specs.tokenizer_model_name = NEMOTRON_3_ULTRA_TOKENIZER_NAME

    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 32

    cfg.optimizer.lr = 1e-4
    cfg.optimizer.min_lr = 1e-5
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.scheduler.lr_warmup_iters = 250
    cfg.scheduler.lr_decay_iters = 1000

    cfg.checkpoint.save_interval = 250
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.async_save = True
    cfg.checkpoint.async_strategy = "nvrx"

    cfg.logger.log_interval = 1
    cfg.rng.seed = 5678

    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        # HybridEP topology for this recipe.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


__all__ = [
    "nemotron_3_ultra_peft_32gpu_h100_bf16_openmathinstruct2_packed_config",
    "nemotron_3_ultra_pretrain_24gpu_h100_bf16_config",
    "nemotron_3_ultra_pretrain_256gpu_h100_bf16_fsdp_config",
    "nemotron_3_ultra_sft_192gpu_h100_bf16_openmathinstruct2_packed_config",
]
