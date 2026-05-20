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

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import bf16_mixed


def qwen3_30b_a3b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3-30B-A3B MoE.

    Recommended parallelism: TP=4, PP=2, EP=4.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-30B-A3B").to_megatron_provider(load_weights=False)

    # Tokenizer (--tokenizer-model)
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-30B-A3B"

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Parallelism settings (MoE-specific: includes expert_model_parallel_size)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # MoE Token Dispatcher settings
    # Note: moe_token_dispatcher_type may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # Options: None, deepep, hybridep
    cfg.model.moe_hybridep_num_sms = 16

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading) - ENABLED for 30B MoE
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # These are defaults for FP8, enable them if using FP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False

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
    # Note: moe_shared_expert_overlap may be overridden by apply_flex_dispatcher_backend at the end
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
    cfg.model.moe_router_force_load_balancing = False

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    return cfg


def qwen3_235b_a22b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3-235B-A22B MoE.

    Recommended parallelism: TP=4, PP=16, CP=2, EP=8.
    Note: Uses account_for_embedding_in_pipeline_split and account_for_loss_in_pipeline_split
    for proper layer distribution in pipeline parallelism.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-235B-A22B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-235B-A22B"

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 2
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # Pipeline split accounting
    cfg.model.account_for_embedding_in_pipeline_split = True
    cfg.model.account_for_loss_in_pipeline_split = True

    # MoE Token Dispatcher settings
    # Note: moe_token_dispatcher_type may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # Training config
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections (includes MoE-specific kernels)
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    # Enable if needed for memory optimization
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False  # MoE FP8 setting

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap (default None, can pass CommOverlapConfig for advanced overlap)
    # cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)  # Uncomment to enable
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    # Note: moe_shared_expert_overlap may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_shared_expert_overlap = False  # Overlap shared expert computation

    # Checkpoint config
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
    cfg.model.moe_router_force_load_balancing = False

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    return cfg


# =============================================================================
# Qwen3 MoE SFT (Full Fine-Tuning) Configs
# =============================================================================


def qwen3_30b_a3b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3-30B-A3B MoE.

    Recommended parallelism: TP=4, PP=2, EP=4 (1 node, 8 GPUs with SP=True)
    """
    cfg = _sft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-30B-A3B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-30B-A3B"

    # Parallelism settings (MoE-specific: includes expert parallelism)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # Global batch size is 32 for MoE packed sequences
    cfg.train.global_batch_size = 32
    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # MoE Token Dispatcher settings, may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = (
        "deepep"  # qwen3 moe has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
    )
    cfg.model.moe_hybridep_num_sms = 16

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 10
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Logger config
    cfg.logger.log_interval = 10

    # Optimizer and scheduler overrides for MoE
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections (includes MoE-specific kernels)
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _sft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False  # MoE FP8 setting

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap
    # cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)  # Uncomment to enable
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    # Note: moe_shared_expert_overlap may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_shared_expert_overlap = False

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _sft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below if using a pretrained checkpoint and provide path to the directory containing pretrained model for finetuning
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    # DDP config - MoE SFT uses these settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    return cfg


def qwen3_235b_a22b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3-235B-A22B MoE.

    Recommended parallelism: TP=4, PP=16, EP=4 (8 nodes, 64 GPUs with SP=True)
    Uses account_for_embedding_in_pipeline_split and account_for_loss_in_pipeline_split.
    """
    cfg = _sft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-235B-A22B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-235B-A22B"

    # Parallelism settings (MoE-specific: includes expert parallelism)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    # Pipeline split accounting (required for 235B model)
    cfg.model.account_for_embedding_in_pipeline_split = True
    cfg.model.account_for_loss_in_pipeline_split = True

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # Global batch size is 32 for MoE packed sequences
    cfg.train.global_batch_size = 32
    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # MoE Token Dispatcher settings, moe_token_dispatcher_type may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = (
        "deepep"  # qwen3 moe has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
    )
    cfg.model.moe_hybridep_num_sms = 16

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 10
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Logger config
    cfg.logger.log_interval = 10

    # Optimizer and scheduler overrides for MoE
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections (includes MoE-specific kernels)
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _sft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False  # MoE FP8 setting

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap
    # cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)  # Uncomment to enable
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    # Note: moe_shared_expert_overlap may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_shared_expert_overlap = False

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _sft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below if using a pretrained checkpoint and provide path to the directory containing pretrained model for finetuning
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    # DDP config - MoE SFT uses these settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    return cfg


# =============================================================================
# Qwen3 MoE PEFT (Parameter-Efficient Fine-Tuning) Configs
# =============================================================================


def qwen3_30b_a3b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen3-30B-A3B MoE.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=4, PP=1, EP=4 (1 node, 8 GPUs with SP=True)
    LoRA/DoRA uses dim=8, alpha=16, target_modules=['linear_qkv', 'linear_proj']
    """
    cfg = _peft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-30B-A3B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-30B-A3B"

    # Parallelism settings (MoE-specific: includes expert parallelism)
    # PEFT uses PP=1 (less parallelism needed since only adapters are trained)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    # MoE LoRA uses smaller dim and specific target modules
    peft_cfg = default_peft_config(peft_scheme)
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        peft_cfg.dim = 8
        peft_cfg.alpha = 16
        peft_cfg.target_modules = ["linear_qkv", "linear_proj"]
    cfg.peft = peft_cfg

    # Global batch size is 32 for MoE packed sequences
    cfg.train.global_batch_size = 32
    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # MoE Token Dispatcher settings, moe_token_dispatcher_type may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = (
        "deepep"  # qwen3 moe has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
    )
    cfg.model.moe_hybridep_num_sms = 16

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 10
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Logger config
    cfg.logger.log_interval = 10

    # Optimizer and scheduler overrides for MoE
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections (includes MoE-specific kernels)
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _sft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False  # MoE FP8 setting

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap
    # cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)  # Uncomment to enable
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    # Note: moe_shared_expert_overlap may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_shared_expert_overlap = False

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _peft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below if using a pretrained checkpoint and provide path to the directory containing pretrained model for finetuning
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    # DDP config - MoE PEFT uses these settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    return cfg


def qwen3_235b_a22b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen3-235B-A22B MoE.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=4, PP=4, EP=4 (8 nodes, 64 GPUs with SP=True)
    LoRA/DoRA uses dim=8, alpha=16, target_modules=['linear_qkv', 'linear_proj']
    Uses account_for_embedding_in_pipeline_split and account_for_loss_in_pipeline_split.
    """
    cfg = _peft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-235B-A22B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-235B-A22B"

    # Parallelism settings (MoE-specific: includes expert parallelism)
    # PEFT uses PP=4 (less than SFT's PP=16)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    # Pipeline split accounting (required for 235B model)
    cfg.model.account_for_embedding_in_pipeline_split = True
    cfg.model.account_for_loss_in_pipeline_split = True

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    # MoE LoRA uses smaller dim and specific target modules
    peft_cfg = default_peft_config(peft_scheme)
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        peft_cfg.dim = 8
        peft_cfg.alpha = 16
        peft_cfg.target_modules = ["linear_qkv", "linear_proj"]
    cfg.peft = peft_cfg

    # Global batch size is 32 for MoE packed sequences
    cfg.train.global_batch_size = 32
    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # MoE Token Dispatcher settings, moe_token_dispatcher_type may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = (
        "deepep"  # qwen3 moe has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
    )
    cfg.model.moe_hybridep_num_sms = 16

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 10
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Logger config
    cfg.logger.log_interval = 10

    # Optimizer and scheduler overrides for MoE
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections (includes MoE-specific kernels)
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _sft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False  # MoE FP8 setting

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap
    # cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)  # Uncomment to enable
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    # Note: moe_shared_expert_overlap may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_shared_expert_overlap = False

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _peft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below if using a pretrained checkpoint and provide path to the directory containing pretrained model for finetuning
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    # DDP config - MoE PEFT uses these settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    return cfg
