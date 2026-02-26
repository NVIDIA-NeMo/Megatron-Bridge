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
from megatron.bridge.recipes.common import _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_squad_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import bf16_mixed


def qwen3_next_80b_a3b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3-Next 80B-A3B.

    Recommended parallelism: TP=1, PP=4, EP=8.
    Note: Qwen3-Next supports Multi-Token Prediction (MTP) with mtp_num_layers and mtp_loss_scaling_factor.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct").to_megatron_provider(
        load_weights=False
    )

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.mmap_bin_files = False  # Qwen3-Next specific setting

    # Parallelism settings (MoE-specific: includes expert_model_parallel_size)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # Multi-Token Prediction (MTP) settings - Qwen3-Next specific
    cfg.model.mtp_num_layers = 1  # Number of MTP layers (0 to disable)
    cfg.model.mtp_loss_scaling_factor = 0.1  # Loss scaling factor for MTP

    # MoE Token Dispatcher settings
    # Note: moe_token_dispatcher_type may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_token_dispatcher_type = "alltoall"  # Options: alltoall, allgather, flex
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # Options: None, deepep, hybridep
    cfg.model.moe_hybridep_num_sms = 16  # Number of SMs for hybridep backend

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
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (recompute & offloading)
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


def qwen3_next_80b_a3b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3-Next 80B-A3B.

    Recommended parallelism: TP=1, PP=2, EP=8
    Note: Packed sequence is NOT supported for Qwen3-Next.
    Note: Qwen3-Next uses no_weight_decay_cond_type = "qwen3_next" for scheduler.

    Returns:
        ConfigContainer with all settings pre-configured for Qwen3-Next 80B-A3B SFT.
    """
    # Get base SFT config
    cfg = _sft_common()

    # Override dataset - Qwen3-Next does NOT support packed_sequence
    cfg.dataset = default_squad_config(seq_length=2048, packed_sequence=False, pad_seq_to_mult=1)

    # Model config from HuggingFace
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct").to_megatron_provider(
        load_weights=False
    )

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    # Sequence length
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Parallelism settings (MoE-specific)
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    # Note: moe_flex_dispatcher_backend may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_flex_dispatcher_backend = (
        "deepep"  # qwen3_next has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
    )
    cfg.model.moe_hybridep_num_sms = 16

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 1000
    cfg.validation.eval_interval = 30
    cfg.validation.eval_iters = 32
    cfg.train.global_batch_size = 64  # packed_sequence=False, so use 64
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Scheduler config - Qwen3-Next specific
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.lr_decay_iters = None  # Will use train_iters
    cfg.scheduler.max_lr = 5e-6
    cfg.scheduler.no_weight_decay_cond_type = "qwen3_next"

    # Optimizer min_lr - Qwen3-Next uses same value as max_lr
    cfg.optimizer.min_lr = 5e-6

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
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (recompute & offloading) - Qwen3-Next uses selective recompute
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["layernorm", "moe", "moe_act"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
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

    # MoE Overlap settings, may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_shared_expert_overlap = False

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    # Uncomment below if using a pretrained checkpoint and provide path to the directory containing pretrained model for finetuning
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.use_megatron_fsdp = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def qwen3_next_80b_a3b_peft_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Qwen3-Next 80B-A3B.

    Note: PEFT is NOT currently supported for Qwen3-Next models.
    This function raises NotImplementedError.

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Raises:
        NotImplementedError: PEFT is not supported for Qwen3-Next models.
    """
    raise NotImplementedError(
        "PEFT is not currently supported for Qwen3-Next models. "
        "Only full SFT is available via qwen3_next_80b_a3b_sft_config()."
    )
