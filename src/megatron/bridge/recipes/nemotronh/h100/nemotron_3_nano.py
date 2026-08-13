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


from typing import cast

import torch
from megatron.core.activations import squared_relu

from megatron.bridge import AutoBridge
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_openmathinstruct2_config, default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


_NEMOTRON_3_NANO_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
# Public Nemotron 3.5 Lightning checkpoint used by the Lightning recipe API.
_NEMOTRON_3_5_LIGHTNING_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
_NEMOTRON_3_5_LIGHTNING_MODEL_REVISION = "b3caaabed0263651a17dc1f2d4ce97e794f76c44"  # pragma: allowlist secret
_OPENMATHINSTRUCT2_REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"  # pragma: allowlist secret


def _nemotron_3_nano_finetune_model() -> HybridModelProvider:
    """Build the Nemotron 3 Nano finetuning provider."""
    model = cast(
        HybridModelProvider,
        AutoBridge.from_hf_pretrained(_NEMOTRON_3_NANO_MODEL_ID).to_megatron_provider(load_weights=False),
    )

    model.seq_length = 2048
    model.apply_rope_fusion = False
    model.attention_backend = "fused"
    model.init_method_std = 0.0173
    model.use_fused_weighted_squared_relu = True
    model.calculate_per_token_loss = True
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.pipeline_dtype = torch.bfloat16
    model.virtual_pipeline_model_parallel_size = None
    model.context_parallel_size = 1
    model.sequence_parallel = False
    model.expert_tensor_parallel_size = 1
    model.expert_model_parallel_size = 8
    return model


def nemotron_3_nano_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return a pre-training config for Nemotron 3 Nano (30B-A3B MoE).

    This is a MoE (Mixture of Experts) model with the following default parallelism:
    - TP=1, PP=1, ETP=1, EP=8, SP=False
    - HybridEP enabled for MoE token dispatch

    Returns:
        ConfigContainer: Pre-training configuration for Nemotron 3 Nano.
    """
    cfg = _pretrain_common()

    # Model Configuration (MoE)
    cfg.model = HybridModelProvider(
        # Architecture (Nemotron 3 Nano 30B-A3B)
        hybrid_layer_pattern="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        num_layers=52,
        hidden_size=2688,
        mamba_num_heads=64,
        kv_channels=128,
        mamba_state_dim=128,
        ffn_hidden_size=1856,
        num_attention_heads=32,
        mamba_head_dim=64,
        seq_length=8192,
        num_query_groups=2,
        # MoE
        num_moe_experts=128,
        moe_ffn_hidden_size=1856,
        moe_shared_expert_intermediate_size=3712,
        moe_router_topk=6,
        moe_router_topk_scaling_factor=2.5,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        # NemotronH base
        mamba_num_groups=8,
        make_vocab_size_divisible_by=128,
        activation_func=squared_relu,
        masked_softmax_fusion=True,
        apply_query_key_layer_scaling=False,
        persist_layer_norm=True,
        attention_softmax_in_fp32=False,
        first_last_layers_bf16=True,
        is_hybrid_model=True,
        moe_aux_loss_coeff=0.0001,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=True,
        moe_shared_expert_overlap=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=8,
    )
    # Tokenizer (--tokenizer-model)
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_NANO_MODEL_ID

    # Dataset Configuration
    cfg.dataset.seq_length = 8192
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.mmap_bin_files = False

    # Parallelism Settings (MoE-specific)
    cfg.model.pipeline_model_parallel_layout = None

    # MoE Token Dispatcher Settings
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 16

    # Training Configuration
    cfg.train.train_iters = 39735
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # Transformer Engine (TE)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba"])
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    # Kernel Selections
    cfg.model.attention_backend = "fused"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory Saving (recompute & offloading)
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe", "layernorm"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # =========================================================================
    # FP8 & MXFP8 (Mixed Precision Settings)
    # =========================================================================
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # FP8 settings (disabled by default, uncomment to enable)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Optimizer Precision Settings
    # Match the measured BF16 performance recipe. The 16-GPU convergence
    # workload provides DP=2, so distributed optimizer state remains sharded
    # while retaining full FP32 optimizer precision.
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.mixed_precision = get_mixed_precision_config(cfg.mixed_precision)
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    # Optimizer hyperparameters
    cfg.optimizer.lr = 1.6e-3
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.min_lr = 1.6e-5
    cfg.scheduler.lr_warmup_iters = 333

    # Communication Overlap
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=True,
    )
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = False

    # Checkpoint Configuration
    # Paths are set in _pretrain_common by default. Override here if needed:
    # cfg.checkpoint.load = "path/to/load"
    # cfg.checkpoint.save = "path/to/save"
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.async_save = False
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    # DDP Configuration
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.check_for_large_grads = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.rerun_state_machine.check_for_nan_in_loss = True

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    cfg.model.init_method_std = 0.0173
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_weighted_squared_relu = True

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def nemotron_3_5_lightning_pretrain_config() -> ConfigContainer:
    """Return the Nemotron 3.5 Lightning BF16 pretraining config."""
    cfg = nemotron_3_nano_pretrain_8gpu_h100_bf16_config()
    cfg.train.global_batch_size = 512
    # Split the 8K sequence across two ranks so each MTP head materializes only
    # half of its vocabulary-loss workspace on an 80-GiB H100. P2P retains the
    # fused-attention path for this model's grouped-query layout.
    cfg.model.context_parallel_size = 2
    cfg.model.cp_comm_type = "p2p"
    cfg.model.recompute_modules = ["moe", "layernorm", "core_attn"]
    set_cuda_graph_modules(cfg.model, ["mamba"])
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.hf_model_id = _NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.model.hf_model_revision = _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
    return cfg


# =============================================================================
# SFT Config
# =============================================================================


def nemotron_3_nano_sft_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for Nemotron 3 Nano.

    Default parallelism: TP=1, PP=1, EP=8, SP=False

    Returns:
        ConfigContainer with all settings pre-configured for Nemotron Nano SFT.
    """
    cfg = _sft_common()

    cfg.model = _nemotron_3_nano_finetune_model()

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None

    # Sequence length
    cfg.model.seq_length = 2048

    # DeePEP settings - set to True to enable DeePEP (enabled by default for Nemotron)
    enable_deepep = True
    if enable_deepep:
        cfg.model.moe_token_dispatcher_type = "flex"
        cfg.model.moe_flex_dispatcher_backend = "deepep"
        cfg.model.moe_shared_expert_overlap = False
    else:
        cfg.model.moe_token_dispatcher_type = "alltoall"
        cfg.model.moe_flex_dispatcher_backend = None
        cfg.model.moe_shared_expert_overlap = True

    cfg.model.moe_hybridep_num_sms = 16

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "fused"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 settings
    # Note: mixed_precision="bf16_mixed" is set as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.model.moe_router_padding_for_fp8 = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    # Training config overrides
    cfg.validation.eval_interval = 500

    # Dataset config - enable_offline_packing=True by default (from _sft_common), seq_length=2048
    # _sft_common already sets seq_length=2048 and enable_offline_packing=True
    # Adjust pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.offline_packing_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides - Nemotron uses specific optimizer settings
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "cosine"

    # Tokenizer
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_NANO_MODEL_ID

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.ckpt_assume_constant_structure = True
    # Uncomment below if using a pretrained checkpoint and provide path to the directory containing pretrained model for finetuning
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    # Logger config
    cfg.logger.log_interval = 10
    cfg.logger.log_timers_to_tensorboard = False

    # RNG config - Nemotron uses seed 1234
    cfg.rng.seed = 1234

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True

    # Communication overlap settings(default None, can pass CommOverlapConfig for advanced overlap), uncomment to enable
    # cfg.comm_overlap = CommOverlapConfig(
    #     tp_comm_bootstrap_backend="nccl",
    #     tp_comm_overlap=True,
    # )
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def nemotron_3_5_lightning_sft_config() -> ConfigContainer:
    """Return a full SFT config for Nemotron 3.5 Lightning."""
    cfg = nemotron_3_nano_sft_8gpu_h100_bf16_config()
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.hf_model_id = _NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.model.hf_model_revision = _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
    return cfg


def nemotron_3_5_lightning_sft_openmathinstruct2_packed_config() -> ConfigContainer:
    """Return the verified 4K packed OpenMathInstruct-2 SFT config."""
    cfg = nemotron_3_5_lightning_sft_config()

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe", "layernorm", "core_attn", "mlp"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None

    cfg.dataset = default_openmathinstruct2_config(
        seq_length=4096,
        enable_offline_packing=True,
        pad_seq_to_mult=2,
    )
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _OPENMATHINSTRUCT2_REVISION}
    if cfg.dataset.offline_packing_specs is not None:
        cfg.dataset.offline_packing_specs.tokenizer_model_name = _NEMOTRON_3_5_LIGHTNING_MODEL_ID

    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.empty_unused_memory_level = 2

    cfg.mixed_precision = get_mixed_precision_config(cfg.mixed_precision)
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.optimizer.lr = 5e-6
    cfg.optimizer.min_lr = 0.0
    cfg.optimizer.overlap_param_gather = False
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0

    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.checkpoint.async_save = False
    cfg.checkpoint.save_interval = 100

    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.logger.tensorboard_dir = None
    cfg.ddp.average_in_collective = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_large_grads = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True
    cfg.dist.distributed_timeout_minutes = 120
    return cfg


# =============================================================================
# PEFT Config
# =============================================================================


def _nemotron_3_nano_peft_8gpu_h100_bf16_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Build a PEFT config for Nemotron 3 Nano.

    Default parallelism: TP=1, PP=1, EP=8, SP=False

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Nemotron Nano PEFT.
    """
    cfg = _peft_common()

    cfg.model = _nemotron_3_nano_finetune_model()

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None

    # Sequence length
    cfg.model.seq_length = 2048

    # DeePEP settings - set to True to enable DeePEP (enabled by default for Nemotron)
    enable_deepep = True
    if enable_deepep:
        cfg.model.moe_token_dispatcher_type = "flex"
        cfg.model.moe_flex_dispatcher_backend = "deepep"
        cfg.model.moe_shared_expert_overlap = False
    else:
        cfg.model.moe_token_dispatcher_type = "alltoall"
        cfg.model.moe_flex_dispatcher_backend = None
        cfg.model.moe_shared_expert_overlap = True

    cfg.model.moe_hybridep_num_sms = 16

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "fused"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 settings
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.model.moe_router_padding_for_fp8 = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    # PEFT config - Nemotron uses Mamba-specific target modules
    mamba_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme, target_modules=mamba_target_modules)
    elif isinstance(peft_scheme, PEFT):
        cfg.peft = peft_scheme
    else:
        # Default to LoRA with Mamba target modules
        cfg.peft = LoRA(
            target_modules=mamba_target_modules,
            dim=32,
            alpha=32,
            dropout=0.0,
            dropout_position="pre",
            lora_A_init_method="xavier",
            lora_B_init_method="zero",
        )

    # Training config overrides
    cfg.validation.eval_interval = 500

    # Dataset config - enable_offline_packing=True by default (from _peft_common), seq_length=2048
    # _peft_common already sets seq_length=2048 and enable_offline_packing=True
    # Adjust pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.offline_packing_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "cosine"

    # Tokenizer
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_NANO_MODEL_ID

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.ckpt_assume_constant_structure = True
    # Uncomment below if using a pretrained checkpoint and provide path to the directory containing pretrained model for finetuning
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    # Logger config
    cfg.logger.log_interval = 10
    cfg.logger.log_timers_to_tensorboard = False

    # RNG config - Nemotron uses seed 1234
    cfg.rng.seed = 1234

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True

    # Communication overlap settings(default None, can pass CommOverlapConfig for advanced overlap), uncomment to enable
    # cfg.comm_overlap = CommOverlapConfig(
    #     tp_comm_bootstrap_backend="nccl",
    #     tp_comm_overlap=True,
    # )
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def nemotron_3_nano_peft_8gpu_h100_bf16_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Nemotron 3 Nano."""
    cfg = _nemotron_3_nano_peft_8gpu_h100_bf16_config(peft_scheme)
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def nemotron_3_5_lightning_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Nemotron 3.5 Lightning."""
    cfg = nemotron_3_nano_peft_8gpu_h100_bf16_config(peft_scheme)
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.hf_model_id = _NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.model.hf_model_revision = _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
    return cfg


__all__ = [
    "nemotron_3_5_lightning_peft_config",
    "nemotron_3_5_lightning_pretrain_config",
    "nemotron_3_5_lightning_sft_config",
    "nemotron_3_5_lightning_sft_openmathinstruct2_packed_config",
    "nemotron_3_nano_peft_8gpu_h100_bf16_config",
    "nemotron_3_nano_pretrain_8gpu_h100_bf16_config",
    "nemotron_3_nano_sft_8gpu_h100_bf16_config",
]
