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
from megatron.core.quantization.quant_config import RecipeConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    deepseek_v4_supports_blackwell_fused_kernels,
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.recipes.common import _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_squad_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
    distributed_muon_with_cosine_annealing,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed, bf16_with_mxfp8_mixed


DEEPSEEK_V4_FLASH_HF_PATH = "deepseek-ai/DeepSeek-V4-Flash"


def _deepseek_v4_mxfp8_quant_recipe() -> RecipeConfig:
    """Use MXFP8 for training and BF16 for DSv4 validation/evaluation paths."""
    return RecipeConfig.from_config_dict(
        {
            "configs": {
                "mxfp8_evaluate_bf16": {
                    "transformer_engine_config_type": "TEQuantizationParams",
                    "training_recipe": {"fp8_quantization_recipe": "mxfp8"},
                    "evaluation_recipe": {},
                },
            },
            "matchers": {
                "all_te_linears": {
                    "config": "mxfp8_evaluate_bf16",
                    "type": "glob",
                    "pattern": "*",
                    "enabled": True,
                },
            },
        }
    )


def deepseek_v4_flash_pretrain_32gpu_h100_bf16_config() -> ConfigContainer:
    """Return the DeepSeek-V4-Flash Blackwell pre-training base config.

    Recommended Blackwell baseline: TP=1, PP=4, EP=8, CP=1.
    """
    use_fused_mhc = deepseek_v4_supports_blackwell_fused_kernels()
    cfg = _pretrain_common()
    cfg.model = AutoBridge.from_hf_pretrained(DEEPSEEK_V4_FLASH_HF_PATH, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )

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
    cfg.model.use_fused_mhc = use_fused_mhc
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
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def deepseek_v4_flash_pretrain_32gpu_h100_fp8mx_config() -> ConfigContainer:
    """Return the DeepSeek-V4-Flash Adam + MXFP8 pre-training config."""
    cfg = deepseek_v4_flash_pretrain_32gpu_h100_bf16_config()

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
    use_fused_mhc = deepseek_v4_supports_blackwell_fused_kernels()
    cfg.model.apply_dsa_kernel_fusion = False
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_mhc = use_fused_mhc
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
    # MCore uses a layer's training quantization recipe during eval unless an
    # evaluation_recipe is provided. Keep DSv4 MTP/validation eval in BF16
    # while training TE linears with MXFP8.
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = True
    cfg.model.mtp_eval_in_bf16 = True
    cfg.model.quant_recipe = _deepseek_v4_mxfp8_quant_recipe()
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def deepseek_v4_flash_pretrain_32gpu_h100_bf16_muon_config() -> ConfigContainer:
    """Return the DeepSeek-V4-Flash BF16 Muon pre-training config."""
    cfg = deepseek_v4_flash_pretrain_32gpu_h100_bf16_config()

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
    use_fused_mhc = deepseek_v4_supports_blackwell_fused_kernels()
    cfg.model.apply_dsa_kernel_fusion = False
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_mhc = use_fused_mhc
    cfg.model.dsa_indexer_loss_coeff = 0.0
    cfg.model.dsa_indexer_use_sparse_loss = False
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act", "mhc"]
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    opt_cfg, scheduler_cfg = distributed_muon_with_cosine_annealing(
        muon_momentum=0.95,
        muon_use_nesterov=True,
        muon_scale_mode="unit_rms_norm",
        muon_fp32_matmul_prec="highest",
        muon_num_ns_steps=5,
        muon_extra_scale_factor=0.2,
        lr_warmup_iters=2000,
        lr_decay_iters=cfg.train.train_iters,
        max_lr=2.7e-4,
        min_lr=2.7e-5,
        weight_decay=0.1,
        clip_grad=1.0,
    )
    # DSv4 Muon uses non-layer-wise optimizer dispatch.
    opt_cfg.optimizer = "muon"
    opt_cfg.adam_beta1 = 0.9
    opt_cfg.adam_beta2 = 0.95
    opt_cfg.adam_eps = 1e-20
    if hasattr(opt_cfg, "muon_coefficient_type"):
        opt_cfg.muon_coefficient_type = "quintic"

    scheduler_cfg.start_weight_decay = 0.1
    scheduler_cfg.end_weight_decay = 0.1
    scheduler_cfg.weight_decay_incr_style = "constant"

    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


# ---------------------------------------------------------------------------
# Supervised fine-tuning (SFT)
# ---------------------------------------------------------------------------


def deepseek_v4_flash_sft_32gpu_h100_bf16_config() -> ConfigContainer:
    """DeepSeek-V4-Flash full SFT, MTP enabled, Hopper-safe.

    Runs unchanged on Hopper (H100/H200) and Blackwell (B200/GB200). Fused mHC
    is enabled only on Blackwell. Full parameter training on unpacked (SBHD)
    sequences with Adam/bf16.
    """
    cfg = _sft_common()
    cfg.model = AutoBridge.from_hf_pretrained(DEEPSEEK_V4_FLASH_HF_PATH, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )

    # --- parallelism (DSv4 hybrid attention requires TP=1) ---
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.params_dtype = torch.bfloat16
    cfg.model.seq_length = 4096

    # --- attention / kernels: fused mHC on Blackwell, unfused mHC on Hopper, unfused DSA ---
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None
    cfg.model.apply_dsa_kernel_fusion = False
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_mhc = deepseek_v4_supports_blackwell_fused_kernels()
    cfg.model.dsa_indexer_loss_coeff = 0.0
    cfg.model.dsa_indexer_use_sparse_loss = False

    # --- MoE ---
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # --- memory (selective recompute, same as pretrain) ---
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act", "mhc"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.cuda_graph_impl = "none"

    # --- MTP enabled ---
    if getattr(cfg.model, "mtp_num_layers", None):
        cfg.model.mtp_loss_scaling_factor = 0.1

    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    # --- tokenizer / dataset (real HF tokenizer; SBHD / unpacked) ---
    cfg.tokenizer.tokenizer_model = DEEPSEEK_V4_FLASH_HF_PATH
    cfg.dataset = default_squad_config(seq_length=4096, enable_offline_packing=False)

    # --- robustness defaults ---
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.dist.enable_megatron_core_experimental = True
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def deepseek_v4_flash_no_mtp_sft_32gpu_h100_bf16_config() -> ConfigContainer:
    """DeepSeek-V4-Flash full SFT with the MTP layer disabled, Hopper-safe.

    Same as :func:`deepseek_v4_flash_sft_config` but drops the Multi-Token
    Prediction layer (fused mHC only on Blackwell, bf16, SBHD).
    """
    cfg = _sft_common()
    cfg.model = AutoBridge.from_hf_pretrained(DEEPSEEK_V4_FLASH_HF_PATH, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )

    # --- parallelism (DSv4 hybrid attention requires TP=1) ---
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.params_dtype = torch.bfloat16
    cfg.model.seq_length = 4096

    # --- attention / kernels: fused mHC on Blackwell, unfused mHC on Hopper, unfused DSA ---
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None
    cfg.model.apply_dsa_kernel_fusion = False
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_mhc = deepseek_v4_supports_blackwell_fused_kernels()
    cfg.model.dsa_indexer_loss_coeff = 0.0
    cfg.model.dsa_indexer_use_sparse_loss = False

    # --- MoE ---
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # --- memory (selective recompute, same as pretrain) ---
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act", "mhc"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.cuda_graph_impl = "none"

    # --- MTP disabled ---
    cfg.model.mtp_num_layers = None
    cfg.model.mtp_loss_scaling_factor = 0.0
    # Drop the hybrid MTP sub-pattern so finalize() does not append a (0-copy) MTP
    # section to hybrid_layer_pattern once MTP is off.
    cfg.model.mtp_hybrid_override_pattern = None
    # The bridge sizes csa_compress_ratios for the main hybrid layers plus one MTP
    # depth. With MTP off, len(csa_compress_ratios) must be >= num_layers (the hybrid
    # main-layer count, which transformer_config validates), so trim the MTP tail.
    ratios = getattr(cfg.model, "csa_compress_ratios", None)
    num_layers = getattr(cfg.model, "num_layers", None)
    if ratios is not None and num_layers is not None and len(ratios) > num_layers:
        cfg.model.csa_compress_ratios = list(ratios)[:num_layers]

    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    # --- tokenizer / dataset (real HF tokenizer; SBHD / unpacked) ---
    cfg.tokenizer.tokenizer_model = DEEPSEEK_V4_FLASH_HF_PATH
    cfg.dataset = default_squad_config(seq_length=4096, enable_offline_packing=False)

    # --- robustness defaults ---
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.dist.enable_megatron_core_experimental = True
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


# NOTE: the SFT recipes enable fused mHC on Blackwell and fused rope on all supported GPUs.
# The historical "fused-kernel SFT NaN" reports are both resolved: fused mHC was a confound,
# and the fused-rope NaN was a bridge config-mapping bug fixed by rotary_percent=1.0 (#4271);
# with that fix, full-model SFT with rope fusion matches the unfused control.
#
# NOTE: there are intentionally no MXFP8 or Muon *SFT* variants either. Both were prototyped
# (mirroring the pretrain recipes) but fail in full-model DSv4-Flash SFT — MXFP8 NaNs at iter-2
# (fp8 x hash-MoE / ClampedSwiGLU numerics) and Muon hits an iter-2 assertion (Muon + expert
# parallelism not yet supported upstream). Both are upstream blockers tracked in README Blockers;
# the pretrain MXFP8/Muon recipes remain. SFT ships Adam/bf16 (validated).


__all__ = [
    "deepseek_v4_flash_no_mtp_sft_32gpu_h100_bf16_config",
    "deepseek_v4_flash_pretrain_32gpu_h100_bf16_config",
    "deepseek_v4_flash_pretrain_32gpu_h100_bf16_muon_config",
    "deepseek_v4_flash_pretrain_32gpu_h100_fp8mx_config",
    "deepseek_v4_flash_sft_32gpu_h100_bf16_config",
]
