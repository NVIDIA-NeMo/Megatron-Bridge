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
from megatron.bridge.models import GPTModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


def _build_standalone_mtp_layout(num_decoder_layers: int, total_stages: int, mtp_layers: int) -> list[list[str]]:
    if mtp_layers <= 0:
        raise ValueError("standalone MTP layout requires mtp_num_layers > 0")
    if total_stages < 3:
        raise ValueError("standalone MTP layout requires at least three PP/VPP stages")

    decoder_stage_count = total_stages - 2
    decoder_layers_per_stage, extra_decoder_layers = divmod(num_decoder_layers, decoder_stage_count)

    layout = []
    for stage_idx in range(decoder_stage_count):
        num_layers = decoder_layers_per_stage + (1 if stage_idx < extra_decoder_layers else 0)
        stage = ["decoder"] * num_layers
        if stage_idx == 0:
            stage = ["embedding"] + stage
        layout.append(stage)

    layout.append(["mtp"] * mtp_layers)
    layout.append(["loss"])
    return layout


def set_deepseek_v3_pipeline_model_parallel_layout(
    model_cfg: GPTModelProvider, layout: str | list[list[str]] | None = None, *, mtp_standalone: bool = False
) -> None:
    """Set the DeepSeek-V3 pipeline model parallel layout.

    Args:
        model_cfg: DeepSeek-V3 model configuration to update.
        layout: Explicit pipeline layout. When provided, this overrides the predefined layouts.
        mtp_standalone: Place MTP layers in a standalone penultimate PP/VPP stage and loss in the
            final stage. Defaults to colocating MTP with loss, matching existing recipes.
    """
    if layout is not None:
        model_cfg.pipeline_model_parallel_layout = layout
        return

    mtp_layers = getattr(model_cfg, "mtp_num_layers", 1) or 0
    last_layer = ["mtp"] * mtp_layers + ["loss"]
    pp_size = model_cfg.pipeline_model_parallel_size or 1
    vp_size = model_cfg.virtual_pipeline_model_parallel_size or 1
    layout_map = {
        (1, 1): None,
        (4, 1): [["embedding"] + ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 13 + last_layer],
        (8, 1): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + last_layer],
        (4, 2): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + last_layer],
        (16, 1): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder"] + last_layer],
        (8, 2): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder"] + last_layer],
        (4, 4): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder"] + last_layer],
    }
    if mtp_standalone:
        num_decoder_layers = getattr(model_cfg, "num_layers", None)
        if not isinstance(num_decoder_layers, int) or isinstance(num_decoder_layers, bool) or num_decoder_layers <= 0:
            raise ValueError("standalone MTP layout requires model config num_layers to be a positive integer")
        model_cfg.pipeline_model_parallel_layout = _build_standalone_mtp_layout(
            num_decoder_layers=num_decoder_layers,
            total_stages=pp_size * vp_size,
            mtp_layers=mtp_layers,
        )
    elif (pp_size, vp_size) in layout_map:
        model_cfg.pipeline_model_parallel_layout = layout_map[(pp_size, vp_size)]


def deepseek_v3_pretrain_1024gpu_h100_bf16_config() -> ConfigContainer:
    """Return a pre-training config for DeepSeek-V3 (671B).

    Recommended parallelism: TP=2, PP=16, EP=64.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V3").to_megatron_provider(load_weights=False)

    # Tokenizer - uses NullTokenizer by default (no HF tokenizer download needed)
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Parallelism settings (MoE-specific: includes expert_model_parallel_size)
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.pipeline_model_parallel_layout = None  # Will be set by set_deepseek_v3_pipeline_model_parallel_layout
    cfg.model.pipeline_dtype = torch.bfloat16  # Required for PP > 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64  # MoE-specific: Expert parallelism
    cfg.model.expert_tensor_parallel_size = 1  # MoE-specific: Expert tensor parallelism
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096

    # MTP (Multi-Token Prediction) configuration
    cfg.model.mtp_num_layers = 1  # Set to 0 or None to disable MTP
    cfg.model.mtp_loss_scaling_factor = 0.1

    # Model-specific settings
    cfg.model.init_method_std = 0.006
    cfg.model.rotary_base = 10000.0
    cfg.model.rotary_scaling_factor = 40
    cfg.model.rotary_base = float(cfg.model.rotary_base)  # Ensure rotary_base is float
    cfg.model.rotary_scaling_factor = int(cfg.model.rotary_scaling_factor)

    # Pipeline split settings (asymmetric stages handled by layout)
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    # Set pipeline layout
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    # MoE Token Dispatcher settings
    # Note: moe_token_dispatcher_type may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"  # Options: None, deepep, hybridep
    cfg.model.moe_hybridep_num_sms = 16  # Number of SMs for hybridep backend

    # Training config (DIFFERENT from _pretrain_common)
    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5  # Different from default 100
    cfg.train.manual_gc_eval = 5

    # Scheduler config (DIFFERENT from _pretrain_common: lr_warmup_iters=2000 vs 500)
    cfg.scheduler.lr_warmup_iters = 2000

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections (includes MoE-specific kernels)
    cfg.model.attention_backend = None  # None means auto selection
    cfg.model.moe_router_fusion = False  # MoE-specific: Fuse router computation
    cfg.model.moe_permute_fusion = True  # MoE-specific: Fuse permute operations
    cfg.model.moe_grouped_gemm = True  # MoE-specific: Use grouped GEMM for experts
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"  # Default from DeepSeekModelProvider

    # Memory saving (recompute & offloading) — no recompute by default.
    # Setting granularity="selective" with modules=None would cause MCore's
    # post-init default-fill (transformer_config.py) to silently fill
    # recompute_modules with ["core_attn"], giving a surprise recompute
    # across all layers. Workloads that want recompute install it via
    # their flat perf recipe (src/megatron/bridge/perf_recipes/...), or users can
    # enable it via argparse (--recompute_modules ...) or Hydra.
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = []

    # Mixed precision - DeepSeek V3 uses custom MixedPrecisionConfig (NOT "bf16_mixed" string)
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
    # FP8 settings (commented - enable if using FP8)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = False  # Pad router for FP8 alignment

    # Optimizer settings - DeepSeek V3 uses precision-aware optimizer with bf16 moments
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16  # Different from default float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16  # Different from default float32
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16  # Different from default float32

    # Communication overlap
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    # Note: moe_shared_expert_overlap may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_shared_expert_overlap = True  # Default from DeepSeekModelProvider

    # Checkpoint config (DIFFERENT from _pretrain_common: save_interval=2000 vs 500)
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config (DIFFERENT: grad_reduce_in_fp32=False)
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = False  # Different from default True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    if cfg.model.apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True  # mla rope fusion is experimental

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        # Model-specific Transformer Engine tuning.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # HybridEP topology for this recipe.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


def deepseek_v3_pretrain_256gpu_h100_bf16_32nodes_config() -> ConfigContainer:
    """Return a pre-training config for DeepSeek-V3 (671B) with minimal nodes (32).

    Recommended parallelism: TP=2, PP=8, EP=32.
    Uses full recompute for memory efficiency.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V3").to_megatron_provider(load_weights=False)

    # Tokenizer - uses NullTokenizer by default (no HF tokenizer download needed)
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Parallelism settings (32 nodes configuration)
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32  # Reduced for 32 nodes
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096

    # MTP (Multi-Token Prediction) configuration
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # Model-specific settings
    cfg.model.init_method_std = 0.006
    cfg.model.rotary_base = 10000.0
    cfg.model.rotary_scaling_factor = 40
    cfg.model.rotary_base = float(cfg.model.rotary_base)
    cfg.model.rotary_scaling_factor = int(cfg.model.rotary_scaling_factor)

    # Pipeline split settings
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    # Set pipeline layout
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    # MoE Token Dispatcher settings
    # Note: moe_token_dispatcher_type may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"  # Options: None, deepep, hybridep
    cfg.model.moe_hybridep_num_sms = 16

    # Training config
    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 2000

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

    # Memory saving - FULL recompute for 32 nodes (memory efficiency)
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = []

    # Mixed precision - DeepSeek V3 uses custom MixedPrecisionConfig
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
    cfg.model.moe_router_padding_for_fp8 = False

    # Optimizer settings - precision-aware optimizer with bf16 moments
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    # Communication overlap
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    # Note: moe_shared_expert_overlap may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_shared_expert_overlap = True

    # Checkpoint config
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    if cfg.model.apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True  # mla rope fusion is experimental

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        # Model-specific Transformer Engine tuning.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # HybridEP topology for this recipe.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


__all__ = [
    "deepseek_v3_pretrain_1024gpu_h100_bf16_config",
    "deepseek_v3_pretrain_256gpu_h100_bf16_32nodes_config",
]
