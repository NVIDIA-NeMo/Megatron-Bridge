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

from megatron.bridge.recipes.deepseek.deepseek_v3 import (
    deepseek_v3_pretrain_config as pretrain_config,
)
from megatron.bridge.recipes.deepseek.deepseek_v3 import (
    set_deepseek_v3_pipeline_model_parallel_layout,
)
from megatron.bridge.recipes.deepseek.deepseek_v4 import (
    deepseek_v4_pro_pretrain_mxfp8_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.cuda_graph import is_full_iteration_cuda_graph


logger = logging.getLogger(__name__)


def set_deepseek_v3_common_configs(cfg: ConfigContainer, moe_a2a_overlap: bool = False) -> None:
    """Set common performance configurations for all DeepSeek-V3 configs."""
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = False


def set_full_iter_cg_configs(cfg: ConfigContainer) -> None:
    """Apply defaults required by full-iteration CUDA graph capture with dropless MoE.

    Dropless MoE produces variable-shaped per-expert tensors that CG cannot
    capture; we pad to a fixed capacity (pad_experts + capacity factor) and use
    MCore PR #4247 paged stashing to recover memory. Callers should gate on
    `is_full_iteration_cuda_graph(cfg.model)`.
    """
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_expert_rank_capacity_factor = 4
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0


def deepseek_v3_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    if cfg.mixed_precision.fp8_recipe == "mxfp8":
        cfg.model.fp8_output_proj = True

    # Apply model-specific settings that were previously passed as constructor args
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute layout based on updated PP/VP sizes
        set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    if cfg.mixed_precision.fp8_recipe == "mxfp8":
        cfg.model.fp8_output_proj = True

    # Apply model-specific settings that were previously passed as constructor args
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute layout based on updated PP/VP sizes
        set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_pretrain_config_vr200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v2"
) -> ConfigContainer:
    """VR200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="vr200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    # Apply model-specific settings that were previously passed as constructor args
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute layout based on updated PP/VP sizes
        set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    if cfg.mixed_precision.fp8_recipe == "mxfp8":
        cfg.model.fp8_output_proj = True

    # Apply model-specific settings that were previously passed as constructor args
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    # Recompute layout based on updated PP/VP sizes
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    # Apply model-specific settings that were previously passed as constructor args
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    # Recompute layout based on updated PP/VP sizes
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    cfg.mixed_precision.fp4_param_gather = False

    return cfg


def deepseek_v3_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    # Apply model-specific settings that were previously passed as constructor args
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute layout based on updated PP/VP sizes
        set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    # Disabling to avoid functional errors. TODO: Test with it enabled and keep it enabled if it works.
    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg


def set_deepseek_v4_pro_common_configs(cfg: ConfigContainer) -> None:
    """Set the full-scale DeepSeek-V4-Pro performance knobs.

    Call after ``set_workload_base_configs``/``set_full_iter_cg_configs`` so
    these settings win over
    ``_set_common_perf_overrides`` (which forces the TE op fuser off and the
    cross-entropy fusion impl back to ``te``).
    """
    cfg.model.moe_router_force_load_balancing = True

    # Fused DSA sparse attention (FlashMLA forward + cuDNN DSA backward) — the
    # dominant perf lever over the unfused PyTorch DSA path.
    cfg.model.apply_dsa_kernel_fusion = True

    # TE op fuser + native cross-entropy fusion (dev backbone disables the "te"
    # cross-entropy fusion path for DSv4).
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # cuteDSL fused grouped-MLP interleave (clamped SwiGLU fusion path).
    cfg.model.moe_mlp_glu_interleave_size = 32

    # Native global MXFP8 (matching the MLM reference), overriding the lib mxfp8 config's
    # eval-oriented choices. The lib bundles a per-layer "kitchen" quant_recipe
    # (TEQuantizationParams, MXFP8-train/BF16-eval) with fp8_param_gather=False; that exists
    # only for DSv4 MTP/validation BF16 eval, which a perf benchmark (eval_iters=0) doesn't
    # use. Perf wants standard TE MXFP8 with fp8 param gather ON (perf-optimal, = MLM).
    cfg.model.quant_recipe = None
    cfg.mixed_precision.fp8_param_gather = True
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = True

    # Train the DSA/CSA indexer, matching the MLM reference. The library recipe
    # zeroes the indexer auxiliary loss for evaluation-oriented use, which leaves
    # the sparse-token selector without a direct training signal. A faithful
    # performance configuration keeps it on.
    cfg.model.dsa_indexer_loss_coeff = 0.01
    cfg.model.dsa_indexer_use_sparse_loss = True

    # No CPU (pinned host) paged-stash spill buffer, matching the MLM reference
    # (cpu factor 0.0; cuda factor stays 1.2). set_full_iter_cg_configs defaults
    # this to 1.0, which mirrors the multi-tens-of-GB stash working set into
    # page-locked host RAM per rank -- with 4 ranks/GB300 node that blows the
    # host cgroup (OOM-killed at iter 2). The 1.2x HBM buffer holds the stash alone.
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 0.0

    # BF16 precision-aware optimizer master gradients, matching the MLM reference.
    # The lib forces main_grads_dtype=fp32 (deepseek_v4.py); MLM uses bf16. With the
    # precision-aware optimizer (enabled here) bf16 master grads are valid and halve
    # the master-grad buffer. grad_reduce_in_fp32 (the DDP reduce path) is already
    # False above -- this is the separate optimizer-side knob.
    cfg.optimizer.main_grads_dtype = torch.bfloat16

    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False


def deepseek_v4_pro_pretrain_config_gb300(
    precision: str = "fp8_mx", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """Build the full 61-layer DeepSeek-V4-Pro GB300 performance config."""
    if precision != "fp8_mx":
        raise NotImplementedError(
            "DeepSeek-V4-Pro performance configs currently support precision='fp8_mx' "
            f"(MXFP8) only; got {precision!r}."
        )

    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v4_pro",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )

    cfg = deepseek_v4_pro_pretrain_mxfp8_config()

    # Pre-apply the validated full-scale parallelism before common overrides.
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.expert_model_parallel_size = base_cfg.expert_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout

    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)
    set_deepseek_v4_pro_common_configs(cfg)

    # Fine-grained activation offloading of attention activations matches the
    # validated full-scale GB300 run.
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["core_attn", "attn_proj"]
    cfg.model.fine_grained_offloading_max_inflight_offloads = 2

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg
