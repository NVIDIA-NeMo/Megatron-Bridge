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

"""Shared helpers for flat performance benchmark recipes.

``_benchmark_common`` applies throughput-measurement defaults.
``_perf_precision`` returns a mixed-precision config for a given dtype.
"""

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import (
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_delayed_scaling_mixed,
    bf16_with_mxfp8_mixed,
    bf16_with_nvfp4_mixed,
)


def _benchmark_common(
    cfg: ConfigContainer,
    cross_entropy_impl: str = "te",
) -> None:
    """Apply benchmark-mode defaults that prioritize throughput measurement over convergence.

    Intended for performance benchmark recipes only. Sets short training runs,
    disables checkpointing/eval, tunes scheduler, and enables perf-oriented kernels.

    This is the fixed-model-shape policy for flat performance recipes.
    Canonical recipes launched with ``scripts/performance --use_recipes``
    retain their own tokenizer vocabulary policy.

    Individual recipes may override any of these after calling this function
    (e.g. Kimi K2 sets ``grad_reduce_in_fp32 = True``).
    """
    cfg.train.train_iters = 50
    cfg.train.eval_iters = 0
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # Performance recipes benchmark a fixed model shape. Synthetic or runtime
    # tokenizers must not resize the embedding and output layers during setup.
    cfg.tokenizer.use_tokenizer_vocab_size = False

    cfg.checkpoint.save = None
    cfg.checkpoint.load = None

    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = None

    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.check_for_large_grads = False

    cfg.rerun_state_machine.check_for_nan_in_loss = False

    cfg.scheduler.lr_decay_iters = cfg.train.train_iters
    cfg.scheduler.lr_warmup_iters = 10

    if getattr(cfg.model, "num_moe_experts", None):
        cfg.model.moe_router_force_load_balancing = True

    if hasattr(cfg.model, "use_transformer_engine_op_fuser") and cfg.model.use_transformer_engine_op_fuser:
        cfg.model.use_transformer_engine_op_fuser = False
    cfg.model.apply_rope_fusion = True
    cfg.model.cross_entropy_fusion_impl = cross_entropy_impl

    if not isinstance(cfg.mixed_precision, str):
        cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # mcore may auto-promote cuda_graph_impl from "none" to "full_iteration" when
    # cuda_graph_scope contains "full_iteration", so consult both fields when
    # deciding whether CUDA graphs will actually run at training time.
    cuda_impl = getattr(cfg.model, "cuda_graph_impl", None)
    cuda_scope = getattr(cfg.model, "cuda_graph_scope", None) or []
    scope_names = {s if isinstance(s, str) else getattr(s, "name", "") for s in cuda_scope}
    graphs_active = (cuda_impl is not None and cuda_impl != "none") or "full_iteration" in scope_names
    if cuda_impl == "none":
        cfg.model.cuda_graph_scope = []
    if cuda_impl is not None or scope_names:
        cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = graphs_active

    if getattr(cfg.model, "moe_flex_dispatcher_backend", None) == "hybridep":
        cfg.model.moe_hybridep_num_sms = 32


def _enable_ncclep(cfg: ConfigContainer) -> None:
    """Switch a flex-dispatcher recipe to the NCCL EP backend (GB300 MoE default).

    Call after ``_benchmark_common``. Only the dispatch stack changes; parallelism, precision,
    recompute and the CUDA graph mode stay as the recipe set them. The calling recipe still declares
    its own ``cfg.env_vars`` inline and adds ``"NCCL_EP_HT_EM_PULL_PUSH": 1`` there.
    """
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "ncclep"


def _enable_full_iteration_cuda_graph(cfg: ConfigContainer) -> None:
    """Capture each training iteration in one CUDA graph, with static-shape buffers for dropless MoE.

    Dropless MoE produces per-expert tensors whose shapes change every step, which graph capture
    cannot record. Expert inputs are padded to a fixed per-rank capacity, and paged stashing
    recovers the memory that padding would otherwise hold.
    """
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True
    # The loss is produced inside the captured graph, so it cannot be checked for NaN.
    cfg.rerun_state_machine.check_for_nan_in_loss = False

    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash = True
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0


def _enable_moe_a2a_overlap(cfg: ConfigContainer) -> None:
    """Overlap MoE expert-parallel dispatch/combine with compute, and delay the expert wgrad compute."""
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
    cfg.comm_overlap.delay_wgrad_compute = True
    # Shared-expert overlap cannot be combined with expert-parallel A2A overlap.
    cfg.model.moe_shared_expert_overlap = False
    _tune_cutedsl_fused_grouped_mlp_for_moe_a2a_overlap(cfg)


def _enable_cutedsl_fused_grouped_mlp(cfg: ConfigContainer) -> None:
    """Run MoE experts through the Transformer Engine op fuser for the CuTeDSL fused grouped MLP.

    Call after ``_benchmark_common``, which turns the op fuser off. Recipes that use it also set
    ``"NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1`` in ``cfg.env_vars``.
    """
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32
    _tune_cutedsl_fused_grouped_mlp_for_moe_a2a_overlap(cfg)


def _tune_cutedsl_fused_grouped_mlp_for_moe_a2a_overlap(cfg: ConfigContainer) -> None:
    """Apply the A2A stream and HybridEP settings used when the fused grouped MLP runs under A2A overlap.

    Does nothing unless both features are enabled, so ``_enable_moe_a2a_overlap`` and
    ``_enable_cutedsl_fused_grouped_mlp`` can be called in either order.
    """
    if not cfg.model.use_transformer_engine_op_fuser:
        return
    if cfg.comm_overlap is None or not cfg.comm_overlap.overlap_moe_expert_parallel_comm:
        return
    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.moe_hybridep_num_sms_preprocessing = 32


def _perf_precision(compute_dtype: str):
    """Return mixed-precision config tuned for perf benchmarks.

    Identical to ``scripts/performance/utils/precision.get_precision_config``
    but importable from the recipe package. Always sets
    ``grad_reduce_in_fp32=False`` so that callers that replace
    ``cfg.mixed_precision`` after ``_benchmark_common()`` still get the
    benchmark-mode default.
    """
    if compute_dtype == "bf16":
        cfg = bf16_mixed()
    elif compute_dtype == "fp8_cs":
        cfg = bf16_with_fp8_current_scaling_mixed()
        cfg.first_last_layers_bf16 = False
    elif compute_dtype == "fp8_ds":
        cfg = bf16_with_fp8_delayed_scaling_mixed()
        cfg.first_last_layers_bf16 = False
    elif compute_dtype == "fp8_mx":
        cfg = bf16_with_mxfp8_mixed()
    elif compute_dtype == "nvfp4":
        cfg = bf16_with_nvfp4_mixed()
    else:
        raise ValueError(f"Unknown compute_dtype: {compute_dtype}")
    cfg.grad_reduce_in_fp32 = False
    return cfg
