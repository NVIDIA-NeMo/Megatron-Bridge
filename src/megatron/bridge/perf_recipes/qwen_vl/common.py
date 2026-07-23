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
# ruff: noqa: F401
"""Common helpers for qwen_vl performance recipes."""

from megatron.bridge.perf_recipes._common import (
    _benchmark_common,
    _enable_overlap_param_gather_with_optimizer_step,
    _perf_precision,
)
from megatron.bridge.recipes.qwen_vl.qwen3_vl import (
    qwen3_vl_30b_a3b_pretrain_mock_config,
    qwen3_vl_235b_a22b_pretrain_mock_config,
)
from megatron.bridge.recipes.qwen_vl.qwen35_vl import (
    qwen35_vl_35b_a3b_pretrain_mock_config,
    qwen35_vl_122b_a10b_pretrain_mock_config,
    qwen35_vl_397b_a17b_pretrain_mock_config,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


def _use_model_vocab_null_tokenizer(cfg: ConfigContainer) -> None:
    """Use a model-sized synthetic tokenizer for Qwen-VL performance runs."""
    if cfg.model.vocab_size is None:
        raise ValueError("Qwen-VL performance recipes require a model vocabulary size.")
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = cfg.model.vocab_size
    # The synthetic tokenizer mirrors the fixed benchmark model shape; it does
    # not define a new tokenizer-derived vocabulary for from-scratch training.
    cfg.tokenizer.use_tokenizer_vocab_size = False


def _qwen35_vl_common(cfg: ConfigContainer) -> None:
    """Apply VLM-specific performance benchmark settings for Qwen3.5-VL.

    Must be called before ``_benchmark_common`` and after setting precision.
    """
    _use_model_vocab_null_tokenizer(cfg)
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = []
    cfg.model.moe_router_fusion = True

    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096

    cfg.model.moe_router_force_load_balancing = True

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False


def _qwen35_vl_post(cfg: ConfigContainer) -> None:
    """VLM post-overrides that must run after ``_benchmark_common``.

    Qwen3.5-VL disables RoPE fusion and CUDA graphs for VLM variable-length
    inputs; these override the perf defaults that ``_benchmark_common`` sets.
    """
    cfg.model.apply_rope_fusion = False
    cfg.model.cuda_graph_impl = "none"
    cfg.optimizer.overlap_param_gather = False


def _qwen35_vl_post_with_overlap(cfg: ConfigContainer) -> None:
    """Apply Qwen3.5-VL post-overrides and optimizer-step param-gather overlap."""
    _qwen35_vl_post(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)


def _qwen35_vl_post_clear_scope_with_overlap(cfg: ConfigContainer) -> None:
    """Apply Qwen3.5-VL post-overrides, clear graph scope, and enable overlap."""
    _qwen35_vl_post(cfg)
    cfg.model.cuda_graph_scope = []
    _enable_overlap_param_gather_with_optimizer_step(cfg)


def _finalize_qwen3_vl(cfg: ConfigContainer) -> None:
    """Apply Qwen3-VL perf defaults that must override generic benchmark defaults."""
    _use_model_vocab_null_tokenizer(cfg)
    # _benchmark_common sets apply_rope_fusion=True; Qwen3-VL asserts it must be False
    # (per-token absolute positional frequencies are incompatible with TE's fused RoPE).
    cfg.model.apply_rope_fusion = False

    # Keep flat recipes aligned with the legacy Qwen3-VL performance path:
    # attn scope is not compatible with Qwen3VLModel CUDA graph capture.
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.model.expert_tensor_parallel_size = 1

    cfg.comm_overlap.overlap_param_gather = False
    cfg.comm_overlap.overlap_grad_reduce = False


def _finalize_qwen3_vl_with_overlap(cfg: ConfigContainer) -> None:
    """Apply Qwen3-VL perf defaults with optimizer-step param-gather overlap."""
    _finalize_qwen3_vl(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)


def _finalize_qwen3_vl_with_moe_a2a_overlap(cfg: ConfigContainer) -> None:
    """Apply Qwen3-VL perf defaults with MoE A2A overlap enabled."""
    _finalize_qwen3_vl(cfg)
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.model.moe_shared_expert_overlap = False


def _finalize_qwen3_vl_with_moe_a2a_and_overlap(cfg: ConfigContainer) -> None:
    """Apply Qwen3-VL perf defaults with MoE A2A and optimizer-step overlap."""
    _finalize_qwen3_vl_with_moe_a2a_overlap(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
