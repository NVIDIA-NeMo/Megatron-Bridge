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

from typing import Literal

from megatron.bridge.perf_recipes._common import (
    _benchmark_common,
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
from megatron.bridge.utils.cuda_graph import clear_cuda_graph_modules, set_cuda_graph_modules


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
    """Apply VLM benchmark settings shared by Qwen3.5/Qwen3.6-VL.

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

    Qwen3.5/Qwen3.6-VL disable RoPE fusion and CUDA graphs for variable-length
    VLM inputs; these override the perf defaults that ``_benchmark_common`` sets.
    """
    cfg.model.apply_rope_fusion = False
    cfg.model.cuda_graph_impl = "none"
    clear_cuda_graph_modules(cfg.model)
    cfg.optimizer.overlap_param_gather = False


def _enable_partial_cuda_graphs(cfg: ConfigContainer) -> None:
    """Re-enable partial (per-layer) CUDA graphs on the language stack, with ``attn``.

    MUST be called AFTER :func:`_qwen35_vl_post`, which sets
    ``cuda_graph_impl="none"`` and clears the module list for the variable-shape
    real-data path. These benchmarks run on MOCK data with a fixed
    ``seq_length`` (4096) and ``moe_router_force_load_balancing=True``, so the
    variable-length concern that motivates disabling graphs does not apply here.

    ``attn`` is the module that matters: on 397B/64x GB300 under forced load
    balancing, adding the attention graph moved the step from 390.5 to 580.2
    TFLOP/s/GPU with GPU kernel time unchanged (+0.5%) and identical per-kernel
    instance counts -- the entire gain is per-launch CPU overhead that graph
    replay removes.

    ``set_cuda_graph_modules`` writes ``cuda_graph_modules`` and nulls
    ``cuda_graph_scope``, so the two never end up set at once (MCore asserts).
    """
    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "moe_router", "moe_preprocess"])
    # The RNG trackers must be re-enabled here, not left to an earlier caller.
    # ``_benchmark_common`` derives them from whatever ``cuda_graph_impl`` held at
    # ITS call time (_common.py:88-92: ``cfg.rng.te_rng_tracker =
    # cfg.model.use_te_rng_tracker = graphs_active``), and by then
    # ``_qwen35_vl_post`` has not yet run -- so with the graphs disabled the flags
    # land on False. Flipping ``cuda_graph_impl`` back on afterwards without them
    # trips MCore's "cuda_graph_impl != none requires use_te_rng_tracker" assertion
    # at model build. The h100 library recipe sets both for the same reason.
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True


def _select_gdn_kernel_backend(cfg: ConfigContainer, backend: Literal["transformer_engine", "fla"]) -> None:
    """Select the kernel for the GatedDeltaNet (GDN) layers when Megatron-Core exposes the choice.

    ``"transformer_engine"`` runs GDN through Transformer Engine's GatedDeltaNetAttention on cuDNN frontend
    kernels; ``"fla"`` is Megatron-Core's default FLA Triton kernel. Qwen3.5-VL runs 3 of every 4 language
    layers as GDN (30 of 40 at 35B-A3B, 45 of 60 at 397B-A17B), so with 16 microbatches per step the GB300
    35B and 397B recipes run 480 and 720 GDN forward+backward passes per rank per step.

    On GB300 at these recipes' shapes (35B: MBS 4 x 4096 tokens x 32 heads x 128; 397B: MBS 1 x 4096 x 64 x 128),
    cuDNN frontend 1.29 takes 54% (35B) and 39% (397B) less time than FLA for the GDN layer forward+backward. End to
    end with MXFP8, one run per arm on launch configurations close to (not exactly) these recipes, throughput rises
    by 9.6% at 397B-A17B on 64 GPUs (EP 32, forced load balancing) and by 12.0% to 19.4% at 35B-A3B on 16 GPUs
    (EP 4, real routing, two images).

    The cuDNN path needs a Megatron-Core with ``TransformerConfig.gdn_kernel_backend`` (NVIDIA/Megatron-LM#6645,
    re-landing as NVIDIA/Megatron-LM#7583); transformer-engine >= 2.19; nvidia-cudnn-frontend >= 1.29.0 (older GDN
    kernels can return NaN); and nvidia-cutlass-dsl >= 4.7.0. With an older CuTe DSL, cuDNN silently runs GDN on its
    cuTile engine, which takes 86% longer than FLA at the 35B shape. Transformer Engine 2.19 also raises under the FP8
    autocast that the FP8-CS and MXFP8 recipes enable unless Megatron-Core turns it off around the GDN call, which
    #7583 does not; 2.20.2 and later ignore FP8 autocast in GDN. ``ConfigContainer.validate`` warns about missing or
    too-old packages, and about Transformer Engine 2.19 with FP8 or FP4, instead of this helper switching backends, so
    a recipe builds the same config in every environment. Compare against FLA with ``model.gdn_kernel_backend=fla``.

    Assigning an unknown field on the model config does not raise: on a Megatron-Core without the field it would
    create an unused attribute and leave the recipe looking enabled while running FLA, so the assignment is
    guarded, as in ``_enable_gdn_conv_fusion`` (``recipes/qwen_vl/h100/qwen35_vl.py``).
    """
    model = cfg.model
    if hasattr(type(model), "gdn_kernel_backend") or hasattr(model, "gdn_kernel_backend"):
        model.gdn_kernel_backend = backend


def _qwen35_vl_post_clear_scope(cfg: ConfigContainer) -> None:
    """Apply Qwen3.5/Qwen3.6-VL post-overrides and clear graph scope."""
    _qwen35_vl_post(cfg)
    cfg.model.cuda_graph_scope = []


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


def _finalize_qwen3_vl_with_moe_a2a_overlap(cfg: ConfigContainer) -> None:
    """Apply Qwen3-VL perf defaults with MoE A2A overlap enabled."""
    _finalize_qwen3_vl(cfg)
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.model.moe_shared_expert_overlap = False
