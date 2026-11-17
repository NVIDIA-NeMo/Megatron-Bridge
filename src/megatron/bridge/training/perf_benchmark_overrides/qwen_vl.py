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

"""qwen_vl benchmark overrides applied to library recipes.

The literal names preserve existing CLI selectors. Common and model overrides
retain their ordering because graph, RNG and precision policies are dependent.
"""

from typing import Literal

import torch

from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    _apply_qwen35_vl_35b_a3b_16gpu_h100_execution_config,
    qwen35_vl_35b_a3b_pretrain_config,
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
from megatron.bridge.training.perf_benchmark_overrides._common import (
    _benchmark_common,
    _enable_ncclep,
    _perf_precision,
)
from megatron.bridge.training.perf_benchmark_overrides.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.utils.cuda_graph import clear_cuda_graph_modules, set_cuda_graph_modules


BENCHMARK_ENVIRONMENTS = {
    "qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    },
    "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "qwen3_vl_235b_a22b_pretrain_64gpu_gb300_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
}


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


def _apply_qwen35_vl_35b_a3b_gb300_bf16(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_common(cfg)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True, overlap_grad_reduce=False, overlap_param_gather=False)
    _benchmark_common(cfg)
    _qwen35_vl_post(cfg)
    _enable_partial_cuda_graphs(cfg)
    _select_gdn_kernel_backend(cfg, "transformer_engine")
    return cfg


def _apply_qwen35_vl_397b_a17b_gb300_fp8mx(cfg: ConfigContainer) -> ConfigContainer:
    apply_overrides(cfg, "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config")
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


LIBRARY_RECIPES = {
    "qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_b200_fp8cs_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_b200_fp8mx_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_b200_bf16_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_b200_fp8cs_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_b200_fp8mx_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_b200_bf16_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_b200_fp8cs_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_b200_fp8mx_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen3_vl_235b_a22b_pretrain_64gpu_b200_bf16_config": qwen3_vl_235b_a22b_pretrain_mock_config,
    "qwen3_vl_235b_a22b_pretrain_64gpu_b200_fp8cs_config": qwen3_vl_235b_a22b_pretrain_mock_config,
    "qwen3_vl_235b_a22b_pretrain_64gpu_b200_fp8mx_config": qwen3_vl_235b_a22b_pretrain_mock_config,
    "qwen3_vl_30b_a3b_pretrain_8gpu_b200_bf16_config": qwen3_vl_30b_a3b_pretrain_mock_config,
    "qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8cs_config": qwen3_vl_30b_a3b_pretrain_mock_config,
    "qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8mx_config": qwen3_vl_30b_a3b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_b300_fp8cs_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_b300_fp8mx_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_b300_bf16_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_b300_fp8cs_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_b300_fp8mx_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_b300_bf16_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_b300_fp8cs_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_b300_fp8mx_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_gb200_fp8cs_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_gb200_fp8mx_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8cs_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8mx_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_gb200_bf16_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_gb200_fp8cs_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_gb200_fp8mx_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen3_vl_235b_a22b_pretrain_64gpu_gb200_bf16_config": qwen3_vl_235b_a22b_pretrain_mock_config,
    "qwen3_vl_235b_a22b_pretrain_64gpu_gb200_fp8cs_config": qwen3_vl_235b_a22b_pretrain_mock_config,
    "qwen3_vl_235b_a22b_pretrain_64gpu_gb200_fp8mx_config": qwen3_vl_235b_a22b_pretrain_mock_config,
    "qwen3_vl_30b_a3b_pretrain_8gpu_gb200_bf16_config": qwen3_vl_30b_a3b_pretrain_mock_config,
    "qwen3_vl_30b_a3b_pretrain_8gpu_gb200_fp8cs_config": qwen3_vl_30b_a3b_pretrain_mock_config,
    "qwen3_vl_30b_a3b_pretrain_8gpu_gb200_fp8mx_config": qwen3_vl_30b_a3b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8cs_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8mx_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8mx_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen3_vl_235b_a22b_pretrain_64gpu_gb300_bf16_config": qwen3_vl_235b_a22b_pretrain_mock_config,
    "qwen3_vl_235b_a22b_pretrain_64gpu_gb300_fp8cs_config": qwen3_vl_235b_a22b_pretrain_mock_config,
    "qwen3_vl_235b_a22b_pretrain_64gpu_gb300_fp8mx_config": qwen3_vl_235b_a22b_pretrain_mock_config,
    "qwen3_vl_30b_a3b_pretrain_8gpu_gb300_bf16_config": qwen3_vl_30b_a3b_pretrain_mock_config,
    "qwen3_vl_30b_a3b_pretrain_8gpu_gb300_fp8cs_config": qwen3_vl_30b_a3b_pretrain_mock_config,
    "qwen3_vl_30b_a3b_pretrain_8gpu_gb300_fp8mx_config": qwen3_vl_30b_a3b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config": qwen35_vl_35b_a3b_pretrain_config,
    "qwen35_vl_35b_a3b_pretrain_16gpu_h100_fp8cs_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_256gpu_h100_bf16_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_256gpu_h100_fp8cs_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen3_vl_235b_a22b_pretrain_256gpu_h100_bf16_config": qwen3_vl_235b_a22b_pretrain_mock_config,
    "qwen3_vl_235b_a22b_pretrain_256gpu_h100_fp8cs_config": qwen3_vl_235b_a22b_pretrain_mock_config,
    "qwen3_vl_30b_a3b_pretrain_16gpu_h100_bf16_config": qwen3_vl_30b_a3b_pretrain_mock_config,
    "qwen3_vl_30b_a3b_pretrain_16gpu_h100_fp8cs_config": qwen3_vl_30b_a3b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_vrnvl72_bf16_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_vrnvl72_fp8cs_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_122b_a10b_pretrain_32gpu_vrnvl72_fp8mx_config": qwen35_vl_122b_a10b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_vrnvl72_bf16_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_vrnvl72_fp8cs_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_35b_a3b_pretrain_8gpu_vrnvl72_fp8mx_config": qwen35_vl_35b_a3b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_vrnvl72_bf16_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_vrnvl72_fp8cs_config": qwen35_vl_397b_a17b_pretrain_mock_config,
    "qwen35_vl_397b_a17b_pretrain_64gpu_vrnvl72_fp8mx_config": qwen35_vl_397b_a17b_pretrain_mock_config,
}
ALLOCATION_AWARE = ()


def apply_overrides(cfg: ConfigContainer, name: str) -> ConfigContainer:
    """Apply model-specific benchmark overrides to an existing library config."""
    match name:
        case "qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.expert_model_parallel_size = 8
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_shared_expert_overlap = False
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                overlap_moe_expert_parallel_comm=True,
                delay_wgrad_compute=True,
            )
            _benchmark_common(cfg)
            _qwen35_vl_post_clear_scope(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_b200_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_b200_fp8mx_config":
            apply_overrides(cfg, "qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True, overlap_grad_reduce=False, overlap_param_gather=False
            )
            _benchmark_common(cfg)
            _qwen35_vl_post(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_b200_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_35b_a3b_pretrain_8gpu_b200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_b200_fp8mx_config":
            apply_overrides(cfg, "qwen35_vl_35b_a3b_pretrain_8gpu_b200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.expert_model_parallel_size = 8
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_shared_expert_overlap = False
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                overlap_moe_expert_parallel_comm=True,
                delay_wgrad_compute=True,
            )
            _benchmark_common(cfg)
            _qwen35_vl_post_clear_scope(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_b200_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_397b_a17b_pretrain_64gpu_b200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_b200_fp8mx_config":
            apply_overrides(cfg, "qwen35_vl_397b_a17b_pretrain_64gpu_b200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_vl_235b_a22b_pretrain_64gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.num_layers_in_first_pipeline_stage = 10
            cfg.model.num_layers_in_last_pipeline_stage = 12
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_a2a_overlap = False
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_vl_235b_a22b_pretrain_64gpu_b200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.num_layers_in_first_pipeline_stage = 10
            cfg.model.num_layers_in_last_pipeline_stage = 12
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_a2a_overlap = False
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_vl_235b_a22b_pretrain_64gpu_b200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.num_layers_in_first_pipeline_stage = 10
            cfg.model.num_layers_in_last_pipeline_stage = 12
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_a2a_overlap = False
            cfg.ddp.overlap_param_gather = False
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_vl_30b_a3b_pretrain_8gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 32
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True, overlap_grad_reduce=False, overlap_param_gather=False
            )
            _benchmark_common(cfg)
            _qwen35_vl_post(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_b300_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_b300_fp8mx_config":
            apply_overrides(cfg, "qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 8
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True, overlap_grad_reduce=False, overlap_param_gather=False
            )
            _benchmark_common(cfg)
            _qwen35_vl_post(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_b300_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_35b_a3b_pretrain_8gpu_b300_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_b300_fp8mx_config":
            apply_overrides(cfg, "qwen35_vl_35b_a3b_pretrain_8gpu_b300_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 64
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True, overlap_grad_reduce=False, overlap_param_gather=False
            )
            _benchmark_common(cfg)
            _qwen35_vl_post(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_b300_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_397b_a17b_pretrain_64gpu_b300_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_b300_fp8mx_config":
            apply_overrides(cfg, "qwen35_vl_397b_a17b_pretrain_64gpu_b300_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True, overlap_grad_reduce=False, overlap_param_gather=False
            )
            _benchmark_common(cfg)
            _qwen35_vl_post(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_gb200_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_gb200_fp8mx_config":
            apply_overrides(cfg, "qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 480
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True, overlap_grad_reduce=False, overlap_param_gather=False
            )
            _benchmark_common(cfg)
            _qwen35_vl_post(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.train.micro_batch_size = 3
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8mx_config":
            apply_overrides(cfg, "qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.train.micro_batch_size = 3
            cfg.model.cuda_graph_impl = "transformer_engine"
            set_cuda_graph_modules(cfg.model, ["moe_router", "moe_preprocess"])
            cfg.model.use_te_rng_tracker = True
            cfg.rng.te_rng_tracker = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True, overlap_grad_reduce=False, overlap_param_gather=False
            )
            _benchmark_common(cfg)
            _qwen35_vl_post(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_gb200_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_397b_a17b_pretrain_64gpu_gb200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_gb200_fp8mx_config":
            apply_overrides(cfg, "qwen35_vl_397b_a17b_pretrain_64gpu_gb200_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_vl_235b_a22b_pretrain_64gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.num_layers_in_first_pipeline_stage = 10
            cfg.model.num_layers_in_last_pipeline_stage = 12
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_vl_235b_a22b_pretrain_64gpu_gb200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.num_layers_in_first_pipeline_stage = 10
            cfg.model.num_layers_in_last_pipeline_stage = 12
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_vl_235b_a22b_pretrain_64gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.num_layers_in_first_pipeline_stage = 10
            cfg.model.num_layers_in_last_pipeline_stage = 12
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_vl_30b_a3b_pretrain_8gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_vl_30b_a3b_pretrain_8gpu_gb200_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_vl_30b_a3b_pretrain_8gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 4
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 32
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 2
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True, overlap_grad_reduce=False, overlap_param_gather=False
            )
            _benchmark_common(cfg)
            _qwen35_vl_post(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8mx_config":
            apply_overrides(cfg, "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config":
            _apply_qwen35_vl_35b_a3b_gb300_bf16(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config":
            _apply_qwen35_vl_35b_a3b_gb300_bf16(cfg)
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config":
            _apply_qwen35_vl_35b_a3b_gb300_bf16(cfg)
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 72,
                "USE_MNNVL": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            }
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 32
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True, overlap_grad_reduce=False, overlap_param_gather=False
            )
            _benchmark_common(cfg)
            _qwen35_vl_post(cfg)
            _enable_partial_cuda_graphs(cfg)
            _select_gdn_kernel_backend(cfg, "transformer_engine")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8mx_config":
            _apply_qwen35_vl_397b_a17b_gb300_fp8mx(cfg)
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.model.vision_cuda_graph_impl = "transformer_engine"
            cfg.model.vision_cuda_graph_scope = ["attn", "mlp"]
            cfg.model.max_vision_cuda_graph_seq_length = 1029
            cfg.optimizer.use_precision_aware_optimizer = True
            cfg.optimizer.main_params_dtype = torch.float32
            cfg.optimizer.main_grads_dtype = torch.float32
            cfg.optimizer.exp_avg_dtype = torch.bfloat16
            cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
            cfg.model.recompute_granularity = "selective"
            cfg.model.recompute_modules = ["core_attn", "gdn_norm_out"]
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 72,
                "USE_MNNVL": 1,
                "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API": 128,
                "NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API": 128,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
            }
            return cfg
        case "qwen3_vl_235b_a22b_pretrain_64gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_vl_235b_a22b_pretrain_64gpu_gb300_bf16"].copy()
            return cfg
        case "qwen3_vl_235b_a22b_pretrain_64gpu_gb300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_vl_235b_a22b_pretrain_64gpu_gb300_bf16"].copy()
            return cfg
        case "qwen3_vl_235b_a22b_pretrain_64gpu_gb300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen3_vl_235b_a22b_pretrain_64gpu_gb300_bf16"].copy()
            return cfg
        case "qwen3_vl_30b_a3b_pretrain_8gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 8
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_vl_30b_a3b_pretrain_8gpu_gb300_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 8
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen3_vl_30b_a3b_pretrain_8gpu_gb300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 8
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 2
            cfg.model.expert_model_parallel_size = 16
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_shared_expert_overlap = False
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=False,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                overlap_moe_expert_parallel_comm=True,
                delay_wgrad_compute=True,
            )
            _benchmark_common(cfg)
            _qwen35_vl_post_clear_scope(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            _benchmark_common(cfg)
            _apply_qwen35_vl_35b_a3b_16gpu_h100_execution_config(cfg)
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_16gpu_h100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 10
            cfg.model.expert_model_parallel_size = 8
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.moe_shared_expert_overlap = False
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=True,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                overlap_moe_expert_parallel_comm=True,
                delay_wgrad_compute=True,
            )
            _benchmark_common(cfg)
            _qwen35_vl_post_clear_scope(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_256gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            _qwen35_vl_common(cfg)
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.expert_model_parallel_size = 32
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_shared_expert_overlap = False
            cfg.comm_overlap = CommOverlapConfig(
                tp_comm_overlap=False,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                overlap_moe_expert_parallel_comm=True,
                delay_wgrad_compute=True,
            )
            _benchmark_common(cfg)
            _qwen35_vl_post_clear_scope(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_256gpu_h100_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_397b_a17b_pretrain_256gpu_h100_bf16_config")
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_vl_235b_a22b_pretrain_256gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.expert_model_parallel_size = 32
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_a2a_overlap = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            _benchmark_common(cfg)
            _finalize_qwen3_vl_with_moe_a2a_overlap(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_vl_235b_a22b_pretrain_256gpu_h100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 8
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 4
            cfg.model.expert_model_parallel_size = 32
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 1024
            cfg.train.micro_batch_size = 1
            cfg.model.moe_a2a_overlap = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            _benchmark_common(cfg)
            _finalize_qwen3_vl_with_moe_a2a_overlap(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_vl_30b_a3b_pretrain_16gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 12
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
            cfg.model.moe_a2a_overlap = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl_with_moe_a2a_overlap(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen3_vl_30b_a3b_pretrain_16gpu_h100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.bias_activation_fusion = True
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.model.moe_router_fusion = True
            cfg.model.seq_length = 4096
            cfg.dataset.seq_length = 4096
            cfg.model.moe_router_force_load_balancing = True
            cfg.ddp.overlap_grad_reduce = False
            cfg.ddp.overlap_param_gather = False
            cfg.optimizer.overlap_param_gather = False
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 2
            cfg.model.context_parallel_size = 1
            cfg.model.virtual_pipeline_model_parallel_size = 12
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 512
            cfg.train.micro_batch_size = 1
            cfg.model.moe_a2a_overlap = True
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
            _benchmark_common(cfg)
            _finalize_qwen3_vl_with_moe_a2a_overlap(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_vrnvl72_bf16_config":
            apply_overrides(cfg, "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_vrnvl72_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8cs_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16"].copy()
            return cfg
        case "qwen35_vl_122b_a10b_pretrain_32gpu_vrnvl72_fp8mx_config":
            apply_overrides(cfg, "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8mx_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_vrnvl72_bf16_config":
            _apply_qwen35_vl_35b_a3b_gb300_bf16(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            _select_gdn_kernel_backend(cfg, "fla")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_vrnvl72_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config")
            _select_gdn_kernel_backend(cfg, "fla")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16"].copy()
            return cfg
        case "qwen35_vl_35b_a3b_pretrain_8gpu_vrnvl72_fp8mx_config":
            apply_overrides(cfg, "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config")
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.model.expert_model_parallel_size = 4
            cfg.train.micro_batch_size = 4
            cfg.train.global_batch_size = 1024
            cfg.dataset.num_workers = 8
            _select_gdn_kernel_backend(cfg, "transformer_engine")
            cfg.model.use_grouped_gemm_for_shared_expert = True
            cfg.model.moe_shared_expert_glu_interleave_size = 32
            cfg.model.overlap_dispatch_backward_with_experts_wgrad = True
            cfg.model.vision_cuda_graph_impl = "transformer_engine"
            cfg.model.vision_cuda_graph_scope = ["attn", "mlp"]
            cfg.model.max_vision_cuda_graph_seq_length = 3136
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
            }
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_vrnvl72_bf16_config":
            apply_overrides(cfg, "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config")
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            _select_gdn_kernel_backend(cfg, "transformer_engine")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_vrnvl72_fp8cs_config":
            apply_overrides(cfg, "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config")
            _select_gdn_kernel_backend(cfg, "fla")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16"].copy()
            return cfg
        case "qwen35_vl_397b_a17b_pretrain_64gpu_vrnvl72_fp8mx_config":
            _apply_qwen35_vl_397b_a17b_gb300_fp8mx(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.train.micro_batch_size = 1
            _select_gdn_kernel_backend(cfg, "transformer_engine")
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.model.recompute_granularity = "selective"
            cfg.model.recompute_modules = ["gdn_norm_out"]
            cfg.model.vision_cuda_graph_impl = "transformer_engine"
            cfg.model.vision_cuda_graph_scope = ["attn", "mlp"]
            cfg.model.max_vision_cuda_graph_seq_length = 784
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
            }
            return cfg
        case _:
            raise ValueError(f"Unknown benchmark overrides: {name}")
