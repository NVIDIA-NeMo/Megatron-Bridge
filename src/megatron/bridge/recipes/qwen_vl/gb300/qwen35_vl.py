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

"""GB300 recipes for Qwen3.5/Qwen3.6-VL."""

from __future__ import annotations

import torch

from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_35b_a3b_pretrain_8gpu_h100_bf16_mock_config,
    qwen35_vl_397b_a17b_pretrain_512gpu_h100_bf16_mock_config,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


def _apply_qwen35_vl_397b_a17b_64gpu_gb300_execution_config(cfg: ConfigContainer) -> None:
    """Apply the 64-GB300 execution policy for Qwen3.5/Qwen3.6-VL 397B-A17B.

    Mirrors ``qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config`` in
    ``perf_recipes/qwen_vl/gb300`` so real-data runs share the benchmark's
    execution policy. Only the parallel layout, dispatcher, graph and env
    settings are copied; dataset/tokenizer/freeze policy stay real-data.
    """
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None
    # EP32, not EP64. Measured on 64x GB300 with real DataComp, n=50 (steps
    # 101-150): EP32 370.2 TF vs EP64 211.0 TF, a 43% loss for EP64 with every
    # other setting identical. Less expert sharding wins here, matching the
    # 35B 16-GPU sweep (EP4 > EP8 > EP16). EP16 is not reachable: it OOMs at
    # iteration 2 (258.7 of 276.6 GiB, allocator efficient, genuine exhaustion).
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    cfg.model.apply_rope_fusion = False
    cfg.model.bias_activation_fusion = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_permute_fusion_into_hybridep = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_fusion = True
    cfg.model.batch_p2p_sync = False

    # Keep the language-side attn + router + preprocess graphs, matching the
    # 35B-A3B real-data recipe. NOTE the shipped GB300 benchmark declares only
    # [moe_router, moe_preprocess] and then _qwen35_vl_post sets
    # cuda_graph_impl="none", so that scope is DEAD CODE -- do not copy it.
    # Vision stays graph-free: DataComp image shapes vary.
    cfg.model.cuda_graph_impl = "transformer_engine"
    # The attn graph's value at 397B tracks the attention backend. With unfused
    # attention it measured -13.9% (n=50: 342.6 TF without vs 295.1 TF with, loss
    # unaffected, node placement ruled out); with FlashAttention the penalty
    # disappears, fused attention being two kernels and cheap to capture. Kept
    # alongside the router/preprocess graphs to match the 35B-A3B real-data recipe.
    set_cuda_graph_modules(cfg.model, ["attn", "moe_router", "moe_preprocess"])
    cfg.model.vision_cuda_graph_impl = "none"
    cfg.model.vision_cuda_graph_scope = []
    cfg.model.max_vision_cuda_graph_seq_length = None
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
    )

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        # Must equal expert_model_parallel_size or HybridEP splits the NVLink
        # domain wrongly. Unset does NOT auto-detect.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API": 128,
        "NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }


def qwen35_vl_397b_a17b_pretrain_config() -> ConfigContainer:
    """Return the full-pretraining config for Qwen3.5/Qwen3.6-VL 397B-A17B on 64 GB300 GPUs.

    Counterpart of :func:`qwen35_vl_35b_a3b_pretrain_config` for the 397B-A17B
    model: the recipe defaults to the mock VLM dataset, while maintained
    launchers may select a real dataset without rebuilding the model execution
    policy. The execution policy is the shipped 64-GB300 benchmark layout.
    """
    cfg = qwen35_vl_397b_a17b_pretrain_512gpu_h100_bf16_mock_config()

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    # Preserve the architecture vocabulary for checkpoint-backed functional
    # training. Qwen's tokenizer may expose only its core vocabulary while
    # multimodal tokens remain represented by the model vocabulary.
    cfg.tokenizer.use_tokenizer_vocab_size = False

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    _apply_qwen35_vl_397b_a17b_64gpu_gb300_execution_config(cfg)

    # Variable-shape DataComp images retain more activation memory than the
    # fixed-shape benchmark. Recompute only the measured activation-heavy
    # outputs; launchers turn this off when memory allows.
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    # No "moe_act": the CuTe DSL fused grouped MLP rejects moe_act recompute
    # (experts.py:275), so it is unavailable in exactly the fast MXFP8 path.
    cfg.model.recompute_modules = ["core_attn", "gdn_norm_out"]

    # Native cross entropy materializes a full FP32 vocabulary buffer for the
    # 248K-token language head. Keep real-data pretraining memory-bounded with
    # the same TE loss kernel used by the verified full-SFT recipe.
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    return cfg


def _apply_qwen35_vl_35b_a3b_16gpu_gb300_execution_config(cfg: ConfigContainer) -> None:
    """Apply the 16-GB300 execution policy for Qwen3.5/Qwen3.6-VL 35B-A3B.

    Counterpart of :func:`_apply_qwen35_vl_397b_a17b_64gpu_gb300_execution_config`.
    Only the parallel layout, dispatcher, graph and env settings are set here;
    dataset/tokenizer/freeze policy stay real-data.
    """
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None
    # EP4 on 16 GPUs. Measured sweep on real DataComp: EP4 > EP8 > EP16, the
    # same "less expert sharding wins" ordering the 397B 64-GPU sweep shows.
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    cfg.model.apply_rope_fusion = False
    cfg.model.bias_activation_fusion = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_permute_fusion_into_hybridep = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_fusion = True
    cfg.model.batch_p2p_sync = False

    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "moe_router", "moe_preprocess"])

    # Vision encoder CUDA graphs. Measured on 16x GB300 with real DataComp
    # Energon and REAL routing (forced load balancing off), n=50 over steps
    # 101-150: 389.5 -> 465.4 TFLOP/s/GPU, +19.5% (31,063 -> 26,025 ms/iter).
    # A matched-iteration cross-check over all 106 common iterations read
    # +16.6%, so the effect is not a window artifact. The gain is launch
    # overhead the graph replay removes, not vision compute: the encoder is
    # ~10:1 launch-bound at this scale.
    cfg.model.vision_cuda_graph_impl = "transformer_engine"
    cfg.model.vision_cuda_graph_scope = ["attn", "mlp"]
    # PATCH count per microbatch, NOT a token or pixel count, and it scales
    # with micro_batch_size because the vision sequence packs every image in
    # the microbatch. Logged at vision_model.py after patch_embed over 30,720
    # samples at micro_batch_size=4: min 2952 | p50 3272 | p99 3448 | max 3532.
    # 3712 = measured max + 5% headroom, a 1.08x pad factor.
    #
    # DO NOT DERIVE THIS VALUE and DO NOT carry it to another
    # micro_batch_size. Deriving it arithmetically from max_pixels produced
    # 784, which aborts before iteration 1 ("sequence length exceeds
    # max_vision_cuda_graph_seq_length"); an oversized cap instead silently
    # wastes vision attention on padding. Re-log it whenever the batch shape
    # changes: the same instrumentation over 30,720 samples at
    # micro_batch_size=1 gives min 720 | p50 828 | p99 884 | max 980 (cap
    # 1029), so 3712 would pad 4.5x there. Sample size matters -- that max of
    # 980 first appeared only after ~21,000 samples, having sat at 940 for the
    # preceding 20,000.
    cfg.model.max_vision_cuda_graph_seq_length = 3712
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
    )

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        # Must equal expert_model_parallel_size or HybridEP splits the NVLink
        # domain wrongly. Unset does NOT auto-detect.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 4,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API": 128,
        "NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # 0, not the 397B recipe's 20. The LayerNorm SM margin trades SM time
        # against communication headroom, so its sign follows the comm share:
        # measured +3.2% at 35B and -13.4% at 397B.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
    }


def qwen35_vl_35b_a3b_pretrain_config() -> ConfigContainer:
    """Return the full-pretraining config for Qwen3.5/Qwen3.6-VL 35B-A3B on 16 GB300 GPUs.

    GB300 counterpart of the shared
    :func:`megatron.bridge.recipes.qwen_vl.h100.qwen35_vl.qwen35_vl_35b_a3b_pretrain_config`,
    which keeps its H100 execution policy. The recipe defaults to the mock VLM
    dataset, while maintained launchers may select a real dataset without
    rebuilding the model execution policy.
    """
    cfg = qwen35_vl_35b_a3b_pretrain_8gpu_h100_bf16_mock_config()

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.train.global_batch_size = 2048
    # micro_batch_size=4 is load-bearing for max_vision_cuda_graph_seq_length
    # above -- changing it invalidates that cap. See the note there.
    cfg.train.micro_batch_size = 4

    # Preserve the architecture vocabulary for checkpoint-backed functional
    # training. Qwen's tokenizer may expose only its core vocabulary while
    # multimodal tokens remain represented by the model vocabulary.
    cfg.tokenizer.use_tokenizer_vocab_size = False

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    _apply_qwen35_vl_35b_a3b_16gpu_gb300_execution_config(cfg)

    # Variable-shape DataComp images retain more activation memory than the
    # fixed-shape benchmark. Recompute only the measured activation-heavy
    # outputs; launchers turn this off when memory allows.
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    # No "moe_act": the CuTe DSL fused grouped MLP rejects moe_act recompute
    # (experts.py:275), so it is unavailable in exactly the fast MXFP8 path.
    cfg.model.recompute_modules = ["core_attn", "gdn_norm_out"]

    # Native cross entropy materializes a full FP32 vocabulary buffer for the
    # 248K-token language head. Keep real-data pretraining memory-bounded with
    # the same TE loss kernel used by the verified full-SFT recipe.
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    return cfg


__all__ = [
    "qwen35_vl_35b_a3b_pretrain_config",
    "qwen35_vl_397b_a17b_pretrain_config",
]
