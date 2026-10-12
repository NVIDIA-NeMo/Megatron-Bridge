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
from megatron.bridge.training.mixed_precision import bf16_mixed, bf16_with_mxfp8_mixed
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
    cfg.model.cuda_graph_impl = "transformer_engine"
    # The attn graph's value at 397B tracks the attention backend. With unfused
    # attention it measured -13.9% (n=50: 342.6 TF without vs 295.1 TF with, loss
    # unaffected, node placement ruled out); with FlashAttention the penalty
    # disappears, fused attention being two kernels and cheap to capture. Kept
    # alongside the router/preprocess graphs to match the 35B-A3B real-data recipe.
    set_cuda_graph_modules(cfg.model, ["attn", "moe_router", "moe_preprocess"])
    # Vision encoder CUDA graphs. Variable DataComp image shapes are handled by
    # padding each vision sequence up to max_vision_cuda_graph_seq_length, so the
    # cap must be MEASURED for this recipe's batch shape -- see the note in
    # _apply_qwen35_vl_35b_a3b_16gpu_gb300_execution_config on why it is a patch
    # count that must never be derived or carried across micro_batch_size.
    # This recipe runs micro_batch_size=1: over 30,720 real DataComp samples the
    # vision sequence measured min 720 | p50 828 | p99 884 | max 980, so
    # 1029 = max + 5%. Earlier 397B launch configs reused the micro_batch_size=4
    # cap of 3712 here, padding every vision attention call ~4.5x.
    cfg.model.vision_cuda_graph_impl = "transformer_engine"
    cfg.model.vision_cuda_graph_scope = ["attn", "mlp"]
    cfg.model.max_vision_cuda_graph_seq_length = 1029
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
    # micro_batch_size=1 is load-bearing for max_vision_cuda_graph_seq_length in
    # the execution config -- changing it invalidates that cap.
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


def qwen35_vl_35b_a3b_pretrain_16gpu_gb300_bf16_config() -> ConfigContainer:
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


def qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_library_config() -> ConfigContainer:
    """Return the MXFP8 full-pretraining config for Qwen3.5/Qwen3.6-VL 35B-A3B on 8 GB300 GPUs.

    Starts from :func:`qwen35_vl_35b_a3b_pretrain_16gpu_gb300_bf16_config` and keeps its parallel layout (EP 4,
    TP/PP/CP 1), CUDA graphs, recompute, batch shape (micro batch 4, global batch 2048) and real routing. On top it
    sets MXFP8, Transformer Engine's CuTe DSL fused grouped MLP, cuDNN GDN, and the HybridEP, dispatcher and
    normalization settings of the configuration it was tuned on. With TP, PP and CP at 1, data parallelism is 8 on
    8 GPUs (expert data parallelism 2), so each rank runs 64 micro batches per step.

    Like the 16-GPU recipe, it defaults to the mock VLM dataset. The measured runs used an Energon dataset
    (``--dataset energon dataset.path=...``) with ``dataset.task_encoder.max_pixels=200704``, and the vision CUDA graph
    cap inherited from the 16-GPU recipe (3712 patches per micro batch) was measured with that image budget at micro
    batch 4. Keep both, or re-measure the cap: a larger image budget can exceed it and stop training with an error.

    cuDNN GDN is selected only on a Megatron-Core that declares ``gdn_kernel_backend`` (NVIDIA/Megatron-LM#7583);
    otherwise the GatedDeltaNet layers keep running on FLA.

    Transformer Engine reads the fused-MLP gate ``NVTE_CUTEDSL_FUSED_GROUPED_MLP`` once, when it is imported.
    ``scripts/training/run_recipe.py``, which ``train.sh`` runs, sets it in time by re-executing itself; with any
    other entry point, export it before Python starts, or Transformer Engine silently runs the unfused grouped GEMM.

    The name carries a ``_library`` suffix because ``qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config`` is
    already a perf recipe (a mock-data benchmark with EP 8 and global batch 512), and bare recipe lookup keeps
    resolving that name to it.

    Measured on GB300 with real DataComp data and real routing (steps 101-150, one run per arm, submitted together),
    a launch configuration close to this recipe reached 11.3% more TFLOP/s per GPU on 8 GPUs than on 16 at the same
    global batch size. The two runs differ in data-parallel size and in micro batches per rank (64 vs 32), and one
    run per arm does not separate these effects, so no cause is claimed.
    """
    cfg = qwen35_vl_35b_a3b_pretrain_16gpu_gb300_bf16_config()

    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    # CuTe DSL fused grouped MLP (FC1 + SwiGLU + FC2 for MXFP8 on GPUs with compute capability 10.x). The op fuser,
    # the 32-wide GLU interleave and the NVTE_CUTEDSL_FUSED_GROUPED_MLP env gate (cfg.env_vars below) are all
    # required. Transformer Engine reads the gate when it is imported and registers the fusion only if it is set
    # (scripts/training/run_recipe.py re-executes itself so the gate is set before that import; other entry points
    # must export it); without the gate or the interleave it silently runs the unfused grouped GEMM.
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32

    cfg.model.moe_flex_dispatcher_num_sms = 32
    # Off: HybridEP's fused permute requires identical dispatch, combine and preprocessing chunk sizes, and this
    # recipe uses 64/128/64 (cfg.env_vars below). The tuned configuration ran this way too.
    cfg.model.moe_permute_fusion_into_hybridep = False

    # The tuned configuration started from qwen35_vl_35b_a3b_pretrain_config and kept these values. The 16-GPU
    # recipe does not carry them over (it leaves the two model fields at Megatron-Core defaults, sets the dispatch
    # and preprocessing chunk sizes to 128 and omits the cuDNN normalization flags), and was never measured with
    # MXFP8.
    cfg.model.moe_hybridep_num_sms_preprocessing = 16
    cfg.model.overlap_dispatch_backward_with_experts_wgrad = True

    # cuDNN GDN (FROST engine: cuDNN frontend >= 1.29 and nvidia-cutlass-dsl >= 4.7): measured +12.0% to +19.4% end
    # to end at 35B-A3B on 16 GB300 GPUs (see _select_gdn_kernel_backend in perf_recipes/qwen_vl/common.py). Guarded
    # like _enable_gdn_conv_fusion: assigning a field Megatron-Core lacks would not raise, and the recipe would look
    # enabled while FLA keeps running.
    if hasattr(type(cfg.model), "gdn_kernel_backend") or hasattr(cfg.model, "gdn_kernel_backend"):
        cfg.model.gdn_kernel_backend = "transformer_engine"

    # The complete environment, declared here rather than inherited from the 16-GPU recipe. It differs from that
    # recipe in the fused-MLP gate, the dispatch and preprocessing chunk sizes (64 instead of 128) and the cuDNN
    # normalization flags.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # Must equal expert_model_parallel_size or HybridEP splits the NVLink
        # domain wrongly. Unset does NOT auto-detect.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 4,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API": 64,
        "NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API": 64,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # 0, as in the 16-GPU recipe: the LayerNorm SM margin follows the communication share (see there).
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
    }
    return cfg


__all__ = [
    "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_library_config",
    "qwen35_vl_35b_a3b_pretrain_16gpu_gb300_bf16_config",
    "qwen35_vl_397b_a17b_pretrain_config",
]
