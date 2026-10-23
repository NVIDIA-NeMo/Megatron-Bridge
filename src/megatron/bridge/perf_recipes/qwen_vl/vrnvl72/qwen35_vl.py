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
"""VRNVL72 performance recipes for Qwen3.5-VL."""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.qwen_vl.common import ConfigContainer, _select_gdn_kernel_backend
from megatron.bridge.perf_recipes.qwen_vl.gb300.qwen35_vl import (
    _build_qwen35_vl_35b_a3b_gb300_bf16,
    _build_qwen35_vl_397b_a17b_gb300_fp8mx,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config,
    qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config,
    qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8cs_config,
    qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8mx_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config,
)


def qwen35_vl_35b_a3b_pretrain_8gpu_vrnvl72_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× VRNVL72, BF16 (alias of GB300)."""
    cfg = _build_qwen35_vl_35b_a3b_gb300_bf16()
    # The GB300 base selects the cuDNN GDN kernel. On VRNVL72 it was measured only with MXFP8 (against FLA
    # at 35B), so this precision keeps FLA until it is measured too.
    _select_gdn_kernel_backend(cfg, "fla")
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_vrnvl72_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× VRNVL72, FP8-CS (alias of GB300)."""
    cfg = qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config()
    # The GB300 base selects the cuDNN GDN kernel. On VRNVL72 it was measured only with MXFP8 (against FLA
    # at 35B), so this precision keeps FLA until it is measured too.
    _select_gdn_kernel_backend(cfg, "fla")
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_vrnvl72_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× VRNVL72, MXFP8, EP=4, HybridEP.

    Starts from the GB300 MXFP8 recipe (CuTe DSL fused grouped MLP, cuDNN GDN, per-layer CUDA graphs, forced load
    balancing) and applies EP 4, MBS 4 and the kernel settings of the measured VRNVL72 runs, at GBS 1024.
    """
    cfg = qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config()

    # EP 4 and MBS 4 as in the measured VRNVL72 runs: each expert-parallel group is one 4-GPU node
    # (NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN below must equal EP). GBS 1024 gives 32 microbatches
    # per step on 8 GPUs.
    cfg.model.expert_model_parallel_size = 4
    cfg.train.micro_batch_size = 4
    cfg.train.global_batch_size = 1024

    # The mock VLM loader runs the HF image processor on the CPU. One worker per rank supplies fewer
    # samples than this batch size consumes on VRNVL72, which would make the step wait for data.
    cfg.dataset.num_workers = 8

    # Measured on VRNVL72 with MXFP8: keep the cuDNN GDN kernel that the GB300 base selects.
    _select_gdn_kernel_backend(cfg, "transformer_engine")

    # Run the shared expert through Transformer Engine's grouped GEMM with the same 32-wide GLU
    # interleaving as the routed experts.
    cfg.model.use_grouped_gemm_for_shared_expert = True
    cfg.model.moe_shared_expert_glu_interleave_size = 32

    # HybridEP: permute outside the dispatcher (already the default; stated explicitly because fusing
    # the permute into HybridEP measured slower on VRNVL72), 16 SMs for preprocessing as in the measured
    # runs, and overlap the dispatch backward with the expert weight-gradient GEMMs. Megatron-Core does
    # not allow this overlap together with EP A2A overlap or delay_wgrad_compute.
    cfg.model.moe_permute_fusion_into_hybridep = False
    cfg.model.moe_hybridep_num_sms_preprocessing = 16
    cfg.model.overlap_dispatch_backward_with_experts_wgrad = True

    # Vision-encoder CUDA graph. The cap is the number of vision patches per microbatch: a longer
    # vision sequence stops the run, and a shorter one is padded to the cap on every replay, so set it
    # to MBS x images per sample x patches per image whenever one of them changes. The Qwen-VL data
    # path resizes each 256x256 mock image twice: qwen_vl_utils snaps it to 252x252, then the HF
    # processor scales it up to at least QWEN_VL_MIN_PIXELS (448 * 448) pixels in area. Each image
    # therefore reaches the encoder as 448x448 = 28x28 patches of 16x16 pixels, 784 patches, and MBS 4
    # gives 3136. With the graph, training runs the encoder as one dense attention over all images of a
    # microbatch (images attend to each other, and padding is not masked) and without the 2D rotary
    # embedding, which the capture does not record; evaluation runs eagerly, per image and with the
    # rotary embedding. vision_cuda_graph_impl = "none" restores that behavior in training as well.
    cfg.model.vision_cuda_graph_impl = "transformer_engine"
    cfg.model.vision_cuda_graph_scope = ["attn", "mlp"]
    cfg.model.max_vision_cuda_graph_seq_length = 3136

    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system: one rank per GPU of the EP 4 group.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 4,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
        # cuDNN normalization kernels.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        # Gate for the CuTe DSL fused grouped MLP that the GB300 base enables. Transformer Engine
        # checks it when it registers the fusion; without it the op fuser runs the unfused path.
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
    }
    return cfg


def qwen35_vl_122b_a10b_pretrain_32gpu_vrnvl72_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× VRNVL72, BF16 (alias of GB300)."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config()
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen35_vl_122b_a10b_pretrain_32gpu_vrnvl72_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× VRNVL72, FP8-CS (alias of GB300)."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8cs_config()
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen35_vl_122b_a10b_pretrain_32gpu_vrnvl72_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× VRNVL72, FP8-MX (alias of GB300)."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8mx_config()
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen35_vl_397b_a17b_pretrain_64gpu_vrnvl72_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× VRNVL72, BF16 (alias of GB300)."""
    cfg = qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config()
    # The GB300 base selects the cuDNN GDN kernel. On VRNVL72 it was measured only with MXFP8 (against FLA
    # at 35B), so this precision keeps FLA until it is measured too.
    _select_gdn_kernel_backend(cfg, "fla")
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen35_vl_397b_a17b_pretrain_64gpu_vrnvl72_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× VRNVL72, FP8-CS (alias of GB300)."""
    cfg = qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config()
    # The GB300 base selects the cuDNN GDN kernel. On VRNVL72 it was measured only with MXFP8 (against FLA
    # at 35B), so this precision keeps FLA until it is measured too.
    _select_gdn_kernel_backend(cfg, "fla")
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen35_vl_397b_a17b_pretrain_64gpu_vrnvl72_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× VRNVL72, MXFP8, EP=32, HybridEP.

    Starts from the GB300 MXFP8 base (cuDNN GDN, per-layer CUDA graphs, forced load balancing), not the tuned GB300
    recipe, and adds the CuTe DSL fused grouped MLP and the settings of the measured VRNVL72 run.
    """
    cfg = _build_qwen35_vl_397b_a17b_gb300_fp8mx()

    # MBS 1, as in the GB300 base; the vision cap below is set for it.
    cfg.train.micro_batch_size = 1

    # Keep the cuDNN GDN kernel that the GB300 base selects: it measured faster than FLA on VRNVL72 at
    # 35B, and the measured 397B run used it.
    _select_gdn_kernel_backend(cfg, "transformer_engine")

    # Route the experts through Transformer Engine's CuTe DSL fused grouped MLP (FC1 + SwiGLU + FC2 for
    # MXFP8). The GB300 397B base does not enable the op fuser (it is off by default), so it is enabled
    # here; with it, Megatron-Core also runs the experts on grouped tensors. The op fuser, the 32-wide
    # GLU interleaving and NVTE_CUTEDSL_FUSED_GROUPED_MLP below are required together.
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32

    # Recompute the GDN output norm, as the measured run did in effect. That run also listed core_attn, but
    # all of its attention layers ran inside CUDA graphs, where Megatron-Core does not recompute. The vision
    # encoder inherits selective recompute of its attention (vision_recompute_granularity="inherit"); inside
    # the vision CUDA graph below that has no effect either, but it applies once the vision graph is off.
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["gdn_norm_out"]

    # HybridEP: permute outside the dispatcher. This is already the default; it is stated explicitly
    # because fusing the permute into HybridEP measured slower on VRNVL72.
    cfg.model.moe_permute_fusion_into_hybridep = False

    # Vision-encoder CUDA graph with the cap for MBS 1: each 256x256 mock image reaches the encoder as
    # 784 patches (see the 35B recipe above). Set the cap to MBS x images per sample x patches per image
    # whenever one of them changes. As in the 35B recipe, the graphed encoder runs without the 2D rotary
    # embedding, and images that share a microbatch attend to each other.
    cfg.model.vision_cuda_graph_impl = "transformer_engine"
    cfg.model.vision_cuda_graph_scope = ["attn", "mlp"]
    cfg.model.max_vision_cuda_graph_seq_length = 784

    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # HybridEP dispatch and preprocessing chunks of the measured run (HybridEP's default is 64).
        "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API": 128,
        "NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API": 128,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
        # Gate for the CuTe DSL fused grouped MLP enabled above. Transformer Engine checks it when it
        # registers the fusion; without it the op fuser runs the unfused path.
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
    }
    return cfg
