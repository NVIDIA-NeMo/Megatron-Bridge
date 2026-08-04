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
"""H100 performance recipes for NemotronH and Nemotron 3."""

import torch

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.nemotronh.common import (
    ConfigContainer,
    _apply_nemotron_3_nano_perf_defaults,
    _benchmark_common,
    _nemotron_3_ultra_perf_fsdp_config,
    _perf_precision,
    nemotron_3_nano_pretrain_config,
    nemotronh_56b_pretrain_config,
)
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


# Placeholder until the public Nemotron 3.5 Nano repository is released.
_NEMOTRON_3_5_NANO_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16"


def nemotronh_56b_pretrain_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 64× H100, FP8 current-scaling."""
    cfg = nemotronh_56b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 192
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mamba"]

    _benchmark_common(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def _nemotron_3_ultra_pretrain_h100_bf16_fsdp_config(
    *,
    num_gpus: int,
    tensor_model_parallel_size: int,
    global_batch_size: int,
    recompute_num_layers: int,
) -> ConfigContainer:
    """Build a PP1 Nemotron 3 Ultra H100 BF16 FSDP candidate."""
    cfg = _nemotron_3_ultra_perf_fsdp_config(
        num_gpus=num_gpus,
        compute_dtype="bf16",
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        expert_model_parallel_size=64,
        global_batch_size=global_batch_size,
        # EDP is one at EP64 on these topologies, so partial optimizer instances
        # are invalid and the full 64-rank dense DP group remains one shard.
        optimizer_shard_group_size=64,
        fine_grained_activation_offloading=False,
        enable_fine_grained_param_gather=True,
    )
    # Use the stateless all-to-all token dispatcher so activation replay does
    # not include DeepEP state. With the op fuser disabled below, moe_act uses
    # MCore's standalone CheckpointWithoutOutput path instead of the SM100-only
    # fused grouped-MLP implementation. This also keeps the latent projection
    # outside whole-MoE replay so its fused FP32 wgrad is visible to FSDP.
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    # The selective Mamba SSD chunk does not cover the full-sequence output
    # projection, whose 128-MiB GEMM output is the remaining H100 forward peak.
    # A two-layer checkpoint moved the failure from the full-sequence Mamba
    # projection to retained routed- and shared-expert activations in later
    # layers. Both 16 and 32 checkpointed layers still exhausted the H100
    # during routed-expert GEMM, permutation, and all-to-all outputs. Checkpoint
    # enough leading Mamba/expert pairs to fit the selected TP layout: the
    # all-108-layer probe proved ample memory headroom but could not meet the
    # four-hour verification window. MCore deliberately skips block-method MTP
    # recompute, leaving its latent projection on the normal fused-gradient
    # path.
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "block"
    cfg.model.recompute_num_layers = recompute_num_layers
    cfg.model.recompute_modules = None

    # Thirty-two chunks reduced the first-forward routed-expert failures to
    # 20-MiB FC2 grouped-linear outputs, but a 256-KiB NCCL buffer did not leave
    # enough free memory for them. Split the supported training MLP path into
    # sixty-four sequence chunks to halve that remaining chunk-local output.
    # Unlike whole-MoE checkpointing, chunking leaves the MTP latent projection
    # on its normal backward path so FSDP observes its fused FP32 wgrad.
    cfg.model.mlp_chunks_for_training = 64

    # Earlier MLP chunking exposed a later first-forward allocation failure in
    # the Mamba output projection.
    # Double the supported SSD chunk size to reduce the number of Mamba
    # chunk-boundary states while keeping its intra-chunk workspace bounded.
    cfg.model.mamba_chunk_size = 256

    # The op-fuser ScaledSReLU path exceeds H100 activation memory before the
    # first optimizer step. Keep TE grouped linears and the fused weighted
    # squared-ReLU implementation, but use their supported non-op-fuser path.
    cfg.model.use_transformer_engine_op_fuser = False

    # The 256-GPU BF16 verification exposed a NaN global gradient norm when
    # gradients were materialized in BF16 before being copied into the FSDP
    # buffer. Megatron-FSDP recommends FP32 main gradients for accuracy at
    # scale, and its TE fusion writes wgrads directly into that FP32 buffer.
    # This topology has enough memory headroom for the wider gradient buffer.
    cfg.ddp.megatron_fsdp_grad_comm_dtype = torch.float32
    cfg.ddp.megatron_fsdp_main_grads_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.model.gradient_accumulation_fusion = True

    # Disable forward parameter prefetch. Even the smallest positive prefetch
    # window admits one complete following FSDP bucket, which leaves
    # insufficient H100 memory for the grouped-MoE activation while using FP32
    # fused main gradients. A one-element communication unit maps to a zero
    # all-gather prefetch window while preserving the existing bucket layout.
    cfg.ddp.suggested_communication_unit_size = 1

    # Performance defaults disable these checks, but an invalid gradient must
    # fail the verification run before the optimizer can corrupt the model.
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.check_for_large_grads = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True
    return cfg


def nemotron_3_ultra_pretrain_128gpu_h100_bf16_fsdp_tp2_config() -> ConfigContainer:
    """Nemotron 3 Ultra pretrain: 128× H100, TP2/PP1 BF16 FSDP."""
    cfg = _nemotron_3_ultra_pretrain_h100_bf16_fsdp_config(
        num_gpus=128,
        tensor_model_parallel_size=2,
        global_batch_size=256,
        # Block 64 and 72 both left the H100 effectively full during a
        # first-forward expert output. Checkpoint every main HybridStack layer;
        # MTP remains on its normal backward path under the block method.
        recompute_num_layers=108,
    )
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # Megatron-FSDP requires more than one CUDA device connection. Eight
        # bounds Hopper driver/stream resources while retaining stream independence.
        "CUDA_DEVICE_MAX_CONNECTIONS": 8,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # Bound NCCL pair buffers to 256 KiB for this capacity-constrained model.
        "NCCL_BUFFSIZE": 262144,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def nemotron_3_ultra_pretrain_256gpu_h100_bf16_fsdp_config() -> ConfigContainer:
    """Nemotron 3 Ultra pretrain: 256× H100, TP4/PP1 BF16 FSDP."""
    cfg = _nemotron_3_ultra_pretrain_h100_bf16_fsdp_config(
        num_gpus=256,
        tensor_model_parallel_size=4,
        global_batch_size=512,
        recompute_num_layers=64,
    )
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # Megatron-FSDP requires more than one CUDA device connection. Eight
        # bounds Hopper driver/stream resources while retaining stream independence.
        "CUDA_DEVICE_MAX_CONNECTIONS": 8,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # Bound NCCL pair buffers to 256 KiB for this capacity-constrained model.
        "NCCL_BUFFSIZE": 262144,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def nemotron_3_nano_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 16× H100, BF16, recompute MoE+layernorm."""
    cfg = nemotron_3_nano_pretrain_config()
    _apply_nemotron_3_nano_perf_defaults(cfg)
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.recompute_granularity = "selective"

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba"])

    cfg.model.recompute_modules = ["moe", "layernorm"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 16
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
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 16× H100, FP8 current-scaling, recompute."""
    cfg = nemotron_3_nano_pretrain_config()
    _apply_nemotron_3_nano_perf_defaults(cfg)
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.model.recompute_granularity = "selective"

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["mamba"])

    cfg.model.recompute_modules = ["moe", "layernorm", "core_attn", "moe_act"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 16
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
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def nemotron_3_5_nano_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Nemotron 3.5 Nano pretrain: 16× H100, BF16."""
    cfg = nemotron_3_nano_pretrain_16gpu_h100_bf16_config()
    # Keep the benchmark workload aligned with the GB200 BF16 recipe. The
    # hardware recipes may tune execution-only knobs such as microbatch size,
    # recompute, and CUDA graph coverage independently.
    cfg.train.global_batch_size = 512
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.hf_model_id = _NEMOTRON_3_5_NANO_MODEL_ID
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_5_NANO_MODEL_ID
    cfg.env_vars = {
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
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def nemotron_3_5_nano_pretrain_16gpu_h100_fp8cs_config() -> ConfigContainer:
    """Nemotron 3.5 Nano pretrain: 16× H100, FP8 current-scaling."""
    cfg = nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config()
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.hf_model_id = _NEMOTRON_3_5_NANO_MODEL_ID
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_5_NANO_MODEL_ID
    cfg.env_vars = {
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
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg
