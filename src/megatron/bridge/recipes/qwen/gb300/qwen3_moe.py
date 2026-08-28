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

"""GB300 recipes for Qwen3 MoE models."""

from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_235b_a22b_pretrain_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed


def qwen3_235b_a22b_256gpu_gb300_fp8mx_pretrain_config() -> ConfigContainer:
    """Return the natural-routing Qwen3-235B MXFP8 pretraining recipe for 256 GB300 GPUs."""
    cfg = qwen3_235b_a22b_pretrain_config()

    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.bias_activation_fusion = True
    cfg.model.apply_rope_fusion = True
    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["moe_act"]
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 3
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8192
    cfg.train.micro_batch_size = 2
    cfg.scheduler.lr_warmup_iters = 40
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.save_interval = 100

    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_hybridep_num_sms = 32
    # Keep dynamic expert work outside the graph so convergence safety checks
    # remain available with natural routing.
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
    cfg.rng.te_rng_tracker = True
    cfg.model.use_te_rng_tracker = True
    cfg.model.offload_modules = []
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.use_transformer_engine_op_fuser = False
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    cfg.mixed_precision.fp8_dot_product_attention = True
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )

    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.check_for_large_grads = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = None

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


__all__ = ["qwen3_235b_a22b_256gpu_gb300_fp8mx_pretrain_config"]
