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
"""GB200 performance recipes for GLM-5.1 and GLM-5.2 pretraining and SFT."""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.perf_recipes._common import _benchmark_common, _perf_precision
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.recipes.common import _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


_GLM52_MODEL_ID = "zai-org/GLM-5.2"
_GLM52_MODEL_REVISION = "4d67f66cc64d3219133b767c253b2ad1425c6c88"  # pragma: allowlist secret
_GLM52_PP6_PERF_LAYOUT = "Et*10|t*12|t*12|t*12|t*16|t*16mL"


def glm52_pretrain_192gpu_gb200_fp8mx_config() -> ConfigContainer:
    """GLM-5.2 perf demo: 192× GB200, MXFP8, 4K mock data, PP6/EP32."""
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(
        _GLM52_MODEL_ID,
        revision=_GLM52_MODEL_REVISION,
    ).to_megatron_provider(load_weights=False)
    cfg.mixed_precision = _perf_precision("fp8_mx")

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.seq_length = 4096
    cfg.dataset.num_dataset_builder_threads = 1

    # Preserve the complete GLM-5.2 model while matching the measured 192-GPU
    # topology. The uneven stages keep DSA index-sharing groups local and move
    # more decoder layers away from the embedding-heavy first stage.
    cfg.model.seq_length = 4096
    cfg.model.qk_pos_emb_head_dim = 64
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = _GLM52_PP6_PERF_LAYOUT
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False

    cfg.train.global_batch_size = 3072
    cfg.train.micro_batch_size = 1

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_permute_fusion_into_hybridep = False
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.persist_layer_norm = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.bias_activation_fusion = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.dsa_indexer_topk = 2048
    # This is an upper-bound performance demo, not a convergence recipe. Force
    # balancing and a disabled auxiliary indexer loss isolate system throughput.
    cfg.model.dsa_indexer_loss_coeff = 0.0
    cfg.model.dsa_indexer_use_sparse_loss = False
    cfg.model.mtp_num_layers = 1

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    cfg.model.cuda_graph_warmup_steps = 0
    cfg.model.high_priority_a2a_comm_stream = False
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_p2p_comm=False,
        batch_p2p_comm=True,
        overlap_grad_reduce=False,
        overlap_param_gather=True,
        overlap_param_gather_with_optimizer_step=False,
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
    )
    cfg.dist.distributed_timeout_minutes = 45
    cfg.dist.enable_megatron_core_experimental = True

    _benchmark_common(cfg, cross_entropy_impl="native")
    cfg.optimizer.lr = 3e-5
    cfg.optimizer.min_lr = 3e-5
    cfg.scheduler.lr_warmup_iters = 0
    cfg.model.apply_rope_fusion = False
    cfg.model.fp8_output_proj = True
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NCCL_GRAPH_REGISTER": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "USE_MNNVL": 1,
    }
    return cfg


def glm51_sft_192gpu_gb200_bf16_config() -> ConfigContainer:
    """GLM-5.1 SFT: 192× GB200, BF16, 128K packed THD, CP=32, cuDNN DSA."""
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained("zai-org/GLM-5.1").to_megatron_provider(load_weights=False)
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.seq_length = 131072
    cfg.dataset.num_workers = 1
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
    cfg.dataset.offline_packing_specs.packed_sequence_size = 131072
    cfg.dataset.offline_packing_specs.pad_seq_to_mult = 64
    cfg.dataset.offline_packing_specs.tokenizer_model_name = "glm5"

    cfg.model.seq_length = 131072
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 32
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = 14
    cfg.model.num_layers_in_last_pipeline_stage = 16

    cfg.train.global_batch_size = 56
    cfg.train.micro_batch_size = 1

    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.optimizer.use_distributed_optimizer = True

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cp_comm_type = "allgather"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.deallocate_pipeline_outputs = True
    cfg.model.persist_layer_norm = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.bias_activation_fusion = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.mtp_num_layers = 1

    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    _benchmark_common(cfg, cross_entropy_impl="native")
    cfg.model.apply_rope_fusion = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.mixed_precision.grad_reduce_in_fp32 = True
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


def glm52_sft_192gpu_gb200_bf16_config() -> ConfigContainer:
    """GLM-5.2 SFT: 192× GB200, BF16, 128K packed THD, CP=32, cuDNN DSA."""
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained("zai-org/GLM-5.2").to_megatron_provider(load_weights=False)
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    cfg.dataset.seq_length = 131072
    cfg.dataset.num_workers = 1
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
    cfg.dataset.offline_packing_specs.packed_sequence_size = 131072
    cfg.dataset.offline_packing_specs.pad_seq_to_mult = 64
    cfg.dataset.offline_packing_specs.tokenizer_model_name = "glm5"

    cfg.model.seq_length = 131072
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 32
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = 14
    cfg.model.num_layers_in_last_pipeline_stage = 16

    cfg.train.global_batch_size = 56
    cfg.train.micro_batch_size = 1

    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.optimizer.use_distributed_optimizer = True

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cp_comm_type = "allgather"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.deallocate_pipeline_outputs = True
    cfg.model.persist_layer_norm = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.bias_activation_fusion = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.mtp_num_layers = 1

    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    _benchmark_common(cfg, cross_entropy_impl="native")
    cfg.model.apply_rope_fusion = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.mixed_precision.grad_reduce_in_fp32 = True
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
