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

"""glm_moe_dsa benchmark overrides applied to library recipes.

The literal names preserve existing CLI selectors. Common and model overrides
retain their ordering because graph, RNG and precision policies are dependent.
"""

from megatron.bridge import AutoBridge
from megatron.bridge.models.glm_moe_dsa.glm5_provider import split_glm_pattern
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.perf_benchmark_overrides._common import _benchmark_common, _perf_precision
from megatron.bridge.training.perf_benchmark_overrides.environment import COMMON_PERF_ENV_VARS


BENCHMARK_ENVIRONMENTS = {
    "glm51_sft_192gpu_gb200_bf16": {
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
    "glm51_sft_416gpu_h100_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
}
LIBRARY_RECIPES = {
    "glm51_sft_192gpu_gb200_bf16_config": _sft_common,
    "glm52_sft_192gpu_gb200_bf16_config": _sft_common,
    "glm51_sft_416gpu_h100_bf16_config": _sft_common,
    "glm52_sft_416gpu_h100_bf16_config": _sft_common,
}
ALLOCATION_AWARE = ()


def apply_overrides(cfg: ConfigContainer, name: str) -> ConfigContainer:
    """Apply model-specific benchmark overrides to an existing library config."""
    match name:
        case "glm51_sft_192gpu_gb200_bf16_config":
            cfg.model = AutoBridge.from_hf_pretrained("zai-org/GLM-5.1").get_model_config()
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
            cfg.model.num_layers_in_first_pipeline_stage = None
            cfg.model.num_layers_in_last_pipeline_stage = None
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["glm51_sft_192gpu_gb200_bf16"].copy()
            block_counts = [14, 12, 12, 12, 12, 16]
            cfg.model.hybrid_layer_pattern = split_glm_pattern(cfg.model.hybrid_layer_pattern, block_counts)
            return cfg
        case "glm52_sft_192gpu_gb200_bf16_config":
            cfg.model = AutoBridge.from_hf_pretrained("zai-org/GLM-5.2").get_model_config()
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
            cfg.model.num_layers_in_first_pipeline_stage = None
            cfg.model.num_layers_in_last_pipeline_stage = None
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["glm51_sft_192gpu_gb200_bf16"].copy()
            block_counts = [14, 12, 12, 12, 12, 16]
            cfg.model.hybrid_layer_pattern = split_glm_pattern(cfg.model.hybrid_layer_pattern, block_counts)
            return cfg
        case "glm51_sft_416gpu_h100_bf16_config":
            cfg.model = AutoBridge.from_hf_pretrained("zai-org/GLM-5.1").get_model_config()
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
            cfg.model.pipeline_model_parallel_size = 13
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.context_parallel_size = 32
            cfg.model.expert_model_parallel_size = 32
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.model.pipeline_model_parallel_layout = None
            cfg.model.account_for_embedding_in_pipeline_split = False
            cfg.model.account_for_loss_in_pipeline_split = False
            cfg.model.num_layers_in_first_pipeline_stage = None
            cfg.model.num_layers_in_last_pipeline_stage = None
            cfg.train.global_batch_size = 520
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
            cfg.model.moe_flex_dispatcher_backend = "deepep"
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["glm51_sft_416gpu_h100_bf16"].copy()
            block_counts = [6, 4] + [8] * 6 + [4] * 5
            cfg.model.hybrid_layer_pattern = split_glm_pattern(cfg.model.hybrid_layer_pattern, block_counts)
            return cfg
        case "glm52_sft_416gpu_h100_bf16_config":
            cfg.model = AutoBridge.from_hf_pretrained("zai-org/GLM-5.2").get_model_config()
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
            cfg.model.pipeline_model_parallel_size = 13
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.context_parallel_size = 32
            cfg.model.expert_model_parallel_size = 32
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.model.pipeline_model_parallel_layout = None
            cfg.model.account_for_embedding_in_pipeline_split = False
            cfg.model.account_for_loss_in_pipeline_split = False
            cfg.model.num_layers_in_first_pipeline_stage = None
            cfg.model.num_layers_in_last_pipeline_stage = None
            cfg.train.global_batch_size = 520
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
            cfg.model.moe_flex_dispatcher_backend = "deepep"
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
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["glm51_sft_416gpu_h100_bf16"].copy()
            block_counts = [6, 4] + [8] * 6 + [4] * 5
            cfg.model.hybrid_layer_pattern = split_glm_pattern(cfg.model.hybrid_layer_pattern, block_counts)
            return cfg
        case _:
            raise ValueError(f"Unknown benchmark overrides: {name}")
