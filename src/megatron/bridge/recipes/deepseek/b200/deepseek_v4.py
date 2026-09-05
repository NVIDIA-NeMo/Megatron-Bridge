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
"""B200 NVL8 recipes for DeepSeek V4 Flash."""

from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_library_config,
)
from megatron.bridge.recipes.deepseek.h100.deepseek_v4 import (
    deepseek_v4_flash_no_mtp_sft_32gpu_h100_bf16_config,
)
from megatron.bridge.recipes.utils.dataset_utils import default_squad_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


_DSV4_FLASH_PP4_VP4_LAYOUT = "Et*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*2|t*2|t*2|t*2|t*2mL"
_DSV4_FLASH_PP5_128K_LAYOUT = "Et*8|t*9|t*9|t*9|t*8L"


def deepseek_v4_flash_pretrain_64gpu_b200_fp8mx_library_config() -> ConfigContainer:
    """Return real-training DeepSeek V4 Flash for 64 B200 GPUs.

    Requires eight physical NVL8 systems with eight contiguous, node-major
    ranks per system. Each EP16 group spans two NVL8 domains, with eight ranks
    per domain. The runtime must expose ``deep_ep.HybridEPBuffer``. Natural,
    unlimited-capacity routing is preserved without paged stash, activation
    offload, or CUDA graphs.
    """
    cfg = deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_library_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.pipeline_model_parallel_layout = _DSV4_FLASH_PP4_VP4_LAYOUT

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_deepep_num_sms = None
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_hybridep_num_sms_preprocessing = 108
    cfg.model.moe_shared_expert_overlap = False

    cfg.model.recompute_modules = ["mhc", "mla_up_proj"]
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = []
    cfg.model.fine_grained_offloading_max_inflight_offloads = None
    cfg.model.moe_pad_experts_for_cuda_graph_inference = False
    cfg.model.cuda_graph_impl = "none"
    set_cuda_graph_modules(cfg.model, [])
    cfg.model.use_te_rng_tracker = False
    cfg.rng.te_rng_tracker = False

    cfg.env_vars = {
        **{key: value for key, value in cfg.env_vars.items() if key != "NVTE_CPU_OFFLOAD_V1"},
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


def deepseek_v4_flash_sft_160gpu_b200_bf16_128k_config() -> ConfigContainer:
    """Return packed 128K BF16 SFT for 160 B200 GPUs.

    The validated topology uses 20 physical NVL8 systems with node-major rank
    assignment. Each EP16 group spans two NVL8 domains. The caller must supply
    the pretrained checkpoint and may replace the default packed SFT dataset.
    The HybridEP runtime and topology-correct NIC mapping are site-provided.
    """
    cfg = deepseek_v4_flash_no_mtp_sft_32gpu_h100_bf16_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 5
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 16
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.pipeline_model_parallel_layout = _DSV4_FLASH_PP5_128K_LAYOUT

    cfg.model.seq_length = 131072
    cfg.model.cp_partition_mode = "contiguous"
    cfg.model.sequence_packing_scheduler = "dp_balanced"
    cfg.model.max_seqlen_per_dp_cp_rank = 8192
    cfg.model.thd_max_packed_sequences = 6
    cfg.model.thd_tail_padding_policy = "append_dummy_seq"
    cfg.model.pad_packed_seq_alignment = "max"
    cfg.model.variable_seq_lengths = True
    cfg.model.calculate_per_token_loss = True

    cfg.model.apply_dsa_kernel_fusion = True
    cfg.model.dsa_indexer_loss_coeff = 0.0
    cfg.model.dsa_indexer_use_sparse_loss = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 108
    cfg.model.moe_router_bias_update_rate = 0.001
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_expert_capacity_factor = None
    cfg.model.moe_expert_rank_capacity_factor = None
    cfg.model.moe_token_dropping = False
    cfg.model.moe_paged_stash = False
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.mlp_chunks_for_training = 1
    cfg.model.use_transformer_engine_op_fuser = True

    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["mhc", "mla_up_proj"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.mhc_recompute_attn_cuda_graph_split = False
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["fused_group_mlp"]
    cfg.model.activation_offload_fraction = 1.0
    cfg.model.fine_grained_offloading_max_inflight_offloads = 2
    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn"])
    cfg.model.cuda_graph_warmup_steps = 1
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.dataset = default_squad_config(seq_length=131072, enable_offline_packing=True, pad_seq_to_mult=64)
    assert cfg.dataset.offline_packing_specs is not None
    cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
    cfg.dataset.offline_packing_specs.num_tokenizer_workers = 1
    cfg.dataset.num_workers = 2
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False

    cfg.train.train_iters = 50
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 128
    cfg.train.empty_unused_memory_level = 2
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True

    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.checkpoint.async_save = False
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.load_optim = False
    cfg.checkpoint.load_rng = False
    cfg.ddp.average_in_collective = False
    cfg.dist.distributed_timeout_minutes = 120
    cfg.env_vars = {
        **cfg.env_vars,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "HYBRID_EP_DOCA_WRITE_FLAGS": 1,
        "NVLINK_DOMAIN_SIZE": 8,
        "NVTE_CPU_OFFLOAD_V1": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "USE_MNNVL": 0,
    }
    return cfg
