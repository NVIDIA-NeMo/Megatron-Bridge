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
"""GB300 performance recipe for DeepSeek V4 Pro."""

import torch

from megatron.bridge.perf_recipes._common import _benchmark_common
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.recipes.deepseek.gb300.deepseek_v4 import (
    deepseek_v4_pro_pretrain_32gpu_gb300_fp8mx_config,
)
from megatron.bridge.recipes.utils.optimizer_utils import distributed_muon_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer, MockVarlenDatasetConfig


def deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """DeepSeek V4 Pro proxy: 64K THD/CP training on 64 GB300 GPUs with MXFP8."""
    cfg = deepseek_v4_pro_pretrain_32gpu_gb300_fp8mx_config()

    cfg.model.num_layers = 15
    cfg.model.moe_layer_freq = [1] * cfg.model.num_layers
    cfg.model.num_moe_experts = 384
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.context_parallel_size = 16
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    cfg.model.seq_length = 65_536
    cfg.model.max_position_embeddings = 65_536
    cfg.model.original_max_position_embeddings = 4096
    cfg.model.vocab_size = 129_280
    cfg.model.actual_vocab_size = 129_280
    cfg.model.make_vocab_size_divisible_by = 3232
    cfg.model.q_lora_rank = 1536
    cfg.model.o_groups = 16
    cfg.model.o_lora_rank = 1024
    cfg.model.csa_compress_ratios = [128, 128, 4, *([128, 4] * 6), 0]
    cfg.model.csa_compress_rotary_base = 40_000.0
    cfg.model.csa_window_size = 128
    cfg.model.rotary_scaling_factor = 4.0
    cfg.model.dsa_indexer_n_heads = 64
    cfg.model.dsa_indexer_head_dim = 128
    cfg.model.moe_n_hash_layers = 0
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1
    cfg.tokenizer.vocab_size = 129_280
    cfg.tokenizer.make_vocab_size_divisible_by = 3232
    cfg.tokenizer.tensor_model_parallel_size = 1

    cfg.dataset = MockVarlenDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        seq_length=65_536,
        num_dataset_builder_threads=1,
        split="99,1,0",
        data_sharding=True,
        dataloader_type="single",
        num_workers=0,
        skip_getting_attention_mask_from_dataset=True,
        data_parallel_size=4,
        context_parallel_size=16,
        sequence_parallel_size=0,
        varlen_mock_dataset_config_json=(
            '{"mode":"distribution","type":"lognormal","format":"thd",'
            '"min_seq_len":65534,"max_seq_len":65534,"mean_seq_len":65534,'
            '"lognormal_sigma":1.1}'
        ),
    )
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1

    cfg.model.sequence_packing_scheduler = "dp_balanced"
    cfg.model.max_seqlen_per_dp_cp_rank = 4096
    cfg.model.pad_packed_seq_alignment = "max"
    cfg.model.thd_max_packed_sequences = 8
    cfg.model.cp_partition_mode = "contiguous"
    cfg.model.calculate_per_token_loss = True

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.moe_router_load_balancing_type = "seq_aux_loss"
    cfg.model.moe_aux_loss_coeff = 1.0e-4
    cfg.model.moe_router_topk_scaling_factor = 2.5
    cfg.model.moe_router_score_function = "sqrtsoftplus"
    cfg.model.moe_router_enable_expert_bias = True
    cfg.model.moe_router_bias_update_rate = 1.0e-3
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_router_fusion = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = False
    cfg.model.moe_router_padding_for_fp8 = False
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.activation_func_clamp_value = None

    cfg.model.apply_dsa_kernel_fusion = True
    cfg.model.dsa_indexer_topk = 1024
    cfg.model.dsa_indexer_loss_coeff = 0.01
    cfg.model.dsa_indexer_use_sparse_loss = True
    cfg.model.use_fused_mhc = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["mla_up_proj", "mhc"]
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = []

    opt_cfg, scheduler_cfg = distributed_muon_with_cosine_annealing(
        muon_momentum=0.9,
        muon_use_nesterov=False,
        muon_scale_mode="spectral",
        muon_fp32_matmul_prec="medium",
        muon_num_ns_steps=5,
        muon_tp_mode="blockwise",
        muon_extra_scale_factor=1.0,
        muon_scalar_optimizer="adam",
        lr_warmup_iters=10,
        lr_decay_iters=50,
        max_lr=3.9e-6,
        min_lr=3.9e-7,
        weight_decay=0.1,
        clip_grad=1.0,
    )
    opt_cfg.optimizer = "muon"
    opt_cfg.use_distributed_optimizer = False
    opt_cfg.use_layer_wise_distributed_optimizer = True
    opt_cfg.use_layer_wise_param_layout = False
    opt_cfg.overlap_param_gather = True
    opt_cfg.muon_coefficient_type = "quintic"
    opt_cfg.adam_beta1 = 0.9
    opt_cfg.adam_beta2 = 0.95
    opt_cfg.main_grads_dtype = torch.float32
    opt_cfg.main_params_dtype = torch.float32
    opt_cfg.exp_avg_dtype = torch.float32
    opt_cfg.exp_avg_sq_dtype = torch.float32
    opt_cfg.use_precision_aware_optimizer = False
    scheduler_cfg.lr_warmup_init = 3.9e-7
    scheduler_cfg.start_weight_decay = 0.1
    scheduler_cfg.end_weight_decay = 0.1

    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    _benchmark_common(cfg, cross_entropy_impl="native")

    # Restore the sample-based schedule after applying the benchmark defaults.
    cfg.train.train_iters = None
    cfg.train.train_samples = 585_937_500
    cfg.scheduler.lr_decay_iters = None
    cfg.scheduler.lr_decay_samples = 584_765_624
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_warmup_samples = 1_536_000
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.train.exit_interval = 30
    cfg.train.manual_gc_interval = 10
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.cuda_graph_warmup_steps = 1
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True
    cfg.model.quant_recipe = None

    cfg.mixed_precision.fp8_param_gather = True
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = True
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.comm_overlap.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_param_gather = True
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.optimizer.optimizer_offload_fraction = 1.0
    cfg.optimizer.barrier_with_L1_time = True

    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "NCCL_BUFFSIZE": 4_194_304,
        "NCCL_GRAPH_REGISTER": 0,
        "NCCL_NET_GDR_C2C": 1,
        "NCCL_NET_GDR_LEVEL": "PHB",
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_CPU_OFFLOAD_V1": 0,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FUSED_ATTN": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
    }
    return cfg


def deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config() -> ConfigContainer:
    """DeepSeek V4 Pro full model: 64K THD/CP training on 256 GB300 GPUs."""
    cfg = deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_config()

    cfg.model.num_layers = 61
    cfg.model.moe_layer_freq = [1] * cfg.model.num_layers
    cfg.model.csa_compress_ratios = [128, 128, 4, *([128, 4] * 29), 0]
    cfg.model.moe_n_hash_layers = 3
    cfg.model.activation_func_clamp_value = 10.0
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = "Et*4|(t*4|)*14tmL"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["mla_up_proj", "mhc"]

    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["core_attn", "attn_proj"]
    cfg.model.fine_grained_offloading_max_inflight_offloads = 2
    cfg.env_vars.update(
        {
            "CUDA_DEVICE_MAX_CONNECTIONS": 32,
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
            "NVLINK_DOMAIN_SIZE": 72,
            "USE_MNNVL": 1,
            "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
            "NVTE_CPU_OFFLOAD_V1": 1,
            "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        }
    )
    return cfg
