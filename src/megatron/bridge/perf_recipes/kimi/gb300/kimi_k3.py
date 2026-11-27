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
"""Experimental native-Hybrid Kimi K3 8K MXFP8 proxy benchmark for 64 GB300 GPUs.

Requires the pinned Megatron-Core dev stack, MoK, FLA Attention Residuals,
and a compatible Emerging-Optimizers installation. These random-init mock-data
benchmarks force balanced expert routing and do not establish convergence.
"""

from dataclasses import fields

from megatron.bridge.models.hybrid import BridgeHybridMLAModelConfig
from megatron.bridge.models.transformer_config import MLATransformerConfig
from megatron.bridge.perf_recipes._common import _benchmark_common, _perf_precision
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.config import ConfigContainer, MockGPTDatasetConfig
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.utils.activation_map import str_to_callable


_PROXY_PATTERN = "KEKEKE+E" * 6


def kimi_k3_pretrain_64gpu_gb300_fp8mx_proxy_config() -> ConfigContainer:
    """Build a full-width, 24-block Kimi K3 proxy with PP1/EP64 and no MTP.

    Requires the pinned MCore dev stack. Uses mock data and forced balanced
    routing to validate execution and measure this reduced-depth workload.
    The ordinary PP1 schedule does not emulate PP4's in-flight microbatches
    or establish full-model memory fit, throughput or convergence.
    """
    required = {"enable_attention_residuals", "attn_res_norm_fusion", "moe_megakernel_backend"}
    if not required.issubset({f.name for f in fields(MLATransformerConfig)}):
        raise RuntimeError("Kimi K3 performance recipes require scripts/switch_mcore.sh dev and its dependencies.")

    pattern = _PROXY_PATTERN
    transformer = MLATransformerConfig(
        num_layers=len(pattern),
        hidden_size=7168,
        ffn_hidden_size=33792,
        num_attention_heads=96,
        kv_channels=64,
        init_method_std=0.02,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        expert_model_parallel_size=64,
        expert_tensor_parallel_size=1,
        context_parallel_size=1,
        max_seq_length_per_sample=8192,
        sequence_parallel=False,
        transformer_impl="transformer_engine",
        is_hybrid_model=True,
        attention_backend="fused",
        experimental_attention_variant="kda",
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=96,
        linear_num_value_heads=96,
        kda_f_lora_rank=128,
        kda_gate_lora_rank=None,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        multi_latent_attention=True,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        qk_layernorm=True,
        attention_latent_norm_epsilon=1.0e-6,
        attention_output_gate=True,
        gated_attention_proj_granularity="elementwise",
        attention_output_gate_fusion=True,
        mla_down_proj_fusion=False,
        no_rope_freq=1,
        enable_attention_residuals=True,
        attn_res_block_layers=24,
        attn_res_impl="fla",
        attn_res_norm_fusion=True,
        gdn_pre_gated_delta_rule_fusion=True,
        normalization="RMSNorm",
        layernorm_epsilon=1.0e-5,
        gated_linear_unit=True,
        activation_func=str_to_callable("situlu"),
        situ_glu_beta1=4.0,
        situ_glu_beta2=25.0,
        add_bias_linear=False,
        num_moe_experts=896,
        moe_ffn_hidden_size=3072,
        moe_latent_size=3584,
        moe_latent_up_projection_rmsnorm=True,
        moe_shared_expert_intermediate_size=6144,
        moe_router_topk=16,
        moe_router_score_function="sigmoid",
        moe_router_dtype="fp32",
        moe_router_pre_softmax=False,
        moe_router_topk_scaling_factor=1.0,
        moe_router_load_balancing_type="quantile_balancing",
        moe_aux_loss_coeff=0.0,
        moe_router_quantile_balancing_estimation_scope="global_batch",
        moe_router_qb_num_bins=1000,
        moe_router_enable_expert_bias=False,
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_enable_routing_replay=False,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_hybridep_num_sms=32,
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_megakernel_backend="mok",
        moe_megakernel_backend_config={"shared_expert_mode": "native"},
        moe_router_padding_for_quantization=True,
        recompute_granularity="selective",
        recompute_modules=["gdn", "moe", "mlp", "mla_up_proj"],
        mtp_num_layers=None,
        mtp_loss_scaling_factor=0.1,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gradient_accumulation_fusion=True,
    )
    cfg = _pretrain_common()
    cfg.model = BridgeHybridMLAModelConfig(
        transformer=transformer,
        hybrid_layer_pattern=pattern,
        vocab_size=163840,
        make_vocab_size_divisible_by=1280,
        should_pad_vocab=False,
        seq_length=8192,
        position_embedding_type="none",
        share_embeddings_and_output_weights=False,
    )
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _benchmark_common(cfg, cross_entropy_impl="native")
    # Preserve the original BF16 run's FP32 gradient reduction policy.
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.dist.enable_megatron_core_experimental = True
    cfg.dist.distributed_timeout_minutes = 60
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc_interval = 10
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 200
    cfg.optimizer.optimizer = "muon"
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.use_layer_wise_param_layout = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.overlap_param_gather = True
    cfg.optimizer.muon_momentum = 0.95
    cfg.optimizer.muon_scale_mode = "spectral"
    cfg.optimizer.muon_extra_scale_factor = 1.0
    cfg.optimizer.muon_num_ns_steps = 5
    cfg.optimizer.muon_coefficient_type = "quintic"
    cfg.optimizer.muon_split_qkv_per_head = True
    cfg.optimizer.muon_tp_mode = "duplicated"
    cfg.optimizer.muon_fp32_matmul_prec = "medium"
    cfg.optimizer.muon_nesterov = False
    cfg.optimizer.optimizer_cpu_offload = False
    cfg.optimizer.optimizer_offload_fraction = 0.0
    cfg.optimizer.overlap_cpu_optimizer_d2h_h2d = False
    cfg.optimizer.chunked_optimizer_state_offload = True
    cfg.optimizer.optimizer_state_offload_chunk_size_mb = 256
    cfg.optimizer.optimizer_state_offload_fraction = 1.0
    cfg.optimizer.lr = 1.0e-5
    cfg.optimizer.min_lr = 1.0e-6
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.999
    cfg.optimizer.adam_eps = 1.0e-8
    cfg.optimizer.clip_grad = 1.0
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.scheduler.lr_decay_iters = 50
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_warmup_init = 1.0e-6
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.weight_decay_incr_style = "constant"
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.comm_overlap = None
    cfg.dataset = MockGPTDatasetConfig(
        seq_length=8192,
        random_seed=20260901,
        split="100,0,0",
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        num_dataset_builder_threads=1,
        data_sharding=True,
        dataloader_type="single",
        create_attention_mask=False,
        skip_getting_attention_mask_from_dataset=True,
    )
    cfg.tokenizer = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=163840, use_tokenizer_vocab_size=False)
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.logger.log_memory_interval = 1
    cfg.logger.log_device_memory_used = True
    cfg.rng.seed = 20260901
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_GRAPH_REGISTER": 0,
        "NCCL_DEBUG": "VERSION",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
    }
    return cfg
