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

"""Hardware-neutral Nemotron 3 Nano recipe builders."""

import torch
from megatron.core.activations import squared_relu

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.cuda_graph import clear_cuda_graph_modules


def _nemotron_3_nano_pretrain_reference_config() -> ConfigContainer:
    """Build the shared Nemotron 3 Nano pre-training contract.

    Hardware-specific library and performance recipes apply their execution
    settings to this config without changing its training semantics.

    Returns:
        ConfigContainer: Pre-training configuration for Nemotron 3 Nano.
    """
    cfg = _pretrain_common()

    cfg.model = HybridModelProvider(
        hybrid_layer_pattern="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        num_layers=52,
        hidden_size=2688,
        mamba_num_heads=64,
        kv_channels=128,
        mamba_state_dim=128,
        ffn_hidden_size=1856,
        num_attention_heads=32,
        mamba_head_dim=64,
        seq_length=8192,
        num_query_groups=2,
        num_moe_experts=128,
        moe_ffn_hidden_size=1856,
        moe_shared_expert_intermediate_size=3712,
        moe_router_topk=6,
        moe_router_topk_scaling_factor=2.5,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        mamba_num_groups=8,
        make_vocab_size_divisible_by=128,
        activation_func=squared_relu,
        masked_softmax_fusion=True,
        apply_query_key_layer_scaling=False,
        persist_layer_norm=True,
        attention_softmax_in_fp32=False,
        first_last_layers_bf16=True,
        is_hybrid_model=True,
        moe_aux_loss_coeff=0.0001,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=True,
        moe_shared_expert_overlap=True,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=8,
    )

    cfg.tokenizer.tokenizer_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    cfg.dataset.seq_length = 8192
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8
    cfg.dataset.mmap_bin_files = False

    cfg.model.pipeline_model_parallel_layout = None

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    cfg.train.train_iters = 39735
    cfg.train.global_batch_size = 3072
    cfg.train.micro_batch_size = 2
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    cfg.model.transformer_impl = "transformer_engine"

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = "fused"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    cfg.model.moe_router_padding_for_fp8 = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.optimizer.lr = 1.6e-3
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.min_lr = 1.6e-5
    cfg.scheduler.lr_warmup_iters = 333

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=True,
    )
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = False

    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    cfg.model.moe_router_force_load_balancing = False

    cfg.model.init_method_std = 0.0173
    cfg.model.apply_rope_fusion = False
    cfg.model.use_fused_weighted_squared_relu = True

    return cfg


def _nemotron_3_nano_finetune_model() -> HybridModelProvider:
    """Build the shared Nemotron 3 Nano model contract for SFT and PEFT."""
    return HybridModelProvider(
        hybrid_layer_pattern="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        num_layers=52,
        hidden_size=2688,
        mamba_num_heads=64,
        kv_channels=128,
        mamba_state_dim=128,
        ffn_hidden_size=1856,
        num_attention_heads=32,
        mamba_head_dim=64,
        seq_length=2048,
        num_query_groups=2,
        num_moe_experts=128,
        moe_ffn_hidden_size=1856,
        moe_shared_expert_intermediate_size=3712,
        moe_router_topk=6,
        moe_router_topk_scaling_factor=2.5,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        mamba_num_groups=8,
        make_vocab_size_divisible_by=128,
        activation_func=squared_relu,
        masked_softmax_fusion=True,
        apply_query_key_layer_scaling=False,
        persist_layer_norm=True,
        attention_softmax_in_fp32=False,
        first_last_layers_bf16=True,
        is_hybrid_model=True,
        moe_aux_loss_coeff=0.0001,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=True,
        moe_shared_expert_overlap=True,
        apply_rope_fusion=False,
        attention_backend="fused",
        init_method_std=0.0173,
        use_fused_weighted_squared_relu=True,
        calculate_per_token_loss=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=8,
    )


def _apply_nemotron_3_nano_finetune_defaults(cfg: ConfigContainer) -> None:
    """Apply hardware-neutral SFT/PEFT semantics and safe execution defaults."""
    cfg.model = _nemotron_3_nano_finetune_model()
    cfg.model.pipeline_model_parallel_layout = None

    # DeepEP is the established packed-finetuning backend. Hardware wrappers
    # may override it only when a matching packed-workload reference is proven.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = None

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    clear_cuda_graph_modules(cfg.model)
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.model.attention_backend = "fused"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.model.moe_router_padding_for_fp8 = False
    cfg.model.moe_router_force_load_balancing = False

    cfg.validation.eval_interval = 500
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.offline_packing_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "cosine"

    cfg.tokenizer.tokenizer_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.ckpt_assume_constant_structure = True

    cfg.logger.log_interval = 10
    cfg.logger.log_timers_to_tensorboard = False
    cfg.rng.seed = 1234

    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True


def _nemotron_3_nano_sft_reference_config() -> ConfigContainer:
    """Build the hardware-neutral Nemotron 3 Nano SFT contract."""
    cfg = _sft_common()
    _apply_nemotron_3_nano_finetune_defaults(cfg)
    return cfg


def _nemotron_3_nano_peft_reference_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Build the hardware-neutral Nemotron 3 Nano PEFT contract."""
    cfg = _peft_common()
    _apply_nemotron_3_nano_finetune_defaults(cfg)

    target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme, target_modules=target_modules)
    elif isinstance(peft_scheme, PEFT):
        cfg.peft = peft_scheme
    else:
        cfg.peft = LoRA(
            target_modules=target_modules,
            dim=32,
            alpha=32,
            dropout=0.0,
            dropout_position="pre",
            lora_A_init_method="xavier",
            lora_B_init_method="zero",
        )

    return cfg
