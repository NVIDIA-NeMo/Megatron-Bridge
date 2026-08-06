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
"""GB200 recipes for GLM-5.2."""

from __future__ import annotations

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import ChatSFTPreprocessingConfig
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config, default_tulu3_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_GLM52_MODEL_ID = "zai-org/GLM-5.2"
_GLM52_MODEL_REVISION = "4d67f66cc64d3219133b767c253b2ad1425c6c88"  # pragma: allowlist secret
_TULU3_REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"  # pragma: allowlist secret
_GLM52_PP6_128K_LAYOUT = "|".join(("E" + "t" * 14, "t" * 16, "t" * 12, "t" * 12, "t" * 12, "t" * 12 + "mL"))


def glm52_pretrain_192gpu_gb200_bf16_config() -> ConfigContainer:
    """GLM-5.2 bounded pretraining on 192 GB200 GPUs."""
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(_GLM52_MODEL_ID, revision=_GLM52_MODEL_REVISION).to_megatron_provider(
        load_weights=False
    )
    cfg.tokenizer.tokenizer_model = _GLM52_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _GLM52_MODEL_REVISION}

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.num_layers_in_first_pipeline_stage = 14
    cfg.model.num_layers_in_last_pipeline_stage = 16
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.microbatch_group_size_per_vp_stage = 6

    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.mtp_num_layers = 1
    cfg.model.calculate_per_token_loss = True
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.persist_layer_norm = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.bias_activation_fusion = True
    cfg.model.apply_rope_fusion = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 1024
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.logger.log_interval = 1

    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8
    cfg.dataset.random_seed = 1234
    cfg.rng.seed = 1234
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=40,
        max_lr=3e-4,
        min_lr=3e-5,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.load = None
    cfg.env_vars = {**COMMON_RECIPE_ENV_VARS}

    return cfg


def glm52_sft_192gpu_gb200_bf16_config() -> ConfigContainer:
    """GLM-5.2 bounded full SFT on 192 GB200 GPUs."""
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_GLM52_MODEL_ID, revision=_GLM52_MODEL_REVISION).to_megatron_provider(
        load_weights=False
    )
    cfg.tokenizer.tokenizer_model = _GLM52_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _GLM52_MODEL_REVISION}
    cfg.model.seq_length = 8192
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.context_parallel_size = 4
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.num_layers_in_first_pipeline_stage = 14
    cfg.model.num_layers_in_last_pipeline_stage = 16
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.microbatch_group_size_per_vp_stage = 6

    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.mtp_num_layers = 1
    cfg.model.calculate_per_token_loss = True
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.deallocate_pipeline_outputs = True
    cfg.model.persist_layer_norm = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.bias_activation_fusion = True
    cfg.model.apply_rope_fusion = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.average_in_collective = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.logger.log_interval = 1

    packing_alignment = 2 * cfg.model.context_parallel_size
    cfg.dataset = default_tulu3_config(
        seq_length=cfg.model.seq_length,
        enable_offline_packing=True,
        pad_seq_to_mult=packing_alignment,
    )
    cfg.dataset.hf_dataset.split = "train[:10000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _TULU3_REVISION}
    cfg.dataset.hf_output_root = "work/data/glm5-2/tulu3-full-sft-gb200-8k-v5"
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.max_train_samples = 10000
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.max_single_sequence_length = cfg.model.seq_length - packing_alignment
    # HybridEP needs a fixed token width; CUDA graphs are disabled, so cu_seqlens can remain dynamic.
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=5e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "USE_MNNVL": 1,
    }
    return cfg


def glm52_sft_192gpu_gb200_bf16_128k_config() -> ConfigContainer:
    """GLM-5.2 128K packed SFT with context parallelism on 192 GB200 GPUs."""
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_GLM52_MODEL_ID, revision=_GLM52_MODEL_REVISION).to_megatron_provider(
        load_weights=False
    )
    cfg.tokenizer.tokenizer_model = _GLM52_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _GLM52_MODEL_REVISION}
    cfg.model.seq_length = 131072
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = _GLM52_PP6_128K_LAYOUT
    cfg.model.context_parallel_size = 32
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.microbatch_group_size_per_vp_stage = None

    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.mtp_num_layers = 1
    cfg.model.calculate_per_token_loss = True
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.deallocate_pipeline_outputs = True
    cfg.model.persist_layer_norm = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.bias_activation_fusion = True
    cfg.model.apply_rope_fusion = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.average_in_collective = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.train.train_iters = 20
    cfg.train.global_batch_size = 56
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.logger.log_interval = 1

    cfg.dataset = default_tulu3_config(
        seq_length=131072,
        enable_offline_packing=True,
        pad_seq_to_mult=64,
    )
    cfg.dataset.hf_dataset = None
    cfg.dataset.dataset_root = "work/data/glm5-2/synthetic-long-sft-128k"
    cfg.dataset.hf_output_root = None
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.max_train_samples = 1120
    cfg.dataset.seed = 1234
    cfg.dataset.preprocessing = ChatSFTPreprocessingConfig()
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.tokenizer_model_name = "glm5"
    # HybridEP needs a fixed token width; CUDA graphs are disabled, so cu_seqlens can remain dynamic.
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    cfg.rng.seed = 5678
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=2,
        max_lr=1e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 20
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "USE_MNNVL": 1,
    }
    return cfg


def glm52_peft_192gpu_gb200_bf16_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """GLM-5.2 bounded PEFT on 192 GB200 GPUs."""
    cfg = _peft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_GLM52_MODEL_ID, revision=_GLM52_MODEL_REVISION).to_megatron_provider(
        load_weights=False
    )
    cfg.tokenizer.tokenizer_model = _GLM52_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _GLM52_MODEL_REVISION}
    cfg.model.seq_length = 2048
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.num_layers_in_first_pipeline_stage = 14
    cfg.model.num_layers_in_last_pipeline_stage = 16
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.microbatch_group_size_per_vp_stage = 6

    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.mtp_num_layers = 1
    cfg.model.calculate_per_token_loss = True
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.persist_layer_norm = True
    cfg.model.bias_dropout_fusion = True
    cfg.model.bias_activation_fusion = True
    cfg.model.apply_rope_fusion = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.average_in_collective = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.logger.log_interval = 1

    peft_cfg = default_peft_config(peft_scheme)
    if isinstance(peft_scheme, str) and peft_scheme.lower() in {"lora", "dora"}:
        peft_cfg.dim = 8
        peft_cfg.alpha = 16
        peft_cfg.dropout = 0.0
        peft_cfg.target_modules = [
            "linear_q_down_proj",
            "linear_q_up_proj",
            "linear_kv_down_proj",
            "linear_kv_up_proj",
            "linear_proj",
        ]
    cfg.peft = peft_cfg

    cfg.dataset = default_tulu3_config(
        seq_length=2048,
        enable_offline_packing=True,
        pad_seq_to_mult=4,
    )
    cfg.dataset.hf_dataset.split = "train[:10000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _TULU3_REVISION}
    cfg.dataset.hf_output_root = "work/data/glm5-2/tulu3-peft-gb200"
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.max_train_samples = 10000
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False

    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 32
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=1e-4,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.env_vars = {**COMMON_RECIPE_ENV_VARS}
    return cfg


__all__ = [
    "glm52_peft_192gpu_gb200_bf16_config",
    "glm52_pretrain_192gpu_gb200_bf16_config",
    "glm52_sft_192gpu_gb200_bf16_128k_config",
    "glm52_sft_192gpu_gb200_bf16_config",
]
