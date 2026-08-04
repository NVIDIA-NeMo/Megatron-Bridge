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

"""Megatron Bridge 0.5 recipes for NVIDIA Nemotron 3.5 Lightning."""

from __future__ import annotations

import torch

from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_peft_config,
    nemotron_3_nano_pretrain_config,
    nemotron_3_nano_sft_config,
)
from megatron.bridge.recipes.utils.finetune_utils import default_openmathinstruct2_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


NEMOTRON_3_5_LIGHTNING_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
NEMOTRON_3_5_LIGHTNING_HF_MODEL_REVISION = (
    "b3caaabed0263651a17dc1f2d4ce97e794f76c44"  # pragma: allowlist secret
)
OPENMATHINSTRUCT2_REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"  # pragma: allowlist secret


def _apply_lightning_checkpoint(cfg: ConfigContainer) -> None:
    """Apply the public Lightning identity and repeated-layer MTP contract."""
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.hf_model_id = NEMOTRON_3_5_LIGHTNING_HF_MODEL_ID
    cfg.model.hf_model_revision = NEMOTRON_3_5_LIGHTNING_HF_MODEL_REVISION
    cfg.tokenizer.tokenizer_model = NEMOTRON_3_5_LIGHTNING_HF_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": NEMOTRON_3_5_LIGHTNING_HF_MODEL_REVISION}


def nemotron_3_5_lightning_pretrain_config() -> ConfigContainer:
    """Return the verified H100 BF16 pretraining config for Lightning."""
    cfg = nemotron_3_nano_pretrain_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 2
    cfg.model.cp_comm_type = "p2p"
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 8192

    cfg.dataset.seq_length = 8192
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8
    cfg.dataset.mmap_bin_files = False

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.moe_shared_expert_overlap = False

    cfg.train.train_iters = 39735
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["mamba"])
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.model.attention_backend = "fused"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe", "layernorm", "core_attn"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.mixed_precision = get_mixed_precision_config(cfg.mixed_precision)
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    cfg.checkpoint.async_save = False
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.check_for_large_grads = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = False
    cfg.rerun_state_machine.check_for_nan_in_loss = True

    cfg.model.init_method_std = 0.0173
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_weighted_squared_relu = True
    cfg.model.calculate_per_token_loss = True
    _apply_lightning_checkpoint(cfg)
    return cfg


def nemotron_3_5_lightning_sft_config() -> ConfigContainer:
    """Return the Lightning full-SFT config."""
    cfg = nemotron_3_nano_sft_config()
    _apply_lightning_checkpoint(cfg)
    return cfg


def nemotron_3_5_lightning_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return the Lightning PEFT config."""
    cfg = nemotron_3_nano_peft_config(peft_scheme)
    _apply_lightning_checkpoint(cfg)
    return cfg


def nemotron_3_5_lightning_sft_openmathinstruct2_packed_config() -> ConfigContainer:
    """Return the verified 4K packed OpenMathInstruct-2 SFT config."""
    cfg = nemotron_3_5_lightning_sft_config()

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe", "layernorm", "core_attn", "mlp"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None

    cfg.dataset = default_openmathinstruct2_config(seq_length=4096, packed_sequence=True, pad_seq_to_mult=2)
    cfg.dataset.hf_kwargs = {"revision": OPENMATHINSTRUCT2_REVISION}
    if cfg.dataset.packed_sequence_specs is not None:
        cfg.dataset.packed_sequence_specs.tokenizer_model_name = NEMOTRON_3_5_LIGHTNING_HF_MODEL_ID

    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.empty_unused_memory_level = 2

    cfg.mixed_precision = get_mixed_precision_config(cfg.mixed_precision)
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.optimizer.lr = 5e-6
    cfg.optimizer.min_lr = 0.0
    cfg.optimizer.overlap_param_gather = False
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 0

    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.checkpoint.async_save = False
    cfg.checkpoint.save_interval = 100

    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.logger.tensorboard_dir = None
    cfg.ddp.average_in_collective = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_large_grads = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True
    cfg.dist.distributed_timeout_minutes = 120
    return cfg


def nemotron_3_5_lightning_pretrain_8k_config() -> ConfigContainer:
    """Return the verified GB200 BF16 8K pretraining config."""
    cfg = nemotron_3_5_lightning_pretrain_config()
    cfg.train.micro_batch_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = None
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.cuda_graph_impl = "none"
    set_cuda_graph_modules(cfg.model, [])
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.ddp.average_in_collective = False
    cfg.checkpoint.save_interval = 50
    return cfg


def nemotron_3_5_lightning_pretrain_8k_fsdp_config() -> ConfigContainer:
    """Return the verified GB200 BF16 8K Megatron-FSDP config."""
    cfg = nemotron_3_5_lightning_pretrain_8k_config()
    cfg.dist.use_megatron_fsdp = True
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.num_distributed_optimizer_instances = 1
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.outer_dp_sharding_strategy = "no_shard"
    cfg.ddp.megatron_fsdp_main_params_dtype = torch.float32
    cfg.ddp.megatron_fsdp_main_grads_dtype = torch.float32
    cfg.ddp.megatron_fsdp_grad_comm_dtype = torch.bfloat16
    cfg.checkpoint.load = None
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    return cfg


def nemotron_3_5_lightning_sft_openmathinstruct2_packed_tp1_config() -> ConfigContainer:
    """Return the optimized GB200 TP1 packed OpenMathInstruct-2 SFT config."""
    cfg = nemotron_3_5_lightning_sft_openmathinstruct2_packed_config()
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.train.empty_unused_memory_level = 0
    cfg.ddp.overlap_param_gather = True
    cfg.optimizer.overlap_param_gather = True
    return cfg


__all__ = [
    "nemotron_3_5_lightning_peft_config",
    "nemotron_3_5_lightning_pretrain_8k_config",
    "nemotron_3_5_lightning_pretrain_8k_fsdp_config",
    "nemotron_3_5_lightning_pretrain_config",
    "nemotron_3_5_lightning_sft_config",
    "nemotron_3_5_lightning_sft_openmathinstruct2_packed_config",
    "nemotron_3_5_lightning_sft_openmathinstruct2_packed_tp1_config",
]
