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

"""H100 library recipes for Nemotron 3 and 3.5 Nano."""

from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.nemotronh._nemotron_3_nano import (
    _nemotron_3_nano_peft_reference_config,
    _nemotron_3_nano_pretrain_reference_config,
    _nemotron_3_nano_sft_reference_config,
)
from megatron.bridge.recipes.utils.dataset_utils import default_openmathinstruct2_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


_NEMOTRON_3_NANO_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
# Placeholder until the public Nemotron 3.5 Nano repository is released.
_NEMOTRON_3_5_NANO_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16"
_OPENMATHINSTRUCT2_REVISION = "469216e3f46f4dacf476b382e192485ea51a143e"  # pragma: allowlist secret


def nemotron_3_nano_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Nemotron 3 Nano pretraining config for eight H100 GPUs.

    TP8 retains the perf recipe's PP1/CP1/EP8/ETP1 HybridEP topology,
    selective recompute, and native vocab-parallel cross entropy. TP
    communication overlap is disabled because its persistent userbuffers
    exhaust checkpoint-restore headroom on 80 GB H100s. The compiled native
    cross-entropy wrapper is disabled because its temporary workspace does not
    fit after FP32 optimizer-state allocation; the underlying native loss is
    unchanged. Unused CUDA cache is released after each optimizer step so the
    first lazy MoE metric collective can allocate after checkpoint resume.
    Validation uses microbatch one without changing its global batch. CUDA
    graphs remain disabled to preserve general-training headroom.

    Returns:
        H100 BF16 pretraining configuration.
    """
    cfg = _nemotron_3_nano_pretrain_reference_config()

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = 16
    cfg.model.moe_shared_expert_overlap = False

    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["moe", "layernorm"]
    cfg.model.cross_entropy_loss_fusion = False

    cfg.comm_overlap.tp_comm_overlap = False
    cfg.train.empty_unused_memory_level = 2
    cfg.validation.eval_micro_batch_size = 1

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def nemotron_3_5_nano_pretrain_config() -> ConfigContainer:
    """Return the Nemotron 3.5 Nano BF16 pretraining config."""
    cfg = _nemotron_3_nano_pretrain_reference_config()

    # Preserve the current Nemotron 3.5 H100 execution policy independently
    # from the memory-safe eight-GPU Nemotron 3 recipe above.
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.sequence_parallel = False
    # Split the 8K sequence across two ranks so each MTP head materializes only
    # half of its vocabulary-loss workspace on an 80-GiB H100. P2P retains the
    # fused-attention path for this model's grouped-query layout.
    cfg.model.context_parallel_size = 2
    cfg.model.cp_comm_type = "p2p"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.moe_flex_dispatcher_num_sms = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.apply_rope_fusion = True
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe", "layernorm", "core_attn"]
    set_cuda_graph_modules(cfg.model, ["mamba"])
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True
    cfg.mixed_precision = get_mixed_precision_config(cfg.mixed_precision)
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.check_for_large_grads = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True
    cfg.checkpoint.async_save = False
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.hf_model_id = _NEMOTRON_3_5_NANO_MODEL_ID
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_5_NANO_MODEL_ID
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
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


def _apply_h100_finetune_execution_config(cfg: ConfigContainer) -> None:
    """Apply the evidenced H100 packed-finetuning execution contract."""
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = 16
    cfg.model.moe_router_force_load_balancing = False


def nemotron_3_nano_sft_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Nemotron 3 Nano SFT config for eight H100 GPUs.

    Packed SFT retains the established DeepEP dispatcher and eager execution.
    TP4 leaves room for full optimizer state and checkpointing.

    Returns:
        H100 BF16 SFT configuration.
    """
    cfg = _nemotron_3_nano_sft_reference_config()
    _apply_h100_finetune_execution_config(cfg)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.sequence_parallel = True

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def nemotron_3_5_nano_sft_config() -> ConfigContainer:
    """Return a full SFT config for Nemotron 3.5 Nano."""
    cfg = _nemotron_3_nano_sft_reference_config()
    _apply_h100_finetune_execution_config(cfg)
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.hf_model_id = _NEMOTRON_3_5_NANO_MODEL_ID
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_5_NANO_MODEL_ID
    return cfg


def nemotron_3_5_nano_sft_openmathinstruct2_packed_config() -> ConfigContainer:
    """Return the verified 4K packed OpenMathInstruct-2 SFT config."""
    cfg = nemotron_3_5_nano_sft_config()

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe", "layernorm", "core_attn", "mlp"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None

    cfg.dataset = default_openmathinstruct2_config(
        seq_length=4096,
        enable_offline_packing=True,
        pad_seq_to_mult=2,
    )
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _OPENMATHINSTRUCT2_REVISION}
    if cfg.dataset.offline_packing_specs is not None:
        cfg.dataset.offline_packing_specs.tokenizer_model_name = _NEMOTRON_3_5_NANO_MODEL_ID

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


def nemotron_3_nano_peft_8gpu_h100_bf16_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Return the Nemotron 3 Nano PEFT config for eight H100 GPUs.

    Args:
        peft_scheme: PEFT scheme, or a custom PEFT instance.

    Returns:
        H100 BF16 PEFT configuration.
    """
    cfg = _nemotron_3_nano_peft_reference_config(peft_scheme=peft_scheme)
    _apply_h100_finetune_execution_config(cfg)

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def nemotron_3_5_nano_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Nemotron 3.5 Nano."""
    cfg = nemotron_3_nano_peft_8gpu_h100_bf16_config(peft_scheme)
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.hf_model_id = _NEMOTRON_3_5_NANO_MODEL_ID
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_5_NANO_MODEL_ID
    return cfg


__all__ = [
    "nemotron_3_5_nano_peft_config",
    "nemotron_3_5_nano_pretrain_config",
    "nemotron_3_5_nano_sft_config",
    "nemotron_3_5_nano_sft_openmathinstruct2_packed_config",
    "nemotron_3_nano_peft_8gpu_h100_bf16_config",
    "nemotron_3_nano_pretrain_8gpu_h100_bf16_config",
    "nemotron_3_nano_sft_8gpu_h100_bf16_config",
]
