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

"""Nemotron 3.5 Super VL training recipes."""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.nemotron_omni.h100.nemotron_omni import (
    _make_nemotron_omni_energon_dataset,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_super import (
    _apply_nemotron_3_super_64gpu_h100_training_stack,
)
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


NEMOTRON_35_SUPER_VL_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Super-120B-A12B-SourceOfTruth"


def _nemotron_35_super_vl_base() -> ConfigContainer:
    """Create the model and training base owned by Nemotron 3.5 Super VL."""
    cfg = _sft_common_vlm()
    cfg.model = AutoBridge.from_hf_pretrained(
        NEMOTRON_35_SUPER_VL_HF_MODEL_ID,
        trust_remote_code=True,
    ).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True

    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False
    cfg.model.freeze_language_model = False
    cfg.model.freeze_sound_encoder = True
    cfg.model.freeze_sound_projection = False

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.attention_backend = "flash"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None

    cfg.train.train_iters = 2000
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.validation.eval_interval = 200
    cfg.validation.eval_iters = 0

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=50,
        lr_decay_iters=None,
        max_lr=6e-6,
        min_lr=6e-7,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.checkpoint.save_interval = 200
    cfg.mixed_precision = "bf16_mixed"

    return cfg


def nemotron_35_super_vl_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """Return the 64-H100 BF16 pretraining config for Super VL.

    The language stack starts from the Nemotron 3 Super H100 recipe, then uses
    the measured Super-VL pipeline balance, selective recompute, and HybridEP
    policy needed by the trainable vision stack. It retains natural expert
    routing, numerical guards, and checkpointing for real-data training. All
    present model stacks remain trainable.

    Returns:
        The Super-VL pretraining configuration.
    """
    cfg = nemotron_35_super_vl_sft_64gpu_h100_bf16_config()
    cfg.train.train_iters = 100
    cfg.train.eval_iters = 0
    cfg.dataset.do_validation = False
    cfg.dataset.pad_to_max_length = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = cfg.train.train_iters
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.async_save = False
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.logger.log_device_memory_used = True
    cfg.logger.tensorboard_dir = None
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.check_for_large_grads = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.num_layers_in_first_pipeline_stage = 38
    cfg.model.num_layers_in_last_pipeline_stage = None
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.mamba_chunk_size = 128

    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["layernorm", "moe"]
    cfg.model.recompute_vision = True
    cfg.model.radio_force_eval_mode = False
    cfg.model.vision_recompute_granularity = "selective"
    cfg.model.vision_recompute_modules = ["core_attn"]
    cfg.model.vision_recompute_method = None
    cfg.model.vision_recompute_num_layers = None

    # Fixed-width MBS1 inputs are equal across DP ranks, so HybridEP can use
    # its dropless eager path without per-layer uneven-input padding. Do not
    # inherit the language performance recipe's expert-capacity token drops.
    cfg.model.moe_expert_capacity_factor = None
    cfg.model.moe_pad_expert_input_to_capacity = False
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = False
    cfg.model.moe_hybridep_assume_equal_dispatch_inputs = True
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.ddp.overlap_param_gather = False
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_hybridep_num_sms_preprocessing = 108
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_permute_fusion_into_hybridep = True
    cfg.model.use_fused_weighted_squared_relu = True
    cfg.model.overlap_moe_expert_parallel_comm = False
    cfg.model.delay_wgrad_compute = False
    cfg.model.overlap_p2p_comm = False
    cfg.model.batch_p2p_comm = True
    cfg.model.batch_p2p_sync = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.apply_rope_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.use_transformer_engine_op_fuser = False
    cfg.model.use_te_rng_tracker = False
    cfg.rng.te_rng_tracker = False

    cfg.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] = 0
    cfg.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] = 0
    cfg.checkpoint.async_save = False
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    return cfg


def nemotron_35_super_vl_sft_64gpu_h100_bf16_config() -> ConfigContainer:
    """Return the 64-H100 BF16 SFT configuration for Nemotron 3.5 Super VL.

    The model and audio-free image/video Energon data path reuse the Nemotron
    Omni stack. The language decoder reuses the measured Nemotron 3 Super
    TP1/PP2/EP32 HybridEP training stack without replacing this checkpoint's
    native one-layer MTP or separate temporal video embedder configuration.

    The Energon shard path must be set with ``dataset.path=<path>``. Samples
    may contain text, images, and videos, but must not contain audio.

    Returns:
        The Super-VL SFT configuration.
    """
    cfg = _nemotron_35_super_vl_base()
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False
    cfg.model.freeze_language_model = False
    cfg.model.freeze_sound_encoder = True
    cfg.model.freeze_sound_projection = True
    cfg.model.calculate_per_token_loss = True

    cfg.dataset = _make_nemotron_omni_energon_dataset(
        cfg.train.micro_batch_size,
        hf_processor_path=NEMOTRON_35_SUPER_VL_HF_MODEL_ID,
    )
    return _apply_nemotron_3_super_64gpu_h100_training_stack(cfg)


def nemotron_35_super_vl_peft_16gpu_h100_bf16_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Return the 16-H100 BF16 PEFT configuration for Nemotron 3.5 Super VL.

    The base language, image, projector, and temporal-video weights remain
    frozen. Adapters target the model-native attention, Mamba, and MLP linear
    projections inherited from Nemotron 3 Super. The Energon dataset preserves
    Omni's processor-sized image and temporal-video metadata contract.

    Args:
        peft_scheme: PEFT scheme, either ``"lora"``, ``"dora"``, or a
            configured PEFT instance.

    Returns:
        The Super-VL PEFT configuration.
    """
    cfg = _nemotron_35_super_vl_base()
    target_modules = ["linear_qkv", "linear_proj", "in_proj", "out_proj", "linear_fc1", "linear_fc2"]
    cfg.peft = default_peft_config(
        peft_scheme,
        target_modules=target_modules,
        dim=32,
        alpha=32,
        dropout=0.0,
        dropout_position="pre",
        lora_A_init_method="xavier",
        lora_B_init_method="zero",
    )

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.pipeline_model_parallel_layout = None

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.freeze_sound_encoder = True
    cfg.model.freeze_sound_projection = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_expert_capacity_factor = None
    cfg.model.moe_pad_expert_input_to_capacity = False
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_vision = False

    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 16
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        lr_decay_iters=100,
        max_lr=1e-4,
        min_lr=0.0,
        adam_beta2=0.95,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.dataset = _make_nemotron_omni_energon_dataset(
        cfg.train.micro_batch_size,
        hf_processor_path=NEMOTRON_35_SUPER_VL_HF_MODEL_ID,
    )
    cfg.dataset.pad_to_max_length = True
    cfg.dataset.do_validation = False

    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.async_save = False
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.logger.log_device_memory_used = True
    cfg.logger.tensorboard_dir = None
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.check_for_large_grads = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


__all__ = [
    "nemotron_35_super_vl_peft_16gpu_h100_bf16_config",
    "nemotron_35_super_vl_pretrain_64gpu_h100_bf16_config",
    "nemotron_35_super_vl_sft_64gpu_h100_bf16_config",
]
